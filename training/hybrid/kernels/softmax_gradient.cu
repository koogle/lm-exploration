// Cross-entropy statistics and the logit gradient, in place, one block per row.
//
// Pass one reduces each row of logits (bf16 or fp32) to its log-normalizer
// lse = max + log(sum exp(x - max)) with an online max/sum per thread, merged
// across warps. Pass two overwrites the row with the gradient of
// ce + z_loss * lse^2: softmax * (1 + 2 z_loss lse) - onehot(target), or zeros
// for an ignored target. The row is read twice and written once in its own dtype.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

constexpr int THREADS = 512;
constexpr int64_t IGNORE_INDEX = -100;

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

// Merge running (max, sum of exp(x - max)) pairs.
__device__ __forceinline__ void merge(float& m, float& s, float other_m, float other_s) {
  const float top = fmaxf(m, other_m);
  if (top == -INFINITY) return;
  s = s * expf(m - top) + other_s * expf(other_m - top);
  m = top;
}

__device__ __forceinline__ void add(float& m, float& s, float x) {
  if (x > m) {
    s = s * expf(m - x) + 1.0f;
    m = x;
  } else {
    s += expf(x - m);
  }
}

// Vectors of VEC elements (16 bytes when aligned, else 1).
template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Vector {
  T values[VEC];
};

template <typename T, int VEC>
__global__ void __launch_bounds__(THREADS)
    softmax_gradient_kernel(T* logits, const int64_t* targets, float* lse_out, float* target_out,
                            float z_loss, int vocab, int64_t stride) {
  __shared__ float shared_m[THREADS / 32], shared_s[THREADS / 32], row_lse;
  const int64_t row = blockIdx.x;
  T* base = logits + row * stride;
  auto* vectors = reinterpret_cast<Vector<T, VEC>*>(base);
  const int count = vocab / VEC;

  float m = -INFINITY, s = 0.0f;
  for (int i = threadIdx.x; i < count; i += THREADS) {
    const Vector<T, VEC> chunk = vectors[i];
#pragma unroll
    for (int j = 0; j < VEC; ++j) add(m, s, to_float(chunk.values[j]));
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    merge(m, s, __shfl_xor_sync(0xffffffff, m, offset), __shfl_xor_sync(0xffffffff, s, offset));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) {
    shared_m[warp] = m;
    shared_s[warp] = s;
  }
  __syncthreads();
  if (warp == 0) {
    m = lane < THREADS / 32 ? shared_m[lane] : -INFINITY;
    s = lane < THREADS / 32 ? shared_s[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      merge(m, s, __shfl_xor_sync(0xffffffff, m, offset), __shfl_xor_sync(0xffffffff, s, offset));
    if (lane == 0) {
      const float lse = m + logf(s);
      const int64_t target = targets[row];
      const bool valid = target != IGNORE_INDEX;
      row_lse = lse;
      lse_out[row] = lse;
      target_out[row] = to_float(base[valid ? target : 0]);
    }
  }
  __syncthreads();
  const float lse = row_lse;
  const int64_t target = targets[row];
  const bool valid = target != IGNORE_INDEX;
  const float scale = valid ? 1.0f + 2.0f * z_loss * lse : 0.0f;
  for (int i = threadIdx.x; i < count; i += THREADS) {
    Vector<T, VEC> chunk = vectors[i];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const int64_t column = static_cast<int64_t>(i) * VEC + j;
      float gradient = expf(to_float(chunk.values[j]) - lse) * scale;
      if (valid && column == target) gradient -= 1.0f;
      chunk.values[j] = from_float<T>(gradient);
    }
    vectors[i] = chunk;
  }
}

template <typename T>
void launch(torch::Tensor& logits, const torch::Tensor& targets, torch::Tensor& lse,
            torch::Tensor& target_logit, float z_loss) {
  const int rows = logits.size(0), vocab = logits.size(1);
  const int64_t stride = logits.stride(0);
  constexpr int WIDE = 16 / sizeof(T);
  auto* data = reinterpret_cast<T*>(logits.data_ptr());
  const bool aligned = vocab % WIDE == 0 && stride % WIDE == 0 &&
                       reinterpret_cast<uintptr_t>(data) % 16 == 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* t = targets.data_ptr<int64_t>();
  if (aligned) {
    softmax_gradient_kernel<T, WIDE><<<rows, THREADS, 0, stream>>>(
        data, t, lse.data_ptr<float>(), target_logit.data_ptr<float>(), z_loss, vocab, stride);
  } else {
    softmax_gradient_kernel<T, 1><<<rows, THREADS, 0, stream>>>(
        data, t, lse.data_ptr<float>(), target_logit.data_ptr<float>(), z_loss, vocab, stride);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

std::vector<torch::Tensor> softmax_gradient(torch::Tensor logits, torch::Tensor targets,
                                            double z_loss) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.stride(1) == 1,
              "logits must be a CUDA [rows, vocab] tensor contiguous along the vocabulary");
  TORCH_CHECK(targets.scalar_type() == torch::kInt64 && targets.is_contiguous() &&
                  targets.numel() == logits.size(0),
              "targets must be contiguous int64 [rows]");
  const c10::cuda::CUDAGuard guard(logits.device());
  auto stats = logits.options().dtype(torch::kFloat32);
  auto lse = torch::empty({logits.size(0)}, stats);
  auto target_logit = torch::empty_like(lse);
  if (logits.size(0) == 0) return {lse, target_logit};
  if (logits.scalar_type() == torch::kBFloat16) {
    launch<__nv_bfloat16>(logits, targets, lse, target_logit, static_cast<float>(z_loss));
  } else {
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be bfloat16 or float32");
    launch<float>(logits, targets, lse, target_logit, static_cast<float>(z_loss));
  }
  return {lse, target_logit};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("softmax_gradient", &softmax_gradient, "In-place cross-entropy logit gradient");
}
