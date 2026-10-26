// KDA's causal depthwise convolutions (width 4) with SiLU on q, k and v, in one launch.
//
// y[t, c] = silu(sum_s w[c, 3 - s] * x[t - s, c] * (position[t] >= s)), where
// position[t] is t's index within its document, so a token never reads an earlier
// document (row starts have position 0, so rows never mix either). Each thread owns
// 8 adjacent channels of one tensor and a strip of tokens, keeps the last three
// inputs in registers and reads every input once (plus a three-token halo).
// The backward recomputes the pre-activation, writes dx and accumulates dw in fp32
// per thread, then adds it atomically. Arithmetic is fp32; x, y and dx keep x's dtype.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

constexpr int WIDTH = 4;
constexpr int VEC = 8;
constexpr int STRIP = 64;
constexpr int THREADS = 64;

struct Tensors {
  const void* x[3];
  const float* w[3];
  void* out[3];
  const void* grad[3];
  float* dw[3];
};

__device__ __forceinline__ void load8(const float* p, float* v) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* v) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(pairs[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}
__device__ __forceinline__ void store8(float* p, const float* v) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}
__device__ __forceinline__ void store8(__nv_bfloat16* p, const float* v) {
  uint4 raw;
  __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) pairs[i] = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = raw;
}

__device__ __forceinline__ float sigmoid(float a) { return 1.0f / (1.0f + __expf(-a)); }

// Pre-activation at the newest token of `window` (window[3] is t, window[0] is t - 3).
__device__ __forceinline__ void pre_activation(const float (&w)[WIDTH][VEC],
                                               const float (&window)[WIDTH][VEC],
                                               int64_t position, float* pre) {
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    float acc = w[3][i] * window[3][i];
#pragma unroll
    for (int s = 1; s < WIDTH; ++s)
      if (position >= s) acc += w[3 - s][i] * window[3 - s][i];
    pre[i] = acc;
  }
}

template <typename T>
__device__ __forceinline__ void load_weights(const float* w, int c0, float (&out)[WIDTH][VEC]) {
#pragma unroll
  for (int i = 0; i < VEC; ++i)
#pragma unroll
    for (int j = 0; j < WIDTH; ++j) out[j][i] = w[(c0 + i) * WIDTH + j];
}

template <typename T>
__global__ void forward_kernel(Tensors tensors, const int64_t* positions, int64_t tokens,
                               int channels) {
  const int c0 = (blockIdx.x * THREADS + threadIdx.x) * VEC;
  if (c0 >= channels) return;
  const int which = blockIdx.z;
  const T* x = static_cast<const T*>(tensors.x[which]);
  T* y = static_cast<T*>(tensors.out[which]);
  float w[WIDTH][VEC];
  load_weights<T>(tensors.w[which], c0, w);
  const int64_t t0 = static_cast<int64_t>(blockIdx.y) * STRIP;
  const int64_t end = min(t0 + STRIP, tokens);
  float window[WIDTH][VEC];
#pragma unroll
  for (int s = 0; s < WIDTH - 1; ++s) {
    const int64_t t = t0 - (WIDTH - 1) + s;
    if (t >= 0) load8(x + t * channels + c0, window[s]);
    else
#pragma unroll
      for (int i = 0; i < VEC; ++i) window[s][i] = 0.0f;
  }
  for (int64_t t = t0; t < end; ++t) {
    load8(x + t * channels + c0, window[3]);
    float pre[VEC], out[VEC];
    pre_activation(w, window, positions[t], pre);
#pragma unroll
    for (int i = 0; i < VEC; ++i) out[i] = pre[i] * sigmoid(pre[i]);
    store8(y + t * channels + c0, out);
#pragma unroll
    for (int s = 0; s < WIDTH - 1; ++s)
#pragma unroll
      for (int i = 0; i < VEC; ++i) window[s][i] = window[s + 1][i];
  }
}

// d[u] = dy[u] * silu'(pre[u]); dx[t] = sum_s w[3 - s] * d[t + s] * (position[t + s] >= s);
// dw[3 - s] += d[u] * x[u - s] * (position[u] >= s), for u in this strip.
template <typename T>
__global__ void backward_kernel(Tensors tensors, const int64_t* positions, int64_t tokens,
                                int channels) {
  const int c0 = (blockIdx.x * THREADS + threadIdx.x) * VEC;
  if (c0 >= channels) return;
  const int which = blockIdx.z;
  const T* x = static_cast<const T*>(tensors.x[which]);
  const T* dy = static_cast<const T*>(tensors.grad[which]);
  T* dx = static_cast<T*>(tensors.out[which]);
  float w[WIDTH][VEC];
  load_weights<T>(tensors.w[which], c0, w);
  const int64_t t0 = static_cast<int64_t>(blockIdx.y) * STRIP;
  const int64_t end = min(t0 + STRIP, tokens);
  float window[WIDTH][VEC], d[WIDTH][VEC], dw[WIDTH][VEC];
  int64_t position[WIDTH];
#pragma unroll
  for (int s = 0; s < WIDTH; ++s) {
    position[s] = 0;
#pragma unroll
    for (int i = 0; i < VEC; ++i) d[s][i] = dw[s][i] = 0.0f;
  }
#pragma unroll
  for (int s = 0; s < WIDTH - 1; ++s) {
    const int64_t t = t0 - (WIDTH - 1) + s;
    if (t >= 0) load8(x + t * channels + c0, window[s]);
    else
#pragma unroll
      for (int i = 0; i < VEC; ++i) window[s][i] = 0.0f;
  }
  for (int64_t u = t0; u < end + WIDTH - 1; ++u) {
    // Slide d and position so index 3 is u.
#pragma unroll
    for (int s = 0; s < WIDTH - 1; ++s) {
      position[s] = position[s + 1];
#pragma unroll
      for (int i = 0; i < VEC; ++i) d[s][i] = d[s + 1][i];
    }
    if (u < tokens) {
      load8(x + u * channels + c0, window[3]);
      float pre[VEC], grad[VEC];
      position[3] = positions[u];
      pre_activation(w, window, position[3], pre);
      load8(dy + u * channels + c0, grad);
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const float sig = sigmoid(pre[i]);
        d[3][i] = grad[i] * sig * (1.0f + pre[i] * (1.0f - sig));
      }
      if (u < end) {
#pragma unroll
        for (int s = 0; s < WIDTH; ++s)
          if (position[3] >= s)
#pragma unroll
            for (int i = 0; i < VEC; ++i) dw[3 - s][i] += d[3][i] * window[3 - s][i];
      }
#pragma unroll
      for (int s = 0; s < WIDTH - 1; ++s)
#pragma unroll
        for (int i = 0; i < VEC; ++i) window[s][i] = window[s + 1][i];
    } else {
      position[3] = 0;
#pragma unroll
      for (int i = 0; i < VEC; ++i) d[3][i] = 0.0f;
    }
    const int64_t t = u - (WIDTH - 1);
    if (t >= t0 && t < end) {
      float out[VEC];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        float acc = w[3][i] * d[0][i];
#pragma unroll
        for (int s = 1; s < WIDTH; ++s)
          if (position[s] >= s) acc += w[3 - s][i] * d[s][i];
        out[i] = acc;
      }
      store8(dx + t * channels + c0, out);
    }
  }
#pragma unroll
  for (int i = 0; i < VEC; ++i)
#pragma unroll
    for (int j = 0; j < WIDTH; ++j) atomicAdd(tensors.dw[which] + (c0 + i) * WIDTH + j, dw[j][i]);
}

void check(const std::vector<torch::Tensor>& xs, const std::vector<torch::Tensor>& ws,
           const torch::Tensor& positions) {
  TORCH_CHECK(xs.size() == 3 && ws.size() == 3, "expected q, k and v");
  for (int i = 0; i < 3; ++i) {
    TORCH_CHECK(xs[i].is_cuda() && xs[i].is_contiguous() && xs[i].dim() == 3, "x must be [B, T, C]");
    TORCH_CHECK(xs[i].sizes() == xs[0].sizes() && xs[i].dtype() == xs[0].dtype(), "q, k, v differ");
    TORCH_CHECK(ws[i].dtype() == torch::kFloat32 && ws[i].is_contiguous(), "weights must be fp32");
    TORCH_CHECK(ws[i].numel() == xs[0].size(2) * WIDTH, "weights must be [C, 1, 4]");
  }
  TORCH_CHECK(xs[0].size(2) % VEC == 0, "channels must be a multiple of 8");
  TORCH_CHECK(positions.dtype() == torch::kInt64 && positions.is_contiguous(), "positions int64");
  TORCH_CHECK(positions.numel() == xs[0].size(0) * xs[0].size(1), "positions must be [B, T]");
}

dim3 grid(const torch::Tensor& x) {
  const int64_t tokens = x.size(0) * x.size(1);
  const int groups = x.size(2) / VEC;
  return dim3((groups + THREADS - 1) / THREADS, (tokens + STRIP - 1) / STRIP, 3);
}

std::vector<torch::Tensor> forward(std::vector<torch::Tensor> xs, std::vector<torch::Tensor> ws,
                                   torch::Tensor positions) {
  check(xs, ws, positions);
  const c10::cuda::CUDAGuard guard(xs[0].device());
  Tensors tensors;
  std::vector<torch::Tensor> outs;
  for (int i = 0; i < 3; ++i) {
    outs.push_back(torch::empty_like(xs[i]));
    tensors.x[i] = xs[i].data_ptr();
    tensors.w[i] = ws[i].data_ptr<float>();
    tensors.out[i] = outs[i].data_ptr();
  }
  const int64_t tokens = xs[0].size(0) * xs[0].size(1);
  const int channels = xs[0].size(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (xs[0].scalar_type() == torch::kBFloat16)
    forward_kernel<__nv_bfloat16><<<grid(xs[0]), THREADS, 0, stream>>>(
        tensors, positions.data_ptr<int64_t>(), tokens, channels);
  else if (xs[0].scalar_type() == torch::kFloat32)
    forward_kernel<float><<<grid(xs[0]), THREADS, 0, stream>>>(
        tensors, positions.data_ptr<int64_t>(), tokens, channels);
  else
    TORCH_CHECK(false, "x must be bf16 or fp32");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return outs;
}

std::vector<torch::Tensor> backward(std::vector<torch::Tensor> xs, std::vector<torch::Tensor> ws,
                                    torch::Tensor positions, std::vector<torch::Tensor> grads) {
  check(xs, ws, positions);
  const c10::cuda::CUDAGuard guard(xs[0].device());
  Tensors tensors;
  std::vector<torch::Tensor> dxs, dws;
  for (int i = 0; i < 3; ++i) {
    TORCH_CHECK(grads[i].sizes() == xs[i].sizes() && grads[i].dtype() == xs[i].dtype(),
                "gradient must match x");
    grads[i] = grads[i].contiguous();
    dxs.push_back(torch::empty_like(xs[i]));
    dws.push_back(torch::zeros_like(ws[i]));
    tensors.x[i] = xs[i].data_ptr();
    tensors.w[i] = ws[i].data_ptr<float>();
    tensors.out[i] = dxs[i].data_ptr();
    tensors.grad[i] = grads[i].data_ptr();
    tensors.dw[i] = dws[i].data_ptr<float>();
  }
  const int64_t tokens = xs[0].size(0) * xs[0].size(1);
  const int channels = xs[0].size(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (xs[0].scalar_type() == torch::kBFloat16)
    backward_kernel<__nv_bfloat16><<<grid(xs[0]), THREADS, 0, stream>>>(
        tensors, positions.data_ptr<int64_t>(), tokens, channels);
  else
    backward_kernel<float><<<grid(xs[0]), THREADS, 0, stream>>>(
        tensors, positions.data_ptr<int64_t>(), tokens, channels);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  dxs.insert(dxs.end(), dws.begin(), dws.end());
  return dxs;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward, "KDA q/k/v short convolutions with SiLU");
  module.def("backward", &backward, "Their input and weight gradients");
}
