// Fused hard-target cross-entropy without Softmax Collapse (Barron et al., eqs. 6-7).
//
// One block per row. With lse = LSE_{j != t}(z_j - z_t), the loss is softplus(lse) and
//   g_j = exp(z_j - z_t - loss) for j != t,   g_t = -exp(lse - loss) = -sum_{j != t} g_j,
// so nothing adds e^-m to 1.0. The forward pass streams the row once with an online
// log-sum-exp and stores lse; the backward pass writes the gradient directly.
// Half/bf16 are computed in fp32, double in double. Built without --use_fast_math:
// accurate expf/logf/log1pf are required.

#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 512;
constexpr int kWarp = 32;

template <typename T>
__device__ __forceinline__ T softplus(T x) {
  return fmax(x, T(0)) + log1p(exp(-fabs(x)));
}

// Combine two partial log-sum-exps (max, sum of exp(x - max)).
template <typename T>
__device__ __forceinline__ void merge(T& m, T& s, T other_m, T other_s) {
  T new_m = fmax(m, other_m);
  if (new_m == -INFINITY) return;
  s = s * exp(m - new_m) + other_s * exp(other_m - new_m);
  m = new_m;
}

template <typename T>
__device__ __forceinline__ void warp_merge(T& m, T& s) {
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    T other_m = __shfl_down_sync(0xffffffff, m, offset);
    T other_s = __shfl_down_sync(0xffffffff, s, offset);
    merge(m, s, other_m, other_s);
  }
}

template <typename scalar_t, typename acc_t>
__global__ void __launch_bounds__(kThreads) forward_kernel(
    const scalar_t* __restrict__ logits, int64_t stride, const int64_t* __restrict__ targets,
    acc_t* __restrict__ lse_out, acc_t* __restrict__ loss_out, int64_t vocab,
    int64_t ignore_index) {
  const int64_t row = blockIdx.x;
  const int64_t target = targets[row];
  if (target == ignore_index) {
    if (threadIdx.x == 0) lse_out[row] = loss_out[row] = acc_t(0);
    return;
  }
  CUDA_KERNEL_ASSERT(target >= 0 && target < vocab);
  const scalar_t* z = logits + row * stride;
  const acc_t z_t = static_cast<acc_t>(z[target]);

  acc_t m = -INFINITY, s = 0;
  for (int64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
    const acc_t x = static_cast<acc_t>(z[j]) - z_t;
    if (j == target || x == -INFINITY) continue;
    if (x > m) {
      s = s * exp(m - x) + acc_t(1);
      m = x;
    } else {
      s += exp(x - m);
    }
  }

  __shared__ acc_t warp_m[kThreads / kWarp], warp_s[kThreads / kWarp];
  const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
  warp_merge(m, s);
  if (lane == 0) {
    warp_m[warp] = m;
    warp_s[warp] = s;
  }
  __syncthreads();
  if (warp == 0) {
    const bool live = lane < blockDim.x / kWarp;
    m = live ? warp_m[lane] : acc_t(-INFINITY);
    s = live ? warp_s[lane] : acc_t(0);
    warp_merge(m, s);
    if (lane == 0) {
      const acc_t lse = m + log(s);
      lse_out[row] = lse;
      loss_out[row] = softplus(lse);
    }
  }
}

template <typename scalar_t, typename acc_t>
__global__ void __launch_bounds__(kThreads) backward_kernel(
    const scalar_t* __restrict__ logits, int64_t stride, const int64_t* __restrict__ targets,
    const acc_t* __restrict__ lse_in, const acc_t* __restrict__ grad_loss,
    scalar_t* __restrict__ grad, int64_t grad_stride, int64_t vocab, int64_t ignore_index) {
  const int64_t row = blockIdx.x;
  const int64_t target = targets[row];
  scalar_t* out = grad + row * grad_stride;
  if (target == ignore_index) {
    for (int64_t j = threadIdx.x; j < vocab; j += blockDim.x) out[j] = scalar_t(0);
    return;
  }
  const scalar_t* z = logits + row * stride;
  const acc_t z_t = static_cast<acc_t>(z[target]);
  const acc_t lse = lse_in[row];
  const acc_t loss = softplus(lse);
  const acc_t scale = grad_loss[row];
  for (int64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
    const acc_t g = j == target ? -exp(lse - loss) : exp(static_cast<acc_t>(z[j]) - z_t - loss);
    out[j] = static_cast<scalar_t>(g * scale);
  }
}

void check_inputs(const torch::Tensor& logits, const torch::Tensor& targets) {
  TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 && logits.stride(1) == 1,
              "logits must be a CUDA (N, V) tensor with contiguous rows");
  TORCH_CHECK(logits.size(1) >= 2, "vocabulary must have at least 2 entries");
  TORCH_CHECK(targets.device() == logits.device() && targets.scalar_type() == torch::kLong &&
                  targets.is_contiguous() && targets.numel() == logits.size(0),
              "targets must be contiguous int64 with one entry per row, on the logits' device");
}

std::vector<torch::Tensor> forward(torch::Tensor logits, torch::Tensor targets,
                                   int64_t ignore_index) {
  check_inputs(logits, targets);
  const c10::cuda::CUDAGuard guard(logits.device());
  const auto rows = logits.size(0), vocab = logits.size(1);
  const auto acc_type = logits.scalar_type() == torch::kDouble ? torch::kDouble : torch::kFloat;
  auto lse = torch::empty({rows}, logits.options().dtype(acc_type));
  auto loss = torch::empty_like(lse);
  if (rows == 0) return {loss, lse};
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(), "stable_ce_forward", [&] {
    using acc_t = at::opmath_type<scalar_t>;
    forward_kernel<scalar_t, acc_t><<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<scalar_t>(), logits.stride(0), targets.data_ptr<int64_t>(),
        lse.data_ptr<acc_t>(), loss.data_ptr<acc_t>(), vocab, ignore_index);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return {loss, lse};
}

torch::Tensor backward(torch::Tensor logits, torch::Tensor targets, torch::Tensor lse,
                       torch::Tensor grad_loss, int64_t ignore_index) {
  check_inputs(logits, targets);
  const c10::cuda::CUDAGuard guard(logits.device());
  const auto rows = logits.size(0), vocab = logits.size(1);
  grad_loss = grad_loss.to(lse.scalar_type()).contiguous();
  auto grad = torch::empty_like(logits, at::MemoryFormat::Contiguous);
  if (rows == 0) return grad;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(), "stable_ce_backward", [&] {
    using acc_t = at::opmath_type<scalar_t>;
    backward_kernel<scalar_t, acc_t><<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<scalar_t>(), logits.stride(0), targets.data_ptr<int64_t>(),
        lse.data_ptr<acc_t>(), grad_loss.data_ptr<acc_t>(), grad.data_ptr<scalar_t>(),
        grad.stride(0), vocab, ignore_index);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  return grad;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Per-row loss and LSE over non-target shifted logits");
  m.def("backward", &backward, "Cross-entropy gradient from the saved LSE");
}
