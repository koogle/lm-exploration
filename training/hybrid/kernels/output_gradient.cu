// Accumulate a BF16 output-projection gradient directly into its FP32 buffer.
// C <- C + grad_logits.T @ hidden, using cuBLAS tensor cores. This avoids
// materializing a full vocabulary-by-width FP32 temporary for every loss chunk.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <climits>
#include <optional>
#include <torch/extension.h>

void accumulate_impl(torch::Tensor destination, torch::Tensor grad, torch::Tensor hidden,
                     const std::optional<torch::Tensor>& coefficients) {
  TORCH_CHECK(destination.is_cuda() && grad.is_cuda() && hidden.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(destination.device() == grad.device() && grad.device() == hidden.device(), "device mismatch");
  TORCH_CHECK(destination.scalar_type() == torch::kFloat32 &&
              grad.scalar_type() == torch::kBFloat16 && hidden.scalar_type() == torch::kBFloat16,
              "FP32 destination and BF16 operands required");
  TORCH_CHECK(destination.dim() == 2 && grad.dim() == 2 && hidden.dim() == 2, "matrices required");
  TORCH_CHECK(destination.is_contiguous() && grad.is_contiguous() && hidden.is_contiguous(),
              "contiguous matrices required");
  TORCH_CHECK(grad.size(0) == hidden.size(0) && destination.size(0) == grad.size(1) &&
              destination.size(1) == hidden.size(1), "shape mismatch");
  const int64_t rows = grad.size(0), vocab = grad.size(1), width = hidden.size(1);
  TORCH_CHECK(rows > 0 && vocab > 0 && width > 0 && rows <= INT_MAX && vocab <= INT_MAX &&
              width <= INT_MAX, "unsupported GEMM dimensions");
  c10::cuda::CUDAGuard guard(destination.device());
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  const float alpha = 1.0f, beta = 1.0f;
  cublasPointerMode_t previous;
  TORCH_CHECK(cublasGetPointerMode(handle, &previous) == CUBLAS_STATUS_SUCCESS, "get pointer mode failed");
  if (coefficients) {
    TORCH_CHECK(coefficients->is_cuda() && coefficients->device() == destination.device() &&
                coefficients->scalar_type() == torch::kFloat32 && coefficients->is_contiguous() &&
                coefficients->numel() == 2, "device FP32 [alpha, beta] required");
  }
  TORCH_CHECK(cublasSetPointerMode(handle, coefficients ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST)
              == CUBLAS_STATUS_SUCCESS, "set pointer mode failed");
  const float* alpha_ptr = coefficients ? coefficients->data_ptr<float>() : &alpha;
  const float* beta_ptr = coefficients ? coefficients->data_ptr<float>() + 1 : &beta;
  // Row-major C[V,D] is column-major C^T[D,V] = H^T[D,R] @ G[R,V].
  const auto status = cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_T, width, vocab, rows, alpha_ptr,
      hidden.data_ptr(), CUDA_R_16BF, width,
      grad.data_ptr(), CUDA_R_16BF, vocab, beta_ptr,
      destination.data_ptr(), CUDA_R_32F, width,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  const auto restore = cublasSetPointerMode(handle, previous);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "output-gradient GEMM failed: ", int(status));
  TORCH_CHECK(restore == CUBLAS_STATUS_SUCCESS, "restore pointer mode failed");
}

void accumulate(torch::Tensor destination, torch::Tensor grad, torch::Tensor hidden) {
  accumulate_impl(destination, grad, hidden, std::nullopt);
}

void accumulate_scaled(torch::Tensor destination, torch::Tensor grad, torch::Tensor hidden,
                       torch::Tensor coefficients) {
  accumulate_impl(destination, grad, hidden, coefficients);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("accumulate", &accumulate);
  m.def("accumulate_scaled", &accumulate_scaled);
}
