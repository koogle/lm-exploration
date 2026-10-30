// Decode-step kernels for HybridLM: one new token per row (generate.py's Decoder).
//
// Every kernel reads the cache slot and per-row document starts from device memory,
// so tensor shapes never change between steps and a whole step is captured once as a
// CUDA graph and replayed.
//
//   kda_step:  per (row, head): the q/k/v short convolutions with their history,
//              SiLU, L2-normalized q and k, the decay and write strength, the delta
//              rule update of the fp32 state, the read-out, its RMSNorm and the output
//              gate. One thread per value column; the column stays in registers.
//   attention: per (row, query head): the query against cache keys [start, slot] of
//              its key head (online softmax, warps striding the keys), scaled by the
//              sink's sigmoid(lse - sink).
//
// Rows whose previous token ended a document (`fresh`) start from a zero state and
// convolution history; attention reads only keys from the row's document start.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int HEAD_DIM = 128;
constexpr int MAX_CONV = 8;
constexpr int ATTENTION_WARPS = 8;

__device__ float block_sum(float value, float* scratch) {
  // Sum over the block's HEAD_DIM threads (4 warps).
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffff, value, offset);
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  float total = 0.f;
  for (int w = 0; w < HEAD_DIM / 32; ++w) total += scratch[w];
  return total;
}

__device__ float convolve(const float* projection, const float* weight, float* history,
                          bool fresh, int row, int channel, int rows, int inner, int width) {
  // history [rows, width - 1, inner], oldest first; returns SiLU of the convolution and
  // shifts the new input in.
  float past[MAX_CONV];
  for (int i = 0; i < width - 1; ++i)
    past[i] = fresh ? 0.f : history[((size_t)row * (width - 1) + i) * inner + channel];
  float x = projection[(size_t)row * inner + channel];
  float out = x * weight[channel * width + width - 1];
  for (int i = 0; i < width - 1; ++i) out += past[i] * weight[channel * width + i];
  for (int i = 0; i < width - 2; ++i)
    history[((size_t)row * (width - 1) + i) * inner + channel] = past[i + 1];
  history[((size_t)row * (width - 1) + width - 2) * inner + channel] = x;
  return out / (1.f + __expf(-out));
}

__global__ void kda_step_kernel(const float* q_in, const float* k_in, const float* v_in,
                                const float* f_in, const float* b_in, const float* g_in,
                                const float* q_weight, const float* k_weight,
                                const float* v_weight, float* q_history, float* k_history,
                                float* v_history, float* state, const bool* fresh,
                                const float* a_log, const float* dt_bias, const float* o_norm,
                                float* out, int rows, int heads, int width, float eps) {
  __shared__ float shared_q[HEAD_DIM], shared_k[HEAD_DIM], shared_decay[HEAD_DIM];
  __shared__ float scratch[HEAD_DIM / 32];
  int row = blockIdx.x / heads, head = blockIdx.x % heads, j = threadIdx.x;
  int inner = heads * HEAD_DIM, channel = head * HEAD_DIM + j;
  bool reset = fresh[row];
  float q = convolve(q_in, q_weight, q_history, reset, row, channel, rows, inner, width);
  float k = convolve(k_in, k_weight, k_history, reset, row, channel, rows, inner, width);
  float v = convolve(v_in, v_weight, v_history, reset, row, channel, rows, inner, width);
  // F.normalize: x / max(||x||, 1e-12); q also by head_dim^-0.5.
  float q_norm = fmaxf(sqrtf(block_sum(q * q, scratch)), 1e-12f);
  float k_norm = fmaxf(sqrtf(block_sum(k * k, scratch)), 1e-12f);
  shared_q[j] = q / q_norm * rsqrtf((float)HEAD_DIM);
  shared_k[j] = k / k_norm;
  float gate_in = f_in[(size_t)row * inner + channel] + dt_bias[channel];
  float softplus = gate_in > 20.f ? gate_in : log1pf(__expf(gate_in));
  shared_decay[j] = __expf(-__expf(a_log[head]) * softplus);
  float beta = 1.f / (1.f + __expf(-b_in[(size_t)row * heads + head]));
  __syncthreads();

  // This thread's value column j of the state [K, V].
  float* column = state + ((size_t)row * heads + head) * HEAD_DIM * HEAD_DIM + j;
  float s[HEAD_DIM];
  float predicted = 0.f;
#pragma unroll
  for (int i = 0; i < HEAD_DIM; ++i) {
    s[i] = reset ? 0.f : column[(size_t)i * HEAD_DIM] * shared_decay[i];
    predicted += shared_k[i] * s[i];
  }
  float error = beta * (v - predicted);
  float read = 0.f;
#pragma unroll
  for (int i = 0; i < HEAD_DIM; ++i) {
    s[i] += shared_k[i] * error;
    column[(size_t)i * HEAD_DIM] = s[i];
    read += shared_q[i] * s[i];
  }
  float scale = rsqrtf(block_sum(read * read, scratch) / HEAD_DIM + eps);
  float gate = 1.f / (1.f + __expf(-g_in[(size_t)row * inner + channel]));
  out[(size_t)row * inner + channel] = read * scale * o_norm[j] * gate;
}

template <typename scalar_t>
__global__ void attention_kernel(const scalar_t* query, const scalar_t* keys,
                                 const scalar_t* values, const int64_t* starts, const int64_t* slot,
                                 const float* sink, float* out, int heads, int kv_heads,
                                 int capacity, float scale) {
  // Block: one (row, query head); warp w takes keys start + w, start + w + WARPS, ...
  // Lane l holds dimensions l, l + 32, l + 64, l + 96.
  __shared__ float merge_best[ATTENTION_WARPS], merge_total[ATTENTION_WARPS];
  __shared__ float merge_sum[ATTENTION_WARPS][HEAD_DIM];
  int row = blockIdx.x / heads, head = blockIdx.x % heads;
  int kv = head / (heads / kv_heads);
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const scalar_t* q = query + (size_t)blockIdx.x * HEAD_DIM;
  float qd[4];
  for (int d = 0; d < 4; ++d) qd[d] = static_cast<float>(q[lane + 32 * d]) * scale;
  size_t base = ((size_t)row * kv_heads + kv) * capacity * HEAD_DIM;
  float best = -INFINITY, total = 0.f, sum[4] = {0.f, 0.f, 0.f, 0.f};
  int64_t last = *slot;
  for (int64_t t = starts[row] + warp; t <= last; t += ATTENTION_WARPS) {
    const scalar_t* key = keys + base + (size_t)t * HEAD_DIM;
    float dot = 0.f;
    for (int d = 0; d < 4; ++d) dot += qd[d] * static_cast<float>(key[lane + 32 * d]);
    for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, offset);
    float next = fmaxf(best, dot);
    float correction = __expf(best - next), weight = __expf(dot - next);
    const scalar_t* value = values + base + (size_t)t * HEAD_DIM;
    for (int d = 0; d < 4; ++d)
      sum[d] = sum[d] * correction + weight * static_cast<float>(value[lane + 32 * d]);
    total = total * correction + weight;
    best = next;
  }
  if (lane == 0) {
    merge_best[warp] = best;
    merge_total[warp] = total;
  }
  for (int d = 0; d < 4; ++d) merge_sum[warp][lane + 32 * d] = sum[d];
  __syncthreads();
  if (warp == 0) {
    float overall = -INFINITY;
    for (int w = 0; w < ATTENTION_WARPS; ++w) overall = fmaxf(overall, merge_best[w]);
    float denominator = 0.f, combined[4] = {0.f, 0.f, 0.f, 0.f};
    for (int w = 0; w < ATTENTION_WARPS; ++w) {
      if (merge_best[w] == -INFINITY) continue;
      float factor = __expf(merge_best[w] - overall);
      denominator += merge_total[w] * factor;
      for (int d = 0; d < 4; ++d) combined[d] += merge_sum[w][lane + 32 * d] * factor;
    }
    // Softmax over [scores, sink] scales the output by sigmoid(lse - sink).
    float lse = overall + logf(denominator);
    float kept = 1.f / (1.f + __expf(sink[head] - lse));
    for (int d = 0; d < 4; ++d)
      out[(size_t)blockIdx.x * HEAD_DIM + lane + 32 * d] = combined[d] / denominator * kept;
  }
}

void check(const torch::Tensor& t, torch::ScalarType type, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
  TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}

torch::Tensor kda_step(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor f,
                       torch::Tensor b, torch::Tensor g, torch::Tensor q_weight,
                       torch::Tensor k_weight, torch::Tensor v_weight, torch::Tensor q_history,
                       torch::Tensor k_history, torch::Tensor v_history, torch::Tensor state,
                       torch::Tensor fresh, torch::Tensor a_log, torch::Tensor dt_bias,
                       torch::Tensor o_norm, double eps) {
  for (auto* t : {&q, &k, &v, &f, &b, &g, &q_weight, &k_weight, &v_weight, &q_history,
                  &k_history, &v_history, &state, &a_log, &dt_bias, &o_norm})
    check(*t, torch::kFloat32, "a KDA input");
  check(fresh, torch::kBool, "fresh");
  int rows = q.size(0), heads = a_log.size(0), width = q_weight.size(1);
  TORCH_CHECK(q.size(1) == heads * HEAD_DIM, "the decode kernels assume a head dimension of 128");
  TORCH_CHECK(width >= 2 && width <= MAX_CONV, "unsupported convolution width");
  auto out = torch::empty_like(q);
  kda_step_kernel<<<rows * heads, HEAD_DIM, 0, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), f.data_ptr<float>(),
      b.data_ptr<float>(), g.data_ptr<float>(), q_weight.data_ptr<float>(),
      k_weight.data_ptr<float>(), v_weight.data_ptr<float>(), q_history.data_ptr<float>(),
      k_history.data_ptr<float>(), v_history.data_ptr<float>(), state.data_ptr<float>(),
      fresh.data_ptr<bool>(), a_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      o_norm.data_ptr<float>(), out.data_ptr<float>(), rows, heads, width, (float)eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor attention(torch::Tensor query, torch::Tensor keys, torch::Tensor values,
                        torch::Tensor starts, torch::Tensor slot, torch::Tensor sink) {
  check(keys, query.scalar_type(), "keys");
  check(values, query.scalar_type(), "values");
  check(query, query.scalar_type(), "query");
  check(starts, torch::kLong, "starts");
  check(slot, torch::kLong, "slot");
  check(sink, torch::kFloat32, "sink");
  int rows = query.size(0), heads = query.size(1), kv_heads = keys.size(1);
  TORCH_CHECK(query.size(2) == HEAD_DIM, "the decode kernels assume a head dimension of 128");
  TORCH_CHECK(heads % kv_heads == 0, "query heads must be a multiple of key heads");
  auto out = torch::empty({rows, heads, HEAD_DIM}, query.options().dtype(torch::kFloat32));
  float scale = 1.f / sqrtf((float)HEAD_DIM);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kBFloat16, at::kHalf, query.scalar_type(), "attention", [&] {
    attention_kernel<scalar_t><<<rows * heads, 32 * ATTENTION_WARPS, 0,
                                 at::cuda::getCurrentCUDAStream()>>>(
        query.data_ptr<scalar_t>(), keys.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
        starts.data_ptr<int64_t>(), slot.data_ptr<int64_t>(), sink.data_ptr<float>(),
        out.data_ptr<float>(), heads, kv_heads, keys.size(2), scale);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kda_step", &kda_step, "One KDA decode step per (row, head)");
  m.def("attention", &attention, "Decode attention with a sink over a key/value cache");
}
