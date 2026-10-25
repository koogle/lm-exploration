// Decode-step kernels for RL rollouts of the 124M decoder (one new token per row).
//
// Every kernel reads the current cache slot, step and per-row positions from device
// memory, so tensor shapes never change between steps and a whole decode step can be
// captured once as a CUDA graph and replayed (`kernels/decode.py`).
//
//   rope_append: q/k RoPE of the new token, k and v written to the cache slot.
//   attention:   one query per (row, head) over cache keys [start[row], slot], online
//                softmax, one warp per key stripe; head dimension 64.
//   sample:      log-sum-exp of logits / T and an inverse-CDF draw from a given uniform,
//                returning the token and its log-probability; one block per row.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

constexpr int HEAD_DIM = 64;
constexpr int ATTENTION_WARPS = 8;

__global__ void rope_append_kernel(const bf16* qkv, const int* positions, const int* slot,
                                   const float* frequencies, bf16* q_out, bf16* k_cache,
                                   bf16* v_cache, int heads, int capacity) {
  // Block: one row; thread: one (head, pair) of q and k, plus two v values.
  int row = blockIdx.x;
  int pair = threadIdx.x % (HEAD_DIM / 2);
  int head = threadIdx.x / (HEAD_DIM / 2);
  if (head >= heads) return;
  int width = heads * HEAD_DIM;
  const bf16* base = qkv + (size_t)row * 3 * width + head * HEAD_DIM + 2 * pair;
  float angle = (float)positions[row] * frequencies[pair];
  // The model rotates in BF16: cos and sin are rounded to BF16 first.
  float c = __bfloat162float(__float2bfloat16(cosf(angle)));
  float s = __bfloat162float(__float2bfloat16(sinf(angle)));
  size_t cache_at = (((size_t)row * heads + head) * capacity + *slot) * HEAD_DIM + 2 * pair;
  size_t q_at = ((size_t)row * heads + head) * HEAD_DIM + 2 * pair;
  for (int part = 0; part < 2; ++part) {
    float even = __bfloat162float(base[part * width]);
    float odd = __bfloat162float(base[part * width + 1]);
    bf16 a = __float2bfloat16(__bfloat162float(__float2bfloat16(even * c)) -
                              __bfloat162float(__float2bfloat16(odd * s)));
    bf16 b = __float2bfloat16(__bfloat162float(__float2bfloat16(even * s)) +
                              __bfloat162float(__float2bfloat16(odd * c)));
    bf16* target = part == 0 ? q_out + q_at : k_cache + cache_at;
    target[0] = a;
    target[1] = b;
  }
  v_cache[cache_at] = base[2 * width];
  v_cache[cache_at + 1] = base[2 * width + 1];
}

__global__ void attention_kernel(const bf16* q, const bf16* k_cache, const bf16* v_cache,
                                 const int* start, const int* slot, bf16* out, int heads,
                                 int capacity, float scale) {
  // Block: one (row, head); warp w takes keys start + w, start + w + WARPS, ...
  int index = blockIdx.x;  // row * heads + head
  int row = index / heads;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const bf16* query = q + (size_t)index * HEAD_DIM;
  float q0 = __bfloat162float(query[2 * lane]) * scale;
  float q1 = __bfloat162float(query[2 * lane + 1]) * scale;
  const bf16* keys = k_cache + (size_t)index * capacity * HEAD_DIM;
  const bf16* values = v_cache + (size_t)index * capacity * HEAD_DIM;
  float best = -INFINITY, total = 0.f, a0 = 0.f, a1 = 0.f;
  int last = *slot;
  for (int j = start[row] + warp; j <= last; j += ATTENTION_WARPS) {
    const __nv_bfloat162 k2 = reinterpret_cast<const __nv_bfloat162*>(keys + (size_t)j * HEAD_DIM)[lane];
    float dot = q0 * __low2float(k2) + q1 * __high2float(k2);
    for (int offset = 16; offset > 0; offset >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, offset);
    float next = fmaxf(best, dot);
    float correction = __expf(best - next), weight = __expf(dot - next);
    const __nv_bfloat162 v2 = reinterpret_cast<const __nv_bfloat162*>(values + (size_t)j * HEAD_DIM)[lane];
    // As in the fused attention kernels, the weights meet V in BF16; sums stay FP32.
    float rounded = __bfloat162float(__float2bfloat16(weight));
    a0 = a0 * correction + rounded * __low2float(v2);
    a1 = a1 * correction + rounded * __high2float(v2);
    total = total * correction + weight;
    best = next;
  }
  __shared__ float maxima[ATTENTION_WARPS], totals[ATTENTION_WARPS];
  __shared__ float sums[ATTENTION_WARPS][HEAD_DIM];
  if (lane == 0) {
    maxima[warp] = best;
    totals[warp] = total;
  }
  sums[warp][2 * lane] = a0;
  sums[warp][2 * lane + 1] = a1;
  __syncthreads();
  if (warp == 0) {
    float overall = -INFINITY;
    for (int w = 0; w < ATTENTION_WARPS; ++w) overall = fmaxf(overall, maxima[w]);
    float denominator = 0.f, r0 = 0.f, r1 = 0.f;
    for (int w = 0; w < ATTENTION_WARPS; ++w) {
      if (maxima[w] == -INFINITY) continue;
      float factor = __expf(maxima[w] - overall);
      denominator += totals[w] * factor;
      r0 += sums[w][2 * lane] * factor;
      r1 += sums[w][2 * lane + 1] * factor;
    }
    out[(size_t)index * HEAD_DIM + 2 * lane] = __float2bfloat16(r0 / denominator);
    out[(size_t)index * HEAD_DIM + 2 * lane + 1] = __float2bfloat16(r1 / denominator);
  }
}

__device__ float block_reduce(float value, float* scratch, bool maximum) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_xor_sync(0xffffffff, value, offset);
    value = maximum ? fmaxf(value, other) : value + other;
  }
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < blockDim.x / 32 ? scratch[lane] : (maximum ? -INFINITY : 0.f);
    for (int offset = 16; offset > 0; offset >>= 1) {
      float other = __shfl_xor_sync(0xffffffff, value, offset);
      value = maximum ? fmaxf(value, other) : value + other;
    }
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  float result = scratch[0];
  __syncthreads();
  return result;
}

__global__ void sample_kernel(const float* logits, const float* uniforms, const int* step,
                              float inverse_temperature, int vocab, int batch, long* tokens,
                              float* logprobs, float* normalizers) {
  // Block: one row. Each thread owns a contiguous chunk of the vocabulary.
  extern __shared__ float scratch[];
  int row = blockIdx.x;
  const float* x = logits + (size_t)row * vocab;
  int chunk = (vocab + blockDim.x - 1) / blockDim.x;
  int begin = threadIdx.x * chunk, end = min(begin + chunk, vocab);
  float local = -INFINITY;
  for (int i = begin; i < end; ++i) local = fmaxf(local, x[i] * inverse_temperature);
  float best = block_reduce(local, scratch, true);
  float mass = 0.f;
  for (int i = begin; i < end; ++i) mass += __expf(x[i] * inverse_temperature - best);
  // Inclusive scan of chunk masses across the block (Hillis-Steele in shared memory).
  float* prefix = scratch + 32;
  prefix[threadIdx.x] = mass;
  __syncthreads();
  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    float add = threadIdx.x >= offset ? prefix[threadIdx.x - offset] : 0.f;
    __syncthreads();
    prefix[threadIdx.x] += add;
    __syncthreads();
  }
  float total = prefix[blockDim.x - 1];
  float target = uniforms[(size_t)(*step) * batch + row] * total;
  float before = threadIdx.x == 0 ? 0.f : prefix[threadIdx.x - 1];
  bool mine = target >= before && (target < prefix[threadIdx.x] || threadIdx.x == blockDim.x - 1);
  if (mine && end > begin) {
    float running = before;
    int chosen = end - 1;
    for (int i = begin; i < end; ++i) {
      running += __expf(x[i] * inverse_temperature - best);
      if (running > target) {
        chosen = i;
        break;
      }
    }
    float normalizer = best + logf(total);
    tokens[row] = chosen;
    logprobs[row] = x[chosen] * inverse_temperature - normalizer;
    normalizers[row] = normalizer;
  }
}

torch::Tensor rope_append(torch::Tensor qkv, torch::Tensor positions, torch::Tensor slot,
                          torch::Tensor frequencies, torch::Tensor k_cache, torch::Tensor v_cache) {
  int batch = qkv.size(0), heads = k_cache.size(1), capacity = k_cache.size(2);
  auto q = torch::empty({batch, heads, HEAD_DIM}, qkv.options());
  rope_append_kernel<<<batch, heads * HEAD_DIM / 2, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()), positions.data_ptr<int>(), slot.data_ptr<int>(),
      frequencies.data_ptr<float>(), reinterpret_cast<bf16*>(q.data_ptr()),
      reinterpret_cast<bf16*>(k_cache.data_ptr()), reinterpret_cast<bf16*>(v_cache.data_ptr()), heads,
      capacity);
  return q;
}

torch::Tensor attention(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                        torch::Tensor start, torch::Tensor slot) {
  int batch = q.size(0), heads = q.size(1), capacity = k_cache.size(2);
  auto out = torch::empty_like(q);
  attention_kernel<<<batch * heads, 32 * ATTENTION_WARPS, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()), reinterpret_cast<const bf16*>(k_cache.data_ptr()),
      reinterpret_cast<const bf16*>(v_cache.data_ptr()), start.data_ptr<int>(), slot.data_ptr<int>(),
      reinterpret_cast<bf16*>(out.data_ptr()), heads, capacity, 1.f / sqrtf((float)HEAD_DIM));
  return out;
}

std::vector<torch::Tensor> sample(torch::Tensor logits, torch::Tensor uniforms, torch::Tensor step,
                                  double temperature) {
  int batch = logits.size(0), vocab = logits.size(1), threads = 1024;
  auto tokens = torch::empty({batch}, logits.options().dtype(torch::kLong));
  auto logprobs = torch::empty({batch}, logits.options());
  auto normalizers = torch::empty({batch}, logits.options());
  sample_kernel<<<batch, threads, (32 + threads) * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<float>(), uniforms.data_ptr<float>(), step.data_ptr<int>(),
      (float)(1.0 / temperature), vocab, batch, tokens.data_ptr<long>(), logprobs.data_ptr<float>(),
      normalizers.data_ptr<float>());
  return {tokens, logprobs, normalizers};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rope_append", &rope_append);
  m.def("attention", &attention);
  m.def("sample", &sample);
}
