// Batch-one decode step for the Moondream 2 text decoder (24 layers, width 2048).
//
// A layer is x += proj(attention(qkv(ln(x)))) + fc2(gelu(fc1(ln(x)))), with one layer
// norm shared by both branches. Each layer is three launches:
//
//   1. ln_qkv_fc1         layer norm, then the 6144 qkv rows and 8192 fc1 rows (+GELU)
//   2. attention          rotary embedding, key/value append, one block per head
//   3. proj_fc2_residual  proj and fc2 rows, added to the residual stream in place
//
// Weights stay in PyTorch's BF16 tensors and are read row by row; all arithmetic and
// the key/value cache are FP32. The host function `md_step` queues all 72 launches on
// the default stream, which PyTorch also uses, so no synchronization is needed
// between this step and the surrounding PyTorch operations.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

constexpr int LAYERS = 24;
constexpr int DIM = 2048;
constexpr int FF = 8192;
constexpr int HEADS = 32;
constexpr int HEAD_DIM = 64;
constexpr int ROT_PAIRS = 16;  // the first 32 dimensions of each head rotate
constexpr int QKV = 3 * DIM;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int ROWS_PER_WARP_1 = 4;  // launch 1: (6144 + 8192) rows / (8 warps * 4) = 448 blocks
constexpr int MAX_CONTEXT = 2048;
constexpr float LN_EPS = 1e-5f;

struct Layer {
    const uint16_t *ln_w, *ln_b, *qkv_w, *qkv_b, *proj_w, *proj_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};

struct State {
    float *x;          // [DIM] residual stream, input and output of a step
    float *qkv;        // [QKV]
    float *h;          // [FF] gelu(fc1)
    float *attn;       // [DIM] attention output before proj
    float *keys;       // [LAYERS][HEADS][MAX_CONTEXT][HEAD_DIM]
    float *values;     // same shape
    const float *rope; // [MAX_CONTEXT][ROT_PAIRS][2] cos, sin
};

static Layer g_layers[LAYERS];
static State g_state;

// BF16 is the upper half of an FP32 bit pattern.
__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(((uint32_t)v) << 16); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum (or maximum) of one value per thread across the block; `scratch` holds WARPS floats.
__device__ float block_sum(float v, float *scratch) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.f;
    for (int w = 0; w < WARPS; ++w) total += scratch[w];
    __syncthreads();
    return total;
}

__device__ float block_max(float v, float *scratch) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float best = scratch[0];
    for (int w = 1; w < WARPS; ++w) best = fmaxf(best, scratch[w]);
    __syncthreads();
    return best;
}

// Dot product of one BF16 weight row with an FP32 vector, computed by a whole warp.
// Each lane reads 16 bytes (8 weights) per iteration, so the warp reads 512 contiguous bytes.
__device__ __forceinline__ float warp_dot(const uint16_t *row, const float *vec, int n, int lane) {
    float acc = 0.f;
    for (int base = lane * 8; base < n; base += 256) {
        const uint4 u = *reinterpret_cast<const uint4 *>(row + base);
        const float *v = vec + base;
        acc += __uint_as_float(u.x << 16) * v[0] + __uint_as_float(u.x & 0xffff0000u) * v[1]
             + __uint_as_float(u.y << 16) * v[2] + __uint_as_float(u.y & 0xffff0000u) * v[3]
             + __uint_as_float(u.z << 16) * v[4] + __uint_as_float(u.z & 0xffff0000u) * v[5]
             + __uint_as_float(u.w << 16) * v[6] + __uint_as_float(u.w & 0xffff0000u) * v[7];
    }
    return warp_sum(acc);
}

__device__ __forceinline__ float gelu_tanh(float v) {
    return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

__global__ void ln_qkv_fc1(const float *x, Layer w, float *qkv, float *h) {
    __shared__ float xn[DIM];
    __shared__ float scratch[WARPS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    float sum = 0.f;
    for (int i = tid; i < DIM; i += THREADS) { xn[i] = x[i]; sum += xn[i]; }
    const float mean = block_sum(sum, scratch) / DIM;
    float sq = 0.f;
    for (int i = tid; i < DIM; i += THREADS) { const float d = xn[i] - mean; sq += d * d; }
    const float scale = rsqrtf(block_sum(sq, scratch) / DIM + LN_EPS);
    for (int i = tid; i < DIM; i += THREADS)
        xn[i] = (xn[i] - mean) * scale * bf(w.ln_w[i]) + bf(w.ln_b[i]);
    __syncthreads();

    const int first = (blockIdx.x * WARPS + warp) * ROWS_PER_WARP_1;
    for (int r = 0; r < ROWS_PER_WARP_1; ++r) {
        const int row = first + r;
        if (row < QKV) {
            const float v = warp_dot(w.qkv_w + (size_t)row * DIM, xn, DIM, lane);
            if (lane == 0) qkv[row] = v + bf(w.qkv_b[row]);
        } else if (row < QKV + FF) {
            const int j = row - QKV;
            const float v = warp_dot(w.fc1_w + (size_t)j * DIM, xn, DIM, lane);
            if (lane == 0) h[j] = gelu_tanh(v + bf(w.fc1_b[j]));
        }
    }
}

// One block per head. `keys`/`values` point at this layer's [HEADS][MAX_CONTEXT][HEAD_DIM].
__global__ void attention(const float *qkv, float *keys, float *values, const float *rope_row,
                          int pos, float *out) {
    __shared__ float raw_q[HEAD_DIM], raw_k[HEAD_DIM], q[HEAD_DIM];
    __shared__ float score[MAX_CONTEXT];
    __shared__ float partial[4][HEAD_DIM];
    __shared__ float scratch[WARPS];
    const int tid = threadIdx.x, head = blockIdx.x;
    float *head_keys = keys + (size_t)head * MAX_CONTEXT * HEAD_DIM;
    float *head_values = values + (size_t)head * MAX_CONTEXT * HEAD_DIM;

    if (tid < HEAD_DIM) {
        raw_q[tid] = qkv[head * HEAD_DIM + tid];
        raw_k[tid] = qkv[DIM + head * HEAD_DIM + tid];
        head_values[(size_t)pos * HEAD_DIM + tid] = qkv[2 * DIM + head * HEAD_DIM + tid];
    }
    __syncthreads();
    if (tid < HEAD_DIM) {
        float qv = raw_q[tid], kv = raw_k[tid];
        if (tid < 2 * ROT_PAIRS) {
            // Real parts are dimensions [0, 16), imaginary parts [16, 32); the rotated
            // pairs are written interleaved, as the reference implementation does.
            const int pair = tid >> 1;
            const float c = rope_row[2 * pair], s = rope_row[2 * pair + 1];
            const float qr = raw_q[pair], qi = raw_q[ROT_PAIRS + pair];
            const float kr = raw_k[pair], ki = raw_k[ROT_PAIRS + pair];
            qv = (tid & 1) ? qr * s + qi * c : qr * c - qi * s;
            kv = (tid & 1) ? kr * s + ki * c : kr * c - ki * s;
        }
        q[tid] = qv;
        head_keys[(size_t)pos * HEAD_DIM + tid] = kv;
    }
    __syncthreads();

    const int count = pos + 1;
    float best = -INFINITY;
    for (int t = tid; t < count; t += THREADS) {
        const float *key = head_keys + (size_t)t * HEAD_DIM;
        float acc = 0.f;
        for (int d = 0; d < HEAD_DIM; ++d) acc += q[d] * key[d];
        acc *= 0.125f;  // 1 / sqrt(HEAD_DIM)
        score[t] = acc;
        best = fmaxf(best, acc);
    }
    best = block_max(best, scratch);
    float sum = 0.f;
    for (int t = tid; t < count; t += THREADS) {
        const float e = __expf(score[t] - best);
        score[t] = e;
        sum += e;
    }
    const float inv = 1.f / block_sum(sum, scratch);

    // Thread (part, d) accumulates dimension d over positions congruent to part mod 4.
    const int d = tid & (HEAD_DIM - 1), part = tid >> 6;
    float acc = 0.f;
    for (int t = part; t < count; t += 4) acc += score[t] * head_values[(size_t)t * HEAD_DIM + d];
    partial[part][d] = acc;
    __syncthreads();
    if (tid < HEAD_DIM)
        out[head * HEAD_DIM + tid] =
            (partial[0][tid] + partial[1][tid] + partial[2][tid] + partial[3][tid]) * inv;
}

// One row per warp: x[row] += proj[row] . attn + fc2[row] . h + biases.
__global__ void proj_fc2_residual(float *x, Layer w, const float *attn, const float *h) {
    __shared__ float a[DIM];
    __shared__ float hs[FF];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < DIM; i += THREADS) a[i] = attn[i];
    for (int i = tid; i < FF; i += THREADS) hs[i] = h[i];
    __syncthreads();
    const int row = blockIdx.x * WARPS + warp;
    const float v = warp_dot(w.proj_w + (size_t)row * DIM, a, DIM, lane)
                  + warp_dot(w.fc2_w + (size_t)row * FF, hs, FF, lane);
    if (lane == 0) x[row] += v + bf(w.proj_b[row]) + bf(w.fc2_b[row]);
}

extern "C" const char *md_error_string(int code) { return cudaGetErrorString((cudaError_t)code); }

// `layers` holds LAYERS * 10 device pointers in Layer field order; `state` holds the
// seven State pointers.
extern "C" int md_init(void **layers, void **state) {
    for (int l = 0; l < LAYERS; ++l) {
        const uint16_t **p = (const uint16_t **)(layers + l * 10);
        g_layers[l] = Layer{p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]};
    }
    g_state = State{(float *)state[0], (float *)state[1], (float *)state[2], (float *)state[3],
                    (float *)state[4], (float *)state[5], (const float *)state[6]};
    return (int)cudaGetLastError();
}

// Advance the residual stream in `state.x` through all layers at position `pos`,
// appending this position's keys and values. Returns without waiting for the GPU.
extern "C" int md_step(int pos) {
    if (pos < 0 || pos >= MAX_CONTEXT) return (int)cudaErrorInvalidValue;
    const State &s = g_state;
    const size_t layer_cache = (size_t)HEADS * MAX_CONTEXT * HEAD_DIM;
    const float *rope_row = s.rope + (size_t)pos * ROT_PAIRS * 2;
    const int blocks_1 = (QKV + FF) / (WARPS * ROWS_PER_WARP_1);
    for (int l = 0; l < LAYERS; ++l) {
        ln_qkv_fc1<<<blocks_1, THREADS>>>(s.x, g_layers[l], s.qkv, s.h);
        attention<<<HEADS, THREADS>>>(s.qkv, s.keys + l * layer_cache, s.values + l * layer_cache,
                                      rope_row, pos, s.attn);
        proj_fc2_residual<<<DIM / WARPS, THREADS>>>(s.x, g_layers[l], s.attn, s.h);
    }
    return (int)cudaGetLastError();
}
