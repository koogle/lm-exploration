// Adapted from Luce-Org/lucebox ad4de665 (Apache-2.0); see NOTICE.md.
// Modified for Qwen3.5-9B, grouped DeltaNet heads and cooperative-grid launch.
/**
 * Fused single-kernel decode for Qwen3.5-9B (hybrid DeltaNet + Full Attention).
 * Supports Pascal (sm_6x), Volta/Turing (sm_70-75), and Ampere+ (sm_80+).
 * BF16 for sm_80+, F16 for sm_6x-sm_75. Accumulation always f32.
 * DeltaNet state: f32 (recurrence needs precision).
 *
 * Pascal-specific adaptations (via TARGET_SM guards):
 *   - __shfl / __shfl_down instead of __shfl_sync / __shfl_down_sync
 *   - membar.gl instead of fence.acq_rel.gpu
 *   - __ldg() instead of ld.global.cg / ld.global.L1::no_allocate
 *   - No tensor cores, no WMMA, no cp.async
 */

#include "half_type.h"
#include "rope.h"
#ifdef USE_THUNDERKITTENS
#include "kittens.cuh"
#endif

// ── Pascal (sm_6x) compatibility shims ──
// Pascal lacks the _sync suffixed warp shuffle and the fence.acq_rel.gpu
// PTX instruction.  Provide thin wrappers so the same source compiles for
// both Pascal and Volta+.
#if TARGET_SM >= 70
// Volta+: native _sync shuffles and fence.acq_rel.gpu
#define SHFL_SYNC(mask, val, lane)    __shfl_sync((mask), (val), (lane))
#define SHFL_DOWN_SYNC(mask, val, d)  __shfl_down_sync((mask), (val), (d))
#define GPU_MEMORY_FENCE()            asm volatile("fence.acq_rel.gpu;" ::: "memory")
#else
// Pascal: no mask argument, no _sync suffix; membar.gl for global fence
#define SHFL_SYNC(mask, val, lane)    __shfl((val), (lane))
#define SHFL_DOWN_SYNC(mask, val, d)  __shfl_down((val), (d))
#define GPU_MEMORY_FENCE()            asm volatile("membar.gl;" ::: "memory")
#endif
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// =============================================================================
// Model constants
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 4096;
constexpr int INTERMEDIATE_SIZE = 12288;
constexpr int NUM_LAYERS = 32;
constexpr float RMS_EPS = 1e-6f;
constexpr int VOCAB_SIZE = 248320;

// Full Attention
constexpr int FA_NUM_Q_HEADS = 16;
constexpr int FA_NUM_KV_HEADS = 4;
constexpr int FA_HEAD_DIM = 256;
constexpr int FA_GQA_RATIO = FA_NUM_Q_HEADS / FA_NUM_KV_HEADS;
constexpr int FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM;
constexpr int FA_GATE_SIZE = FA_Q_SIZE;
constexpr int FA_QPROJ_SIZE = FA_Q_SIZE + FA_GATE_SIZE;
constexpr int FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM;
constexpr int FA_ROTARY_DIM = 64;
constexpr float FA_ROPE_THETA = 10000000.0f;

// DeltaNet
constexpr int DN_NUM_HEADS = 32; // value/state heads
constexpr int DN_KEY_HEADS = 16;
constexpr int DN_HEAD_GROUP = DN_NUM_HEADS / DN_KEY_HEADS;
constexpr int DN_KEY_DIM = 128;
constexpr int DN_VALUE_DIM = 128;
constexpr int DN_CONV_KERNEL = 4;
constexpr int DN_QK_SIZE = DN_KEY_HEADS * DN_KEY_DIM;
constexpr int DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM;
constexpr int DN_CONV_CHANNELS = DN_QK_SIZE + DN_QK_SIZE + DN_V_SIZE;

constexpr int MAX_ACT_DIM = (HIDDEN_SIZE > INTERMEDIATE_SIZE) ? HIDDEN_SIZE : INTERMEDIATE_SIZE;

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 132
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
constexpr int SHMEM_FLOATS = (MAX_ACT_DIM > NUM_WARPS * FA_HEAD_DIM)
    ? MAX_ACT_DIM : NUM_WARPS * FA_HEAD_DIM;

#ifndef LM_NUM_BLOCKS
#define LM_NUM_BLOCKS 512
#endif
#ifndef LM_BLOCK_SIZE
#define LM_BLOCK_SIZE 256
#endif

static int g_decode_blocks_override = 0;

__device__ __constant__ int LAYER_TYPE[NUM_LAYERS] = {
    0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
};

// =============================================================================
// Weight structs — ALL BF16
// =============================================================================

struct FullAttnWeights {
    const half_t *input_layernorm_weight;   // [1024]
    const half_t *q_proj_weight;             // [4096, 1024]
    const half_t *k_proj_weight;             // [512, 1024]
    const half_t *v_proj_weight;             // [512, 1024]
    const half_t *q_norm_weight;              // [256]
    const half_t *k_norm_weight;              // [256]
    const half_t *o_proj_weight;             // [1024, 2048]
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;          // [3584, 1024]
    const half_t *up_proj_weight;            // [3584, 1024]
    const half_t *down_proj_weight;          // [1024, 3584]
};

struct DeltaNetWeights {
    const half_t *input_layernorm_weight;
    const half_t *qkv_proj_weight;           // [6144, 1024]
    const half_t *z_proj_weight;             // [2048, 1024]
    const half_t *beta_proj_weight;          // [16, 1024]
    const half_t *alpha_proj_weight;         // [16, 1024]
    const half_t *conv1d_weight;             // [6144, 1, 4]
    const float *a_log;                     // [16]
    const float *dt_bias;                   // [16]
    const half_t *norm_weight;               // [128]
    const half_t *out_proj_weight;           // [1024, 2048]
    const half_t *post_attn_layernorm_weight;
    const half_t *gate_proj_weight;
    const half_t *up_proj_weight;
    const half_t *down_proj_weight;
};

struct LayerWeights {
    int layer_type;
    int _pad[3];
    union {
        DeltaNetWeights dn;
        FullAttnWeights fa;
    };
};

// =============================================================================
// Atomic barrier
// =============================================================================

struct AtomicGridSync {
    unsigned int *counter;
    unsigned int *generation;
    unsigned int nblocks;
    unsigned int local_gen;

    __device__ void sync() { cooperative_groups::this_grid().sync(); }

};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += SHFL_DOWN_SYNC(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float fast_exp(float x) {
    float y; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.44269504088896340736f)); return y;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    float y; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(1.0f + fast_exp(-x))); return y;
}

__device__ __forceinline__ float fast_silu(float x) { return x * fast_sigmoid(x); }

__device__ __forceinline__ uint4 load_128bit(const uint4 *ptr) {
    uint4 out;
#if TARGET_SM >= 80
    // Ampere+: bypass L1 to avoid read-for-ownership pollution
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
#elif TARGET_SM >= 70
    // Volta/Turing: .cg (cache global) for read-only coalesced access
    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
#else
    // Pascal: no .cg cache modifier — use __ldg (cached read-only)
    out = __ldg(ptr);
#endif
    return out;
}

// BF16 dot product: 8 bf16 weights × 8 bf16 activations → f32
__device__ __forceinline__ float dot8_bf16(const uint4 &w_u4, const half_t *act) {
    const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++)
        sum += H2F(w[i]) * H2F(act[i]);
    return sum;
}

// =============================================================================
// RMSNorm — reads bf16 input, writes bf16 output
// =============================================================================

__device__ void rmsnorm_redundant(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,        // shared memory bf16
    half_t *__restrict__ g_residual)   // global bf16
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = H2F(__ldg(input + i));
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// RMSNorm from bf16 buffer (for post-attn norm)
__device__ void rmsnorm_from_bf16(
    const half_t *__restrict__ input,
    const half_t *__restrict__ weight,
    half_t *__restrict__ s_out,
    half_t *__restrict__ g_residual)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ float smem_reduce[NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float v = H2F(input[i]);
        s_out[i] = F2H(v);
        local_sum_sq += v * v;
    }

    if (block_id == 0) {
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE)
            g_residual[i] = s_out[i];
    }

    local_sum_sq = warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0)
            smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + RMS_EPS);
    }
    __syncthreads();

    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
        float w = H2F(__ldg(weight + i));
        float v = H2F(s_out[i]);
        s_out[i] = F2H(v * rstd * (1.0f + w));
    }
    __syncthreads();
}

// =============================================================================
// BF16 Matvec: warp-per-row, activations in shared memory (bf16)
// =============================================================================

__device__ void matvec_bf16(
    const half_t *__restrict__ s_input,  // shared memory bf16 [in_dim]
    const half_t *__restrict__ weight,   // [out_dim, in_dim] bf16
    float *__restrict__ output,                  // [out_dim] f32 (accumulate in f32)
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
#ifdef USE_THUNDERKITTENS
            // A real TK register-vector projection variant. Keep the same
            // vectorized weight loads and BF16 output boundary as the control.
            // This is SIMD GEMV, not a tensor-core or whole-model TK rewrite.
            kittens::rv_fl<256> accum, weights, inputs, products;
            accum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 packed = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *values = reinterpret_cast<const half_t *>(&packed);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    weights[i][0] = H2F(values[i]);
                    inputs[i][0] = H2F(s_input[k + i]);
                }
                kittens::warp::mul(products, weights, inputs);
                kittens::warp::add(accum, accum, products);
            }
            kittens::warp::sum(sum, accum);
#else
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                sum += dot8_bf16(w_u4, s_input + k);
            }
            sum = warp_reduce_sum(sum);
#endif
            if (lane_id == 0) output[m] = H2F(F2H(sum));
        }
    }
}

// Fused gate+up+SiLU matvec (bf16 weights, bf16 activations)
__device__ void matvec_gate_up_silu_bf16(
    const half_t *__restrict__ s_input,
    const half_t *__restrict__ gate_weight,
    const half_t *__restrict__ up_weight,
    float *__restrict__ output,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *g_row = gate_weight + m * in_dim;
            const half_t *u_row = up_weight + m * in_dim;
            float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 g_u4 = load_128bit(reinterpret_cast<const uint4 *>(g_row + k));
                uint4 u_u4 = load_128bit(reinterpret_cast<const uint4 *>(u_row + k));
                gate_sum += dot8_bf16(g_u4, s_input + k);
                up_sum += dot8_bf16(u_u4, s_input + k);
            }
            gate_sum = warp_reduce_sum(gate_sum);
            up_sum = warp_reduce_sum(up_sum);
            if (lane_id == 0) {
                float g = H2F(F2H(gate_sum)), u = H2F(F2H(up_sum));
                output[m] = H2F(F2H(H2F(F2H(fast_silu(g))) * u));
            }
        }
    }
}

// Down projection + residual → bf16 hidden
__device__ void matvec_down_residual_bf16(
    const float *__restrict__ s_input,           // shared [INTER] f32
    const half_t *__restrict__ weight,    // [HIDDEN, INTER] bf16
    const half_t *__restrict__ residual,  // [HIDDEN] bf16
    half_t *__restrict__ hidden_out,      // [HIDDEN] bf16
    int in_dim, int out_dim, int num_blocks)
{
    // This needs f32 input (MLP intermediate is f32). Convert on the fly.
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            // Weight is bf16, input is f32 — convert input to bf16 on the fly
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = F2H(H2F(F2H(sum)) + H2F(residual[m]));
        }
    }
}

// O projection + residual → bf16
__device__ void matvec_o_residual_bf16(
    const float *__restrict__ s_input,           // shared [Q_SIZE] f32
    const half_t *__restrict__ weight,
    const half_t *__restrict__ residual,
    half_t *__restrict__ hidden_out,
    int in_dim, int out_dim, int num_blocks)
{
    int block_id = blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (out_dim + num_blocks - 1) / num_blocks;
    int row_start = block_id * rows_per_block;
    int row_end = min(row_start + rows_per_block, out_dim);

    for (int m_base = row_start; m_base < row_end; m_base += NUM_WARPS) {
        int m = m_base + warp_id;
        if (m < row_end) {
            const half_t *w_row = weight + m * in_dim;
            float sum = 0.0f;
            for (int k = lane_id * 8; k < in_dim; k += WARP_SIZE * 8) {
                uint4 w_u4 = load_128bit(reinterpret_cast<const uint4 *>(w_row + k));
                const half_t *w = reinterpret_cast<const half_t *>(&w_u4);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    sum += H2F(w[i]) * s_input[k + i];
            }
            sum = warp_reduce_sum(sum);
            if (lane_id == 0)
                hidden_out[m] = F2H(H2F(F2H(sum)) + H2F(residual[m]));
        }
    }
}

// =============================================================================
// Full Attention layer (bf16)
// =============================================================================

__device__ void full_attention_layer(
    AtomicGridSync &grid,
    const FullAttnWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ k_cache,
    half_t *__restrict__ v_cache,
    half_t *__restrict__ g_residual,  // [HIDDEN] bf16
    float *__restrict__ g_activations,        // scratch f32
    float *__restrict__ g_q,                  // [FA_QPROJ_SIZE] f32
    float *__restrict__ g_kv,                 // [FA_KV_SIZE*2] f32
    float *__restrict__ g_attn_out,           // [FA_Q_SIZE] f32
    float *__restrict__ g_mlp_inter,          // [INTER] f32
    half_t *__restrict__ hidden_out,   // [HIDDEN] bf16
    int position, int max_seq_len,
    half_t *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + QKV projection
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.q_proj_weight, g_q, HIDDEN_SIZE, FA_QPROJ_SIZE, num_blocks);
    matvec_bf16(s_norm, w.k_proj_weight, g_kv, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    matvec_bf16(s_norm, w.v_proj_weight, g_kv + FA_KV_SIZE, HIDDEN_SIZE, FA_KV_SIZE, num_blocks);
    grid.sync();

    // Phase 2: QK norm + partial RoPE + KV cache write
    if (block_id == 0) {
        float *k_buf = g_kv, *v_buf = g_kv + FA_KV_SIZE;
        for (int h = warp_id; h < FA_NUM_KV_HEADS; h += NUM_WARPS) {
            float *kh = k_buf + h * FA_HEAD_DIM, *vh = v_buf + h * FA_HEAD_DIM;
            half_t *kc = k_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            half_t *vc = v_cache + h * max_seq_len * FA_HEAD_DIM + position * FA_HEAD_DIM;
            float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += kh[i]*kh[i];
            ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
            sc = SHFL_SYNC(0xffffffff, sc, 0);
            for (int i = lane_id; i < FA_ROTARY_DIM/2; i += WARP_SIZE) {
                int p = i + FA_ROTARY_DIM/2;
                float fe = float(2*i) / float(FA_ROTARY_DIM);
                float freq = float(position) / powf(FA_ROPE_THETA, fe);
                float cv = cosf(freq), sv = sinf(freq);
                apply_rope_pair(kh, kc, w.k_norm_weight, i, p, sc, cv, sv);
                vc[i] = F2H(vh[i]); vc[p] = F2H(vh[p]);
            }
            for (int i = FA_ROTARY_DIM + lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                kc[i] = F2H(kh[i]*sc*(1.0f+H2F(__ldg(w.k_norm_weight+i))));
                vc[i] = F2H(vh[i]);
            }
        }
    }
    // Q norm + RoPE (all blocks)
    {
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        for (int qh = hs; qh < he; qh++) {
            float *qh_ptr = g_q + qh * FA_HEAD_DIM * 2;
            if (warp_id == 0) {
                float ss = 0; for (int i = lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) ss += qh_ptr[i]*qh_ptr[i];
                ss = warp_reduce_sum(ss); float sc = rsqrtf(ss / float(FA_HEAD_DIM) + RMS_EPS);
                sc = SHFL_SYNC(0xffffffff, sc, 0);
                for (int i = lane_id; i < FA_ROTARY_DIM/2; i += WARP_SIZE) {
                    int p = i + FA_ROTARY_DIM/2;
                    float fe = float(2*i) / float(FA_ROTARY_DIM);
                    float freq = float(position) / powf(FA_ROPE_THETA, fe);
                    float cv = cosf(freq), sv = sinf(freq);
                    apply_rope_pair(qh_ptr, qh_ptr, w.q_norm_weight, i, p, sc, cv, sv);
                }
                for (int i = FA_ROTARY_DIM + lane_id; i < FA_HEAD_DIM; i += WARP_SIZE) {
                    qh_ptr[i] = H2F(F2H(qh_ptr[i]*sc*(1.0f+H2F(__ldg(w.q_norm_weight+i)))));
                }
            }
        }
    }
    grid.sync();

    // Phase 3: Attention decode (online softmax + sigmoid gate)
    {
        int cache_len = position + 1;
        float attn_scale = 1.0f / sqrtf(float(FA_HEAD_DIM));
        int hpb = (FA_NUM_Q_HEADS + num_blocks - 1) / num_blocks;
        int hs = block_id * hpb, he = min(hs + hpb, FA_NUM_Q_HEADS);
        __shared__ float s_max_score[NUM_WARPS];
        __shared__ float s_sum_exp[NUM_WARPS];
        float *s_attn_scratch = reinterpret_cast<float *>(shmem);
        constexpr int EPL = FA_HEAD_DIM / WARP_SIZE;

        for (int qh = hs; qh < he; qh++) {
            int kvh = qh / FA_GQA_RATIO;
            float *q_head = g_q + qh * FA_HEAD_DIM * 2;
            float *out_head = g_attn_out + qh * FA_HEAD_DIM;
            float max_score = -INFINITY, sum_exp = 0;
            float out_acc[EPL], q_local[EPL];
            for (int e = 0; e < EPL; e++) { out_acc[e] = 0; q_local[e] = q_head[lane_id*EPL+e]; }

            for (int pos = warp_id; pos < cache_len; pos += NUM_WARPS) {
                const half_t *k_pos = k_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                const half_t *v_pos = v_cache + kvh*max_seq_len*FA_HEAD_DIM + pos*FA_HEAD_DIM;
                float score = 0;
                for (int e = 0; e < EPL; e++) score += q_local[e] * H2F(__ldg(k_pos + lane_id*EPL+e));
                score = warp_reduce_sum(score) * attn_scale;
                score = SHFL_SYNC(0xffffffff, score, 0);
                float old_max = max_score; max_score = fmaxf(max_score, score);
                float exp_diff = fast_exp(old_max - max_score);
                sum_exp = sum_exp * exp_diff + fast_exp(score - max_score);
                float wt = fast_exp(score - max_score);
                for (int e = 0; e < EPL; e++)
                    out_acc[e] = out_acc[e]*exp_diff + wt*H2F(__ldg(v_pos + lane_id*EPL+e));
            }
            if (lane_id == 0) { s_max_score[warp_id] = max_score; s_sum_exp[warp_id] = sum_exp; }
            for (int e = 0; e < EPL; e++) s_attn_scratch[warp_id*FA_HEAD_DIM + lane_id*EPL+e] = out_acc[e];
            __syncthreads();

            if (warp_id == 0) {
                float gm = -INFINITY; for (int ww = 0; ww < NUM_WARPS; ww++) if (s_max_score[ww] > -INFINITY) gm = fmaxf(gm, s_max_score[ww]);
                float ts = 0; float fo[EPL]; for (int e = 0; e < EPL; e++) fo[e] = 0;
                for (int ww = 0; ww < NUM_WARPS; ww++) {
                    if (s_max_score[ww] > -INFINITY) {
                        float s = fast_exp(s_max_score[ww]-gm); ts += s_sum_exp[ww]*s;
                        for (int e = 0; e < EPL; e++) fo[e] += s_attn_scratch[ww*FA_HEAD_DIM+lane_id*EPL+e]*s;
                    }
                }
                float *gate_ptr = q_head + FA_HEAD_DIM;
                float rcp = 1.0f / ts;
                for (int e = 0; e < EPL; e++) {
                    int idx = lane_id*EPL+e;
                    out_head[idx] = H2F(F2H(H2F(F2H(fo[e]*rcp)) * H2F(F2H(fast_sigmoid(gate_ptr[idx])))));
                }
            }
            __syncthreads();
        }
    }
    grid.sync();

    // Phase 4: O projection + residual → bf16
    {
        float *s_attn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < FA_Q_SIZE; i += BLOCK_SIZE) s_attn[i] = g_attn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_attn, w.o_proj_weight, g_residual, hidden_out, FA_Q_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    // Load MLP intermediate to shared (f32)
    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();

    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

// =============================================================================
// DeltaNet layer (bf16) — warp-cooperative state-in-registers recurrence
// =============================================================================

__device__ void deltanet_layer(
    AtomicGridSync &grid,
    const DeltaNetWeights &w,
    const half_t *__restrict__ input,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_activations,
    float *__restrict__ g_qkv,
    float *__restrict__ g_z,
    float *__restrict__ g_beta,
    float *__restrict__ g_alpha,
    float *__restrict__ g_dn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ dn_state,     // [DN_NUM_HEADS, DN_KEY, DN_VAL] f32
    float *__restrict__ conv_buf,     // [DN_CONV_CH, DN_CONV_K] f32
    half_t *__restrict__ hidden_out,
    int dn_layer_idx,
    half_t *__restrict__ shmem)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Phase 1: RMSNorm + projections
    half_t *s_norm = shmem;
    rmsnorm_redundant(input, w.input_layernorm_weight, s_norm, g_residual);

    matvec_bf16(s_norm, w.qkv_proj_weight, g_qkv, HIDDEN_SIZE, DN_CONV_CHANNELS, num_blocks);
    matvec_bf16(s_norm, w.z_proj_weight, g_z, HIDDEN_SIZE, DN_V_SIZE, num_blocks);
    matvec_bf16(s_norm, w.beta_proj_weight, g_beta, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    matvec_bf16(s_norm, w.alpha_proj_weight, g_alpha, HIDDEN_SIZE, DN_NUM_HEADS, num_blocks);
    grid.sync();

    // Phase 2: convolution owns each source channel once. Q/K are shared by
    // two value heads; updating their history inside each value-head block races.
    float *layer_conv = conv_buf + dn_layer_idx * DN_CONV_CHANNELS * DN_CONV_KERNEL;
    for (int ch = block_id * BLOCK_SIZE + threadIdx.x; ch < DN_CONV_CHANNELS;
         ch += num_blocks * BLOCK_SIZE) {
        float *history = layer_conv + ch * DN_CONV_KERNEL;
        history[0] = history[1]; history[1] = history[2]; history[2] = history[3];
        history[3] = g_qkv[ch];
        float co = 0;
        for (int t=0; t<DN_CONV_KERNEL; ++t)
            co += history[t] * H2F(w.conv1d_weight[ch*DN_CONV_KERNEL+t]);
        g_qkv[ch] = H2F(F2H(fast_silu(H2F(F2H(co)))));
    }
    grid.sync();
    // Phase 3: one value-head state per CTA; h/2 selects its shared Q/K head.
    if (block_id < DN_NUM_HEADS) {
        int h = block_id, kh = h / DN_HEAD_GROUP;
        __shared__ float s_q[DN_KEY_DIM], s_k[DN_KEY_DIM], s_v[DN_VALUE_DIM];
        for (int i=threadIdx.x; i<DN_KEY_DIM; i+=BLOCK_SIZE) {
            s_q[i] = g_qkv[kh*DN_KEY_DIM+i];
            s_k[i] = g_qkv[DN_QK_SIZE+kh*DN_KEY_DIM+i];
        }
        for (int i=threadIdx.x; i<DN_VALUE_DIM; i+=BLOCK_SIZE)
            s_v[i] = g_qkv[2*DN_QK_SIZE+h*DN_VALUE_DIM+i];
        // Beta/alpha activations
        if (threadIdx.x == 0) {
            g_beta[h] = H2F(F2H(fast_sigmoid(g_beta[h])));
            float a_log_val = __ldg(w.a_log + h);
            float dt_b = __ldg(w.dt_bias + h);
            float x = g_alpha[h] + dt_b;
            float sp = (x > 20.0f) ? x : logf(1.0f + fast_exp(x));
            g_alpha[h] = fast_exp(-fast_exp(a_log_val) * sp);
        }
        __syncthreads();

        // L2 normalize Q, K
        constexpr float Q_SCALE = 1.0f / 11.313708498984761f;
        if (warp_id == 0) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_q[i]*s_q[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f)*Q_SCALE;
            n = SHFL_SYNC(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_q[i] *= n;
        }
        if (warp_id == 1) {
            float sq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) sq += s_k[i]*s_k[i];
            sq = warp_reduce_sum(sq); float n = rsqrtf(sq+1e-6f);
            n = SHFL_SYNC(0xffffffff,n,0); for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) s_k[i] *= n;
        }
        __syncthreads();

        float decay = g_alpha[h], beta = g_beta[h];

        // k·q dot
        __shared__ float s_kq;
        if (warp_id == 0) {
            float kq = 0; for (int i = lane_id; i < DN_KEY_DIM; i += WARP_SIZE) kq += s_k[i]*s_q[i];
            kq = warp_reduce_sum(kq); if (lane_id == 0) s_kq = kq;
        }
        __syncthreads();
        float kq = s_kq;

        // Warp-cooperative recurrence (state in global memory — decode is 1 token, fine)
        float *state = dn_state + h * DN_KEY_DIM * DN_VALUE_DIM;
        float *out_head = g_dn_out + h * DN_VALUE_DIM;

        constexpr int J_PER_WARP = DN_VALUE_DIM / NUM_WARPS;
        constexpr int I_PER_LANE = DN_KEY_DIM / WARP_SIZE;

#pragma unroll
        for (int jj = 0; jj < J_PER_WARP; jj++) {
            int j = warp_id * J_PER_WARP + jj;
            float s_regs[I_PER_LANE], stk = 0, sqv = 0;
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                float sv = state[j*DN_KEY_DIM+i]; s_regs[ii] = sv;
                stk += sv * s_k[i]; sqv += sv * s_q[i];
            }
            stk = warp_reduce_sum(stk); sqv = warp_reduce_sum(sqv);
            stk = SHFL_SYNC(0xffffffff,stk,0); sqv = SHFL_SYNC(0xffffffff,sqv,0);
            float error_j = (s_v[j] - decay * stk) * beta;
            float o_j = decay * sqv + error_j * kq;
            if (lane_id == 0) out_head[j] = H2F(F2H(o_j));
#pragma unroll
            for (int ii = 0; ii < I_PER_LANE; ii++) {
                int i = lane_id + ii * WARP_SIZE;
                state[j*DN_KEY_DIM+i] = s_regs[ii] * decay + s_k[i] * error_j;
            }
        }

        // Gated RMSNorm
        __syncthreads();
        {
            __shared__ float smem_gnorm[NUM_WARPS];
            float sq = 0; for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) sq += out_head[i]*out_head[i];
            sq = warp_reduce_sum(sq); if (lane_id == 0) smem_gnorm[warp_id] = sq; __syncthreads();
            if (warp_id == 0) { float v = (lane_id < NUM_WARPS) ? smem_gnorm[lane_id] : 0; v = warp_reduce_sum(v); if (lane_id == 0) smem_gnorm[0] = rsqrtf(v/DN_VALUE_DIM + RMS_EPS); }
            __syncthreads(); float rstd = smem_gnorm[0];
            for (int i = threadIdx.x; i < DN_VALUE_DIM; i += BLOCK_SIZE) {
                float normed = H2F(F2H(H2F(F2H(out_head[i] * rstd)) * H2F(__ldg(w.norm_weight + i))));
                float gate = fast_silu(g_z[h*DN_VALUE_DIM+i]);
                out_head[i] = H2F(F2H(normed * gate));
            }
        }
    } else {
        // Idle blocks: could prefetch weights
    }
    grid.sync();

    // Phase 4: Out projection + residual → bf16
    {
        float *s_dn = reinterpret_cast<float *>(shmem);
        for (int i = threadIdx.x; i < DN_V_SIZE; i += BLOCK_SIZE) s_dn[i] = g_dn_out[i];
        __syncthreads();
        matvec_o_residual_bf16(s_dn, w.out_proj_weight, g_residual, hidden_out, DN_V_SIZE, HIDDEN_SIZE, num_blocks);
    }
    grid.sync();

    // Phase 5: Post-attn norm + MLP
    half_t *s_act = shmem;
    rmsnorm_from_bf16(hidden_out, w.post_attn_layernorm_weight, s_act, g_residual);

    matvec_gate_up_silu_bf16(s_act, w.gate_proj_weight, w.up_proj_weight,
                              g_mlp_inter, HIDDEN_SIZE, INTERMEDIATE_SIZE, num_blocks);
    grid.sync();

    float *s_mlp = reinterpret_cast<float *>(shmem);
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += BLOCK_SIZE) s_mlp[i] = g_mlp_inter[i];
    __syncthreads();
    matvec_down_residual_bf16(s_mlp, w.down_proj_weight, g_residual, hidden_out,
                               INTERMEDIATE_SIZE, HIDDEN_SIZE, num_blocks);
    grid.sync();
}

// =============================================================================
// Main decode kernel
// =============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
decode_kernel(
    const half_t *__restrict__ embed_weight,
    const half_t *__restrict__ final_norm_weight,
    const half_t *__restrict__ lm_head_weight,
    const LayerWeights *__restrict__ layer_weights,
    half_t *__restrict__ fa_k_cache,
    half_t *__restrict__ fa_v_cache,
    float *__restrict__ dn_states,
    float *__restrict__ conv_bufs,
    half_t *__restrict__ hidden_buffer,
    float *__restrict__ g_activations,
    half_t *__restrict__ g_residual,
    float *__restrict__ g_qkv_scratch,
    float *__restrict__ g_kv_scratch,
    float *__restrict__ g_attn_out,
    float *__restrict__ g_mlp_inter,
    float *__restrict__ g_z_scratch,
    float *__restrict__ g_beta_scratch,
    float *__restrict__ g_alpha_scratch,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_generation,
    float *__restrict__ seen_token_mask,
    float repetition_penalty,
    int input_token_id, int position, int max_seq_len, int debug)
{
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;

    if (block_id == 0 && threadIdx.x == 0 &&
        seen_token_mask && repetition_penalty > 1.0f &&
        input_token_id >= 0 && input_token_id < VOCAB_SIZE) {
        seen_token_mask[input_token_id] = 1.0f;
    }

    AtomicGridSync grid{barrier_counter, barrier_generation, (unsigned int)num_blocks, 0};

    // Block-local workspace for normalization, MLP and attention merge.
    extern __shared__ __align__(16) char shmem_raw[];
    half_t *shmem_bf16 = reinterpret_cast<half_t *>(shmem_raw);

    const half_t *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

    int fa_kv_stride = FA_NUM_KV_HEADS * max_seq_len * FA_HEAD_DIM;
    int dn_state_stride = DN_NUM_HEADS * DN_KEY_DIM * DN_VALUE_DIM;

    int dn_layer_idx = 0, fa_layer_idx = 0;

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        const half_t *layer_input = (layer == 0) ? embed_row : hidden_buffer;

        if (LAYER_TYPE[layer] == 0) {
            deltanet_layer(
                grid, layer_weights[layer].dn, layer_input,
                g_residual, g_activations, g_qkv_scratch, g_z_scratch,
                g_beta_scratch, g_alpha_scratch, g_attn_out, g_mlp_inter,
                dn_states + dn_layer_idx * dn_state_stride,
                conv_bufs, hidden_buffer, dn_layer_idx, shmem_bf16);
            dn_layer_idx++;
        } else {
            full_attention_layer(
                grid, layer_weights[layer].fa, layer_input,
                fa_k_cache + fa_layer_idx * fa_kv_stride,
                fa_v_cache + fa_layer_idx * fa_kv_stride,
                g_residual, g_activations, g_qkv_scratch, g_kv_scratch,
                g_attn_out, g_mlp_inter, hidden_buffer,
                position, max_seq_len, shmem_bf16);
            fa_layer_idx++;
        }
        if (debug) {
            if (block_id == 0)
                for (int i=threadIdx.x; i<HIDDEN_SIZE; i+=BLOCK_SIZE)
                    g_activations[(layer+1)*HIDDEN_SIZE+i] = H2F(hidden_buffer[i]);
            grid.sync();
        }
    }

    // Final RMSNorm (block 0 only)
    if (block_id == 0) {
        __shared__ float smem_reduce[NUM_WARPS];
        int warp_id = threadIdx.x / WARP_SIZE, lane_id = threadIdx.x % WARP_SIZE;
        float local_sum_sq = 0;
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float v = H2F(hidden_buffer[i]); g_activations[i] = v; local_sum_sq += v*v;
        }
        local_sum_sq = warp_reduce_sum(local_sum_sq);
        if (lane_id == 0) smem_reduce[warp_id] = local_sum_sq; __syncthreads();
        if (warp_id == 0) { float sum = (lane_id < NUM_WARPS) ? smem_reduce[lane_id] : 0; sum = warp_reduce_sum(sum); if (lane_id == 0) smem_reduce[0] = rsqrtf(sum/HIDDEN_SIZE + RMS_EPS); }
        __syncthreads(); float rstd = smem_reduce[0];
        for (int i = threadIdx.x; i < HIDDEN_SIZE; i += BLOCK_SIZE) {
            float wt = H2F(__ldg(final_norm_weight + i));
            g_normalized[i] = H2F(F2H(g_activations[i] * rstd * (1.0f + wt)));
        }
    }
}

// =============================================================================
// C entry point
// =============================================================================

// Occupancy-limited cooperative launch: do not silently fall back to a spin barrier.
extern "C" int maximum_blocks() {
    int device=0, sms=0, active=0;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) return 0;
    if (cudaFuncSetAttribute(decode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SHMEM_FLOATS*sizeof(float)) != cudaSuccess) return 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, decode_kernel,
                    BLOCK_SIZE, SHMEM_FLOATS*sizeof(float)) != cudaSuccess) return 0;
    return sms*active;
}
extern "C" const char *error_string(int code) { return cudaGetErrorString((cudaError_t)code); }
extern "C" int decode(void **p, int token, int position, int max_seq, int debug, void *stream_ptr) {
    int resident = maximum_blocks();
    if (resident < DN_NUM_HEADS) return (int)cudaErrorCooperativeLaunchTooLarge;
    int blocks = min(resident, NUM_BLOCKS);
    // First 22 arguments are device pointers, then repetition penalty and scalar indices.
    float penalty = 1.0f;
    void *args[27];
    for (int i=0; i<22; ++i) args[i] = &p[i];
    args[22] = &penalty;
    args[23] = &token; args[24] = &position; args[25] = &max_seq; args[26] = &debug;
    return (int)cudaLaunchCooperativeKernel((void*)decode_kernel,
                dim3(blocks), dim3(BLOCK_SIZE), args,
                SHMEM_FLOATS*sizeof(float), (cudaStream_t)stream_ptr);
}
