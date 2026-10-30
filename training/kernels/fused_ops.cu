// Fused element-wise kernels for the 124M decoder: RMSNorm, QKV split + RoPE,
// SwiGLU activation and softmax cross-entropy. Each has a forward and a backward.
// Activations are BF16, the residual stream and all arithmetic are FP32.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float block_sum(float value, float* shared) {
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffff, value, offset);
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  int warps = blockDim.x >> 5;
  value = threadIdx.x < warps ? shared[threadIdx.x] : 0.f;
  if (warp == 0)
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffff, value, offset);
  if (threadIdx.x == 0) shared[0] = value;
  __syncthreads();
  value = shared[0];
  __syncthreads();
  return value;
}

__device__ __forceinline__ float block_max(float value, float* shared) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  int warps = blockDim.x >> 5;
  value = threadIdx.x < warps ? shared[threadIdx.x] : -INFINITY;
  if (warp == 0)
    for (int offset = 16; offset > 0; offset >>= 1)
      value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  if (threadIdx.x == 0) shared[0] = value;
  __syncthreads();
  value = shared[0];
  __syncthreads();
  return value;
}

// ---- RMSNorm: y = x * rstd * w, rstd = 1 / sqrt(mean(x^2) + eps). One warp per row;
// each lane owns width/128 float4 column groups, so dw accumulates in registers.

// Kernels are instantiated for G = width / 128 column groups, so per-lane arrays stay in
// registers. Width must be a multiple of 128 and at most 1024.
constexpr int kMaxGroups = 8;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffff, value, offset);
  return value;
}

template <int G>
__global__ void rmsnorm_forward_kernel(const float* x, const float* w, bf16* y, float* rstd,
                                       int rows, int width, float eps) {
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= rows) return;
  constexpr int groups = G;
  const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * width);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float4 values[G];
  float sum = 0.f;
  _Pragma("unroll") for (int g = 0; g < groups; ++g) {
    values[g] = xr[g * 32 + lane];
    sum += values[g].x * values[g].x + values[g].y * values[g].y + values[g].z * values[g].z +
           values[g].w * values[g].w;
  }
  float r = rsqrtf(warp_sum(sum) / width + eps);
  if (lane == 0) rstd[row] = r;
  __nv_bfloat162* yr = reinterpret_cast<__nv_bfloat162*>(y + (size_t)row * width);
  _Pragma("unroll") for (int g = 0; g < groups; ++g) {
    float4 v = values[g], s = w4[g * 32 + lane];
    int column = (g * 32 + lane) * 2;
    yr[column] = __floats2bfloat162_rn(v.x * r * s.x, v.y * r * s.y);
    yr[column + 1] = __floats2bfloat162_rn(v.z * r * s.z, v.w * r * s.w);
  }
}

// dx = rstd * (w*dy - xhat * mean(xhat * w * dy)); each warp loops over rows with a stride
// and writes its partial dw, summed afterwards.
template <int G>
__global__ void rmsnorm_backward_kernel(const bf16* dy, const float* x, const float* w,
                                        const float* rstd, float* dx, float* dw_partial,
                                        int rows, int width) {
  int lane = threadIdx.x & 31;
  int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int warps = gridDim.x * (blockDim.x >> 5);
  constexpr int groups = G;
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float4 dw[G];
  _Pragma("unroll") for (int g = 0; g < groups; ++g) dw[g] = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int row = warp; row < rows; row += warps) {
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * width);
    const __nv_bfloat162* dyr = reinterpret_cast<const __nv_bfloat162*>(dy + (size_t)row * width);
    float s = rstd[row], dot = 0.f;
    float4 xs[G], gs[G];
    _Pragma("unroll") for (int g = 0; g < groups; ++g) {
      int column = (g * 32 + lane) * 2;
      float2 a = __bfloat1622float2(dyr[column]), b = __bfloat1622float2(dyr[column + 1]);
      gs[g] = make_float4(a.x, a.y, b.x, b.y);
      xs[g] = xr[g * 32 + lane];
      float4 wv = w4[g * 32 + lane];
      dot += (gs[g].x * wv.x * xs[g].x + gs[g].y * wv.y * xs[g].y + gs[g].z * wv.z * xs[g].z +
              gs[g].w * wv.w * xs[g].w) * s;
    }
    float mean = warp_sum(dot) / width;
    float4* dxr = reinterpret_cast<float4*>(dx + (size_t)row * width);
    _Pragma("unroll") for (int g = 0; g < groups; ++g) {
      float4 wv = w4[g * 32 + lane], v = xs[g], gr = gs[g];
      float4 h = make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
      dxr[g * 32 + lane] = make_float4(s * (gr.x * wv.x - h.x * mean), s * (gr.y * wv.y - h.y * mean),
                                       s * (gr.z * wv.z - h.z * mean), s * (gr.w * wv.w - h.w * mean));
      dw[g].x += gr.x * h.x;
      dw[g].y += gr.y * h.y;
      dw[g].z += gr.z * h.z;
      dw[g].w += gr.w * h.w;
    }
  }
  float4* out = reinterpret_cast<float4*>(dw_partial + (size_t)warp * width);
  _Pragma("unroll") for (int g = 0; g < groups; ++g) out[g * 32 + lane] = dw[g];
}

// ---- Residual add + RMSNorm: s = r + d + b (FP32), y = RMSNorm(s) (BF16). `d` is the BF16
// output of a bias-free linear and `b` its bias. One warp per row.

template <int G>
__global__ void add_rmsnorm_forward_kernel(const float* r, const bf16* d, const float* b, const float* w,
                                           float* s, bf16* y, float* rstd, int rows, int width, float eps) {
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= rows) return;
  constexpr int groups = G;
  const float4* rr = reinterpret_cast<const float4*>(r + (size_t)row * width);
  const __nv_bfloat162* dr = reinterpret_cast<const __nv_bfloat162*>(d + (size_t)row * width);
  const float4* b4 = reinterpret_cast<const float4*>(b);
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float4* sr = reinterpret_cast<float4*>(s + (size_t)row * width);
  float4 values[G];
  float sum = 0.f;
  _Pragma("unroll") for (int g = 0; g < groups; ++g) {
    int i = g * 32 + lane;
    float4 base = rr[i], bias = b4[i];
    float2 lo = __bfloat1622float2(dr[2 * i]), hi = __bfloat1622float2(dr[2 * i + 1]);
    float4 v = make_float4(base.x + lo.x + bias.x, base.y + lo.y + bias.y, base.z + hi.x + bias.z,
                           base.w + hi.y + bias.w);
    values[g] = v;
    sr[i] = v;
    sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  float scale = rsqrtf(warp_sum(sum) / width + eps);
  if (lane == 0) rstd[row] = scale;
  __nv_bfloat162* yr = reinterpret_cast<__nv_bfloat162*>(y + (size_t)row * width);
  _Pragma("unroll") for (int g = 0; g < groups; ++g) {
    int i = g * 32 + lane;
    float4 v = values[g], wv = w4[i];
    yr[2 * i] = __floats2bfloat162_rn(v.x * scale * wv.x, v.y * scale * wv.y);
    yr[2 * i + 1] = __floats2bfloat162_rn(v.z * scale * wv.z, v.w * scale * wv.w);
  }
}

// ds = ds_out + RMSNorm backward; dr = ds (FP32), dd = ds (BF16). Partial sums of dw = dy*xhat
// and db = ds are kept per warp and summed afterwards.
template <int G>
__global__ void add_rmsnorm_backward_kernel(const bf16* dy, const float* ds_out, const float* s,
                                            const float* w, const float* rstd, float* dr, bf16* dd,
                                            float* dw_partial, float* db_partial, int rows, int width) {
  int lane = threadIdx.x & 31;
  int warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  int warps = gridDim.x * (blockDim.x >> 5);
  constexpr int groups = G;
  const float4* w4 = reinterpret_cast<const float4*>(w);
  float4 dw[G], db[G];
  _Pragma("unroll") for (int g = 0; g < groups; ++g) dw[g] = db[g] = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int row = warp; row < rows; row += warps) {
    const float4* sr = reinterpret_cast<const float4*>(s + (size_t)row * width);
    const __nv_bfloat162* dyr = reinterpret_cast<const __nv_bfloat162*>(dy + (size_t)row * width);
    float scale = rstd[row], dot = 0.f;
    float4 xs[G], gs[G];
    _Pragma("unroll") for (int g = 0; g < groups; ++g) {
      int i = g * 32 + lane;
      float2 a = __bfloat1622float2(dyr[2 * i]), c = __bfloat1622float2(dyr[2 * i + 1]);
      gs[g] = make_float4(a.x, a.y, c.x, c.y);
      xs[g] = sr[i];
      float4 wv = w4[i];
      dot += (gs[g].x * wv.x * xs[g].x + gs[g].y * wv.y * xs[g].y + gs[g].z * wv.z * xs[g].z +
              gs[g].w * wv.w * xs[g].w) * scale;
    }
    float mean = warp_sum(dot) / width;
    float4* drr = reinterpret_cast<float4*>(dr + (size_t)row * width);
    __nv_bfloat162* ddr = reinterpret_cast<__nv_bfloat162*>(dd + (size_t)row * width);
    const float4* dor = ds_out ? reinterpret_cast<const float4*>(ds_out + (size_t)row * width) : nullptr;
    _Pragma("unroll") for (int g = 0; g < groups; ++g) {
      int i = g * 32 + lane;
      float4 wv = w4[i], v = xs[g], gr = gs[g];
      float4 h = make_float4(v.x * scale, v.y * scale, v.z * scale, v.w * scale);
      float4 out = dor ? dor[i] : make_float4(0.f, 0.f, 0.f, 0.f);
      out.x += scale * (gr.x * wv.x - h.x * mean);
      out.y += scale * (gr.y * wv.y - h.y * mean);
      out.z += scale * (gr.z * wv.z - h.z * mean);
      out.w += scale * (gr.w * wv.w - h.w * mean);
      drr[i] = out;
      ddr[2 * i] = __floats2bfloat162_rn(out.x, out.y);
      ddr[2 * i + 1] = __floats2bfloat162_rn(out.z, out.w);
      dw[g].x += gr.x * h.x; dw[g].y += gr.y * h.y; dw[g].z += gr.z * h.z; dw[g].w += gr.w * h.w;
      db[g].x += out.x; db[g].y += out.y; db[g].z += out.z; db[g].w += out.w;
    }
  }
  float4* wout = reinterpret_cast<float4*>(dw_partial + (size_t)warp * width);
  float4* bout = reinterpret_cast<float4*>(db_partial + (size_t)warp * width);
  _Pragma("unroll") for (int g = 0; g < groups; ++g) {
    wout[g * 32 + lane] = dw[g];
    bout[g * 32 + lane] = db[g];
  }
}

// ---- QKV split + RoPE on adjacent feature pairs: (B, T, 3C) + bias -> q, k, v (B, H, T, D).
// The backward also sums the bias gradient (pre-rotation dqkv) per block in shared memory.

__device__ __forceinline__ void rope_angle(int position, int pair, int head_dim, float* c, float* s) {
  float frequency = powf(10000.f, -2.f * pair / head_dim);
  sincosf(position * frequency, s, c);
}

// positions: int32 [batch * length], each token's position within its document.
__global__ void qkv_rope_forward_kernel(const bf16* qkv, const float* bias, const int* positions,
                                        bf16* q, bf16* k, bf16* v, int batch, int length, int heads,
                                        int head_dim) {
  int width = heads * head_dim;
  size_t pairs = (size_t)batch * length * width / 2;
  for (size_t index = blockIdx.x * (size_t)blockDim.x + threadIdx.x; index < pairs;
       index += (size_t)gridDim.x * blockDim.x) {
    int feature = (index % (width / 2)) * 2;
    size_t token = index / (width / 2);
    int t = token % length, b = token / length;
    int h = feature / head_dim, d = feature % head_dim;
    const bf16* source = qkv + token * 3 * width;
    size_t target = (((size_t)b * heads + h) * length + t) * head_dim + d;
    float c, s;
    rope_angle(positions[token], d / 2, head_dim, &c, &s);
    for (int part = 0; part < 2; ++part) {
      int column = part * width + feature;
      float even = __bfloat162float(source[column]) + bias[column];
      float odd = __bfloat162float(source[column + 1]) + bias[column + 1];
      bf16* out = part == 0 ? q : k;
      out[target] = __float2bfloat16(even * c - odd * s);
      out[target + 1] = __float2bfloat16(even * s + odd * c);
    }
    int column = 2 * width + feature;
    v[target] = __float2bfloat16(__bfloat162float(source[column]) + bias[column]);
    v[target + 1] = __float2bfloat16(__bfloat162float(source[column + 1]) + bias[column + 1]);
  }
}

__global__ void qkv_rope_backward_kernel(const bf16* dq, const bf16* dk, const bf16* dv,
                                         const int* positions, bf16* dqkv, float* dbias, int batch,
                                         int length, int heads, int head_dim) {
  extern __shared__ float local[];
  int width = heads * head_dim;
  for (int i = threadIdx.x; i < 3 * width; i += blockDim.x) local[i] = 0.f;
  __syncthreads();
  size_t pairs = (size_t)batch * length * width / 2;
  size_t first = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  // The grid stride is a multiple of width / 2, so this thread's feature never changes.
  int feature = (first % (width / 2)) * 2;
  int h = feature / head_dim, d = feature % head_dim;
  float sums[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (size_t index = first; index < pairs; index += (size_t)gridDim.x * blockDim.x) {
    size_t token = index / (width / 2);
    int t = token % length, b = token / length;
    bf16* target = dqkv + token * 3 * width;
    size_t source = (((size_t)b * heads + h) * length + t) * head_dim + d;
    float c, s;
    rope_angle(positions[token], d / 2, head_dim, &c, &s);
    for (int part = 0; part < 3; ++part) {
      const bf16* in = part == 0 ? dq : part == 1 ? dk : dv;
      float ge = __bfloat162float(in[source]), go = __bfloat162float(in[source + 1]);
      float even = ge, odd = go;
      if (part < 2) {
        // Transpose of the rotation.
        even = ge * c + go * s;
        odd = -ge * s + go * c;
      }
      int column = part * width + feature;
      target[column] = __float2bfloat16(even);
      target[column + 1] = __float2bfloat16(odd);
      sums[2 * part] += even;
      sums[2 * part + 1] += odd;
    }
  }
  if (first < pairs)
    for (int part = 0; part < 3; ++part) {
      atomicAdd(&local[part * width + feature], sums[2 * part]);
      atomicAdd(&local[part * width + feature + 1], sums[2 * part + 1]);
    }
  __syncthreads();
  for (int i = threadIdx.x; i < 3 * width; i += blockDim.x) atomicAdd(&dbias[i], local[i]);
}

// ---- SwiGLU activation: h = silu(g + bg) * (u + bu), eight BF16 values per thread
// (hidden width % 8 == 0). The backward also sums both bias gradients per block.

union Pack8 {
  uint4 raw;
  __nv_bfloat162 pairs[4];
};

__global__ void swiglu_forward_kernel(const bf16* g, const bf16* u, const float* bg, const float* bu,
                                      bf16* h, size_t n, int hidden) {
  for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 8; i < n;
       i += (size_t)gridDim.x * blockDim.x * 8) {
    int column = i % hidden;
    Pack8 gate, up, out;
    gate.raw = *reinterpret_cast<const uint4*>(g + i);
    up.raw = *reinterpret_cast<const uint4*>(u + i);
    for (int j = 0; j < 4; ++j) {
      float2 x = __bfloat1622float2(gate.pairs[j]), v = __bfloat1622float2(up.pairs[j]);
      int c = column + 2 * j;
      x.x += bg[c]; x.y += bg[c + 1]; v.x += bu[c]; v.y += bu[c + 1];
      out.pairs[j] = __floats2bfloat162_rn(x.x / (1.f + __expf(-x.x)) * v.x, x.y / (1.f + __expf(-x.y)) * v.y);
    }
    *reinterpret_cast<uint4*>(h + i) = out.raw;
  }
}

__global__ void swiglu_backward_kernel(const bf16* dh, const bf16* g, const bf16* u, const float* bg,
                                       const float* bu, bf16* dg, bf16* du, float* dbg, float* dbu,
                                       size_t n, int hidden) {
  extern __shared__ float local[];
  float* local_g = local;
  float* local_u = local + hidden;
  for (int i = threadIdx.x; i < 2 * hidden; i += blockDim.x) local[i] = 0.f;
  __syncthreads();
  size_t first = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 8;
  // The grid stride is a multiple of the row width, so this thread's columns never change.
  int column = first % hidden;
  float bias_g[8], bias_u[8], sum_g[8] = {0.f}, sum_u[8] = {0.f};
  for (int j = 0; j < 8; ++j) {
    bias_g[j] = bg[column + j];
    bias_u[j] = bu[column + j];
  }
  for (size_t i = first; i < n; i += (size_t)gridDim.x * blockDim.x * 8) {
    Pack8 grad, gate, up, out_g, out_u;
    grad.raw = *reinterpret_cast<const uint4*>(dh + i);
    gate.raw = *reinterpret_cast<const uint4*>(g + i);
    up.raw = *reinterpret_cast<const uint4*>(u + i);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 d = __bfloat1622float2(grad.pairs[j]), x = __bfloat1622float2(gate.pairs[j]),
             v = __bfloat1622float2(up.pairs[j]);
      x.x += bias_g[2 * j]; x.y += bias_g[2 * j + 1]; v.x += bias_u[2 * j]; v.y += bias_u[2 * j + 1];
      float sx = 1.f / (1.f + __expf(-x.x)), sy = 1.f / (1.f + __expf(-x.y));
      float gx = d.x * v.x * sx * (1.f + x.x * (1.f - sx)), gy = d.y * v.y * sy * (1.f + x.y * (1.f - sy));
      float ux = d.x * x.x * sx, uy = d.y * x.y * sy;
      out_g.pairs[j] = __floats2bfloat162_rn(gx, gy);
      out_u.pairs[j] = __floats2bfloat162_rn(ux, uy);
      sum_g[2 * j] += gx; sum_g[2 * j + 1] += gy;
      sum_u[2 * j] += ux; sum_u[2 * j + 1] += uy;
    }
    *reinterpret_cast<uint4*>(dg + i) = out_g.raw;
    *reinterpret_cast<uint4*>(du + i) = out_u.raw;
  }
  if (first < n)
    for (int j = 0; j < 8; ++j) {
      atomicAdd(&local_g[column + j], sum_g[j]);
      atomicAdd(&local_u[column + j], sum_u[j]);
    }
  __syncthreads();
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    atomicAdd(&dbg[i], local_g[i]);
    atomicAdd(&dbu[i], local_u[i]);
  }
}

// ---- Softmax cross-entropy, one block per row: per-row loss and d(loss)/d(logits).
// One online pass finds max and sum of exponentials; a second writes the gradient. Each
// thread reads a pack before overwriting it, so the gradient may replace the logits.
__global__ void cross_entropy_kernel(const bf16* logits, const int64_t* targets, float* loss,
                                     bf16* grad, int vocab) {
  __shared__ float shared[32];
  const uint4* row = reinterpret_cast<const uint4*>(logits + (size_t)blockIdx.x * vocab);
  uint4* grad_row = reinterpret_cast<uint4*>(grad + (size_t)blockIdx.x * vocab);
  int64_t target = targets[blockIdx.x];
  int packs = vocab / 8;
  if (target < 0) {
    for (int i = threadIdx.x; i < packs; i += blockDim.x) grad_row[i] = make_uint4(0, 0, 0, 0);
    if (threadIdx.x == 0) loss[blockIdx.x] = 0.f;
    return;
  }
  float m = -INFINITY, total = 0.f;
  for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    Pack8 p;
    p.raw = row[i];
    for (int j = 0; j < 4; ++j) {
      float2 v = __bfloat1622float2(p.pairs[j]);
      float hi = fmaxf(v.x, v.y);
      if (hi > m) {
        total *= __expf(m - hi);
        m = hi;
      }
      total += __expf(v.x - m) + __expf(v.y - m);
    }
  }
  float maximum = block_max(m, shared);
  float log_total = __logf(block_sum(total * __expf(m - maximum), shared));
  if (threadIdx.x == 0)
    loss[blockIdx.x] = maximum + log_total - __bfloat162float(logits[(size_t)blockIdx.x * vocab + target]);
  __syncthreads();
  for (int i = threadIdx.x; i < packs; i += blockDim.x) {
    Pack8 p, out;
    p.raw = row[i];
    for (int j = 0; j < 4; ++j) {
      float2 v = __bfloat1622float2(p.pairs[j]);
      int column = i * 8 + 2 * j;
      float a = __expf(v.x - maximum - log_total) - (column == target ? 1.f : 0.f);
      float b = __expf(v.y - maximum - log_total) - (column + 1 == target ? 1.f : 0.f);
      out.pairs[j] = __floats2bfloat162_rn(a, b);
    }
    grad_row[i] = out.raw;
  }
}

int blocks_for(size_t n, int threads) { return (int)((n + threads - 1) / threads); }

#define DISPATCH_GROUPS(width, ...)                                              \
  switch ((width) / 128) {                                                       \
    case 1: { constexpr int G = 1; __VA_ARGS__; break; }                         \
    case 2: { constexpr int G = 2; __VA_ARGS__; break; }                         \
    case 3: { constexpr int G = 3; __VA_ARGS__; break; }                         \
    case 4: { constexpr int G = 4; __VA_ARGS__; break; }                         \
    case 5: { constexpr int G = 5; __VA_ARGS__; break; }                         \
    case 6: { constexpr int G = 6; __VA_ARGS__; break; }                         \
    case 7: { constexpr int G = 7; __VA_ARGS__; break; }                         \
    case 8: { constexpr int G = 8; __VA_ARGS__; break; }                         \
    default: TORCH_CHECK(false, "width must be a multiple of 128, at most 1024"); \
  }

// Grid-stride kernels that sum bias gradients keep each thread on fixed columns: the grid's
// thread count is a multiple of the number of work items per row.
int column_stable_blocks(int items_per_row, int threads) {
  int blocks = 264;
  while ((size_t)blocks * threads % items_per_row) ++blocks;
  return blocks;
}

}  // namespace

std::vector<torch::Tensor> rmsnorm_forward(torch::Tensor x, torch::Tensor w, double eps) {
  int width = x.size(-1);
  int rows = x.numel() / width;
  TORCH_CHECK(width % 128 == 0 && width <= 32 * 4 * kMaxGroups, "RMSNorm width must be a multiple of 128, at most 1024");
  auto y = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
  auto rstd = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  DISPATCH_GROUPS(width, rmsnorm_forward_kernel<G><<<blocks_for(rows, 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), w.data_ptr<float>(), reinterpret_cast<bf16*>(y.data_ptr()),
      rstd.data_ptr<float>(), rows, width, (float)eps))
  return {y, rstd};
}

std::vector<torch::Tensor> rmsnorm_backward(torch::Tensor dy, torch::Tensor x, torch::Tensor w,
                                            torch::Tensor rstd) {
  int width = x.size(-1);
  int rows = x.numel() / width;
  int blocks = std::min(blocks_for(rows, 8), 264);  // 8 warps per block, ~2 blocks per SM
  auto dx = torch::empty_like(x);
  auto partial = torch::empty({blocks * 8, width}, x.options().dtype(torch::kFloat32));
  DISPATCH_GROUPS(width, rmsnorm_backward_kernel<G><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(dy.data_ptr()), x.data_ptr<float>(), w.data_ptr<float>(),
      rstd.data_ptr<float>(), dx.data_ptr<float>(), partial.data_ptr<float>(), rows, width))
  return {dx, partial.sum(0)};
}

std::vector<torch::Tensor> add_rmsnorm_forward(torch::Tensor r, torch::Tensor d, torch::Tensor b,
                                               torch::Tensor w, double eps) {
  int width = r.size(-1);
  int rows = r.numel() / width;
  TORCH_CHECK(width % 128 == 0 && width <= 32 * 4 * kMaxGroups, "width must be a multiple of 128, at most 1024");
  auto s = torch::empty_like(r);
  auto y = torch::empty(r.sizes(), r.options().dtype(torch::kBFloat16));
  auto rstd = torch::empty({rows}, r.options().dtype(torch::kFloat32));
  DISPATCH_GROUPS(width, add_rmsnorm_forward_kernel<G><<<blocks_for(rows, 8), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      r.data_ptr<float>(), reinterpret_cast<const bf16*>(d.data_ptr()), b.data_ptr<float>(),
      w.data_ptr<float>(), s.data_ptr<float>(), reinterpret_cast<bf16*>(y.data_ptr()),
      rstd.data_ptr<float>(), rows, width, (float)eps))
  return {s, y, rstd};
}

std::vector<torch::Tensor> add_rmsnorm_backward(torch::Tensor dy, c10::optional<torch::Tensor> ds_out,
                                                torch::Tensor s, torch::Tensor w, torch::Tensor rstd) {
  int width = s.size(-1);
  int rows = s.numel() / width;
  int blocks = std::min(blocks_for(rows, 8), 264);
  auto dr = torch::empty_like(s);
  auto dd = torch::empty(s.sizes(), s.options().dtype(torch::kBFloat16));
  auto dw = torch::empty({blocks * 8, width}, s.options());
  auto db = torch::empty({blocks * 8, width}, s.options());
  const float* ds_ptr = ds_out.has_value() ? ds_out->data_ptr<float>() : nullptr;
  DISPATCH_GROUPS(width, add_rmsnorm_backward_kernel<G><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(dy.data_ptr()), ds_ptr, s.data_ptr<float>(), w.data_ptr<float>(),
      rstd.data_ptr<float>(), dr.data_ptr<float>(), reinterpret_cast<bf16*>(dd.data_ptr()),
      dw.data_ptr<float>(), db.data_ptr<float>(), rows, width))
  return {dr, dd, dw.sum(0), db.sum(0)};
}

std::vector<torch::Tensor> qkv_rope_forward(torch::Tensor qkv, torch::Tensor bias,
                                            torch::Tensor positions, int64_t heads) {
  positions = positions.to(torch::kInt32).contiguous();
  int batch = qkv.size(0), length = qkv.size(1), width = qkv.size(2) / 3;
  int head_dim = width / heads;
  auto q = torch::empty({batch, heads, length, head_dim}, qkv.options());
  auto k = torch::empty_like(q), v = torch::empty_like(q);
  size_t pairs = (size_t)batch * length * width / 2;
  qkv_rope_forward_kernel<<<std::min(blocks_for(pairs, 256), 4096), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(qkv.data_ptr()), bias.data_ptr<float>(),
      positions.data_ptr<int>(), reinterpret_cast<bf16*>(q.data_ptr()), reinterpret_cast<bf16*>(k.data_ptr()),
      reinterpret_cast<bf16*>(v.data_ptr()), batch, length, heads, head_dim);
  return {q, k, v};
}

std::vector<torch::Tensor> qkv_rope_backward(torch::Tensor dq, torch::Tensor dk, torch::Tensor dv,
                                             torch::Tensor positions) {
  positions = positions.to(torch::kInt32).contiguous();
  dq = dq.contiguous();
  dk = dk.contiguous();
  dv = dv.contiguous();
  int batch = dq.size(0), heads = dq.size(1), length = dq.size(2), head_dim = dq.size(3);
  int width = heads * head_dim;
  auto dqkv = torch::empty({batch, length, 3 * width}, dq.options());
  auto dbias = torch::zeros({3 * width}, dq.options().dtype(torch::kFloat32));
  size_t pairs = (size_t)batch * length * width / 2;
  qkv_rope_backward_kernel<<<column_stable_blocks(width / 2, 256), 256, 3 * width * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(dq.data_ptr()), reinterpret_cast<const bf16*>(dk.data_ptr()),
      reinterpret_cast<const bf16*>(dv.data_ptr()), positions.data_ptr<int>(),
      reinterpret_cast<bf16*>(dqkv.data_ptr()),
      dbias.data_ptr<float>(), batch, length, heads, head_dim);
  return {dqkv, dbias};
}

torch::Tensor swiglu_forward(torch::Tensor g, torch::Tensor u, torch::Tensor bg, torch::Tensor bu) {
  auto h = torch::empty_like(g);
  size_t n = g.numel();
  int hidden = g.size(-1);
  TORCH_CHECK(hidden % 8 == 0, "SwiGLU width must be a multiple of 8");
  swiglu_forward_kernel<<<std::min(blocks_for(n / 8, 256), 4096), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(g.data_ptr()), reinterpret_cast<const bf16*>(u.data_ptr()),
      bg.data_ptr<float>(), bu.data_ptr<float>(), reinterpret_cast<bf16*>(h.data_ptr()), n, hidden);
  return h;
}

std::vector<torch::Tensor> swiglu_backward(torch::Tensor dh, torch::Tensor g, torch::Tensor u,
                                           torch::Tensor bg, torch::Tensor bu) {
  dh = dh.contiguous();
  auto dg = torch::empty_like(g), du = torch::empty_like(u);
  int hidden = g.size(-1);
  auto dbg = torch::zeros({hidden}, g.options().dtype(torch::kFloat32));
  auto dbu = torch::zeros_like(dbg);
  size_t n = g.numel();
  swiglu_backward_kernel<<<column_stable_blocks(hidden / 8, 256), 256, 2 * hidden * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(dh.data_ptr()), reinterpret_cast<const bf16*>(g.data_ptr()),
      reinterpret_cast<const bf16*>(u.data_ptr()), bg.data_ptr<float>(), bu.data_ptr<float>(),
      reinterpret_cast<bf16*>(dg.data_ptr()), reinterpret_cast<bf16*>(du.data_ptr()),
      dbg.data_ptr<float>(), dbu.data_ptr<float>(), n, hidden);
  return {dg, du, dbg, dbu};
}

// Returns per-row losses and overwrites `logits` with d(sum of losses)/d(logits).
torch::Tensor cross_entropy_(torch::Tensor logits, torch::Tensor targets) {
  int rows = logits.size(0), vocab = logits.size(1);
  TORCH_CHECK(vocab % 8 == 0 && logits.is_contiguous(), "Contiguous logits, vocabulary a multiple of 8");
  auto loss = torch::empty({rows}, logits.options().dtype(torch::kFloat32));
  auto data = reinterpret_cast<bf16*>(logits.data_ptr());
  cross_entropy_kernel<<<rows, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      data, targets.data_ptr<int64_t>(), loss.data_ptr<float>(), data, vocab);
  return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward);
  m.def("rmsnorm_backward", &rmsnorm_backward);
  m.def("add_rmsnorm_forward", &add_rmsnorm_forward);
  m.def("add_rmsnorm_backward", &add_rmsnorm_backward);
  m.def("qkv_rope_forward", &qkv_rope_forward);
  m.def("qkv_rope_backward", &qkv_rope_backward);
  m.def("swiglu_forward", &swiglu_forward);
  m.def("swiglu_backward", &swiglu_backward);
  m.def("cross_entropy_", &cross_entropy_);
}
