// Document-causal GQA attention, bit-exact with hybrid/exact_attention.py.
//
// Every float operation is a correctly rounded __fmul_rn / __fadd_rn / __fdiv_rn
// (and the file is built with -fmad=false), applied in the order the reference
// defines: dot products sum over the head dimension in order, softmax sums and
// value/gradient accumulations sum over keys (or query heads, then query rows) in
// order. Blocks own disjoint outputs, so there are no atomics and results are
// deterministic. Allowed keys of row i are [start_i, i]; tiles outside are skipped
// and masked entries are exact zeros.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

constexpr int TILE = 64;           // Query rows and keys per tile.
constexpr int THREADS = 256;       // 16 x 16 threads, each a 4 x 4 score micro-tile.
constexpr int PAD = TILE + 4;      // Row stride of transposed [D][TILE] tiles.
constexpr int SROW = TILE + 1;     // Row stride of [TILE][TILE] score tiles.

__device__ __forceinline__ float fmul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float fsub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float fdiv(float a, float b) { return __fdiv_rn(a, b); }

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float exp_fixed(float x) {
  const float y = fmul(x, 0x1.715476p+0f);
  if (!(y >= -125.0f)) return 0.0f;
  const float n = rintf(y);
  const float f = fsub(y, n);
  float p = 0x1.ffcbfcp-17f;
  p = fadd(fmul(p, f), 0x1.430912p-13f);
  p = fadd(fmul(p, f), 0x1.5d87fep-10f);
  p = fadd(fmul(p, f), 0x1.3b2ab6p-7f);
  p = fadd(fmul(p, f), 0x1.c6b08ep-5f);
  p = fadd(fmul(p, f), 0x1.ebfbe0p-3f);
  p = fadd(fmul(p, f), 0x1.62e430p-1f);
  p = fadd(fmul(p, f), 1.0f);
  return fmul(p, __int_as_float((static_cast<int>(n) + 127) << 23));
}

__device__ __forceinline__ float log_fixed(float x) {
  const int bits = __float_as_int(x);
  int exponent = ((bits >> 23) & 0xFF) - 127;
  float m = __int_as_float((bits & 0x7FFFFF) | 0x3F800000);
  if (m > 0x1.6a09e6p+0f) {
    m = fmul(m, 0.5f);
    exponent += 1;
  }
  const float s = fdiv(fsub(m, 1.0f), fadd(m, 1.0f));
  const float s2 = fmul(s, s);
  float t = 0x1.c71c72p-4f;
  t = fadd(fmul(t, s2), 0x1.24924ap-3f);
  t = fadd(fmul(t, s2), 0x1.99999ap-3f);
  t = fadd(fmul(t, s2), 0x1.555556p-2f);
  t = fadd(fmul(t, s2), 1.0f);
  return fadd(fmul(static_cast<float>(exponent), 0x1.62e430p-1f), fmul(fmul(s, t), 2.0f));
}

// Load rows [row0, row0 + TILE) of a [T, D] matrix as fp32, transposed to [D][PAD].
template <typename T, int D>
__device__ void load_transposed(float* dst, const T* src, int row0, int length) {
  for (int index = threadIdx.x; index < TILE * D; index += THREADS) {
    const int r = index % TILE, d = index / TILE, row = row0 + r;
    dst[d * PAD + r] = row < length ? to_float(src[static_cast<long>(row) * D + d]) : 0.0f;
  }
}

// Load rows [row0, row0 + TILE) of a [T, D] matrix as fp32 in [TILE][D] layout.
template <typename T, int D>
__device__ void load_rows(float* dst, const T* src, int row0, int length) {
  for (int index = threadIdx.x; index < TILE * D; index += THREADS) {
    const int r = index / D, d = index % D, row = row0 + r;
    dst[index] = row < length ? to_float(src[static_cast<long>(row) * D + d]) : 0.0f;
  }
}

// out[a][c] = seqsum_d(left[d][rows + a] * right[d][cols + c]) for a 4 x 4 micro-tile.
template <int D>
__device__ __forceinline__ void dot_tile(float out[4][4], const float* left, const float* right,
                                         int rows, int cols) {
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int c = 0; c < 4; ++c) out[a][c] = 0.0f;
#pragma unroll 4
  for (int d = 0; d < D; ++d) {
    const float4 x = *reinterpret_cast<const float4*>(left + d * PAD + rows);
    const float4 y = *reinterpret_cast<const float4*>(right + d * PAD + cols);
    const float xs[4] = {x.x, x.y, x.z, x.w}, ys[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int c = 0; c < 4; ++c) out[a][c] = fadd(out[a][c], fmul(xs[a], ys[c]));
  }
}

// acc[a][e] = acc + seqsum_j(weights[rows + a][j] * values[j][dims + e]) over one tile.
template <int D>
__device__ __forceinline__ void accumulate_tile(float acc[4][D / 16], const float* weights,
                                                const float* values, int rows, int dims) {
  for (int j = 0; j < TILE; ++j) {
    float w[4];
#pragma unroll
    for (int a = 0; a < 4; ++a) w[a] = weights[(rows + a) * SROW + j];
#pragma unroll
    for (int e = 0; e < D / 16; ++e) {
      const float value = values[j * D + dims + e];
#pragma unroll
      for (int a = 0; a < 4; ++a) acc[a][e] = fadd(acc[a][e], fmul(w[a], value));
    }
  }
}

__device__ __forceinline__ bool allowed(int row, int key, int start, int length) {
  return row < length && key <= row && key >= start;
}

template <typename T, int D>
__global__ void __launch_bounds__(THREADS)
    forward_kernel(const T* q, const T* k, const T* v, const int* starts, T* out, float* lse,
                   float* row_max, float* row_sum, int heads, int kv_heads, int length,
                   float scale) {
  extern __shared__ __align__(16) float shared[];
  float* qT = shared;
  float* kT = qT + D * PAD;
  float* vS = kT + D * PAD;
  float* sP = vS + TILE * D;
  float* rowM = sP + TILE * SROW;
  float* rowL = rowM + TILE;
  int* rowStart = reinterpret_cast<int*>(rowL + TILE);

  const int tile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hk = h / (heads / kv_heads);
  const int q0 = tile * TILE;
  const int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
  const T* qh = q + (static_cast<long>(b) * heads + h) * length * D;
  const T* kh = k + (static_cast<long>(b) * kv_heads + hk) * length * D;
  const T* vh = v + (static_cast<long>(b) * kv_heads + hk) * length * D;
  const int* startb = starts + static_cast<long>(b) * length;

  load_transposed<T, D>(qT, qh, q0, length);
  if (threadIdx.x < TILE) {
    const int row = q0 + threadIdx.x;
    rowStart[threadIdx.x] = row < length ? startb[row] : length;
    rowM[threadIdx.x] = -INFINITY;
    rowL[threadIdx.x] = 0.0f;
  }
  __syncthreads();
  const int first = startb[q0] / TILE, last = min(q0 + TILE - 1, length - 1) / TILE;

  // Pass 1: row maxima over allowed keys (order-independent).
  for (int kt = first; kt <= last; ++kt) {
    load_transposed<T, D>(kT, kh, kt * TILE, length);
    __syncthreads();
    float s[4][4];
    dot_tile<D>(s, qT, kT, ty * 4, tx * 4);
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int r = ty * 4 + a, key = kt * TILE + tx * 4 + c;
        sP[r * SROW + tx * 4 + c] =
            allowed(q0 + r, key, rowStart[r], length) ? fmul(s[a][c], scale) : -INFINITY;
      }
    __syncthreads();
    if (threadIdx.x < TILE) {
      float m = rowM[threadIdx.x];
      for (int j = 0; j < TILE; ++j) m = fmaxf(m, sP[threadIdx.x * SROW + j]);
      rowM[threadIdx.x] = m;
    }
    __syncthreads();
  }

  // Pass 2: e = exp(s - m), l = seqsum e, acc = seqsum e * v.
  float acc[4][D / 16] = {};
  for (int kt = first; kt <= last; ++kt) {
    load_transposed<T, D>(kT, kh, kt * TILE, length);
    load_rows<T, D>(vS, vh, kt * TILE, length);
    __syncthreads();
    float s[4][4];
    dot_tile<D>(s, qT, kT, ty * 4, tx * 4);
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int r = ty * 4 + a, key = kt * TILE + tx * 4 + c;
        sP[r * SROW + tx * 4 + c] = allowed(q0 + r, key, rowStart[r], length)
                                        ? exp_fixed(fsub(fmul(s[a][c], scale), rowM[r]))
                                        : 0.0f;
      }
    __syncthreads();
    if (threadIdx.x < TILE) {
      float l = rowL[threadIdx.x];
      for (int j = 0; j < TILE; ++j) l = fadd(l, sP[threadIdx.x * SROW + j]);
      rowL[threadIdx.x] = l;
    }
    accumulate_tile<D>(acc, sP, vS, ty * 4, tx * (D / 16));
    __syncthreads();
  }

  T* oh = out + (static_cast<long>(b) * heads + h) * length * D;
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int r = ty * 4 + a, row = q0 + r;
    if (row >= length) continue;
#pragma unroll
    for (int e = 0; e < D / 16; ++e)
      oh[static_cast<long>(row) * D + tx * (D / 16) + e] = from_float<T>(fdiv(acc[a][e], rowL[r]));
  }
  if (threadIdx.x < TILE && q0 + static_cast<int>(threadIdx.x) < length) {
    const long index = (static_cast<long>(b) * heads + h) * length + q0 + threadIdx.x;
    row_max[index] = rowM[threadIdx.x];
    row_sum[index] = rowL[threadIdx.x];
    lse[index] = fadd(rowM[threadIdx.x], log_fixed(rowL[threadIdx.x]));
  }
}

// D_i = seqsum_d(dout_id * out_id), one thread per row.
template <typename T, int D>
__global__ void delta_kernel(const T* out, const T* dout, float* delta, long rows) {
  const long row = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float total = 0.0f;
  for (int d = 0; d < D; ++d)
    total = fadd(total, fmul(to_float(dout[row * D + d]), to_float(out[row * D + d])));
  delta[row] = total;
}

// p and ds for one micro-tile; row/key orientation chosen by the caller.
__device__ __forceinline__ void probabilities(float s, float dp, float m, float l, float delta,
                                              float dlse, float scale, bool ok, float* p,
                                              float* ds) {
  if (!ok) {
    *p = 0.0f;
    *ds = 0.0f;
    return;
  }
  *p = fdiv(exp_fixed(fsub(fmul(s, scale), m)), l);
  *ds = fmul(*p, fadd(fsub(dp, delta), dlse));
}

template <typename T, int D>
__global__ void __launch_bounds__(THREADS)
    dq_kernel(const T* q, const T* k, const T* v, const int* starts, const T* dout,
              const float* row_max, const float* row_sum, const float* delta, const float* dlse,
              T* dq, int heads, int kv_heads, int length, float scale) {
  extern __shared__ __align__(16) float shared[];
  float* qT = shared;
  float* doT = qT + D * PAD;
  float* kT = doT + D * PAD;
  float* vT = kT + D * PAD;
  float* kS = vT + D * PAD;
  float* dS = kS + TILE * D;
  float* rowM = dS + TILE * SROW;
  float* rowL = rowM + TILE;
  float* rowD = rowL + TILE;
  float* rowG = rowD + TILE;
  int* rowStart = reinterpret_cast<int*>(rowG + TILE);

  const int tile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hk = h / (heads / kv_heads);
  const int q0 = tile * TILE;
  const int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
  const long head_offset = (static_cast<long>(b) * heads + h) * length;
  const T* kh = k + (static_cast<long>(b) * kv_heads + hk) * length * D;
  const T* vh = v + (static_cast<long>(b) * kv_heads + hk) * length * D;
  const int* startb = starts + static_cast<long>(b) * length;

  load_transposed<T, D>(qT, q + head_offset * D, q0, length);
  load_transposed<T, D>(doT, dout + head_offset * D, q0, length);
  if (threadIdx.x < TILE) {
    const int row = q0 + threadIdx.x;
    const bool valid = row < length;
    rowStart[threadIdx.x] = valid ? startb[row] : length;
    rowM[threadIdx.x] = valid ? row_max[head_offset + row] : 0.0f;
    rowL[threadIdx.x] = valid ? row_sum[head_offset + row] : 1.0f;
    rowD[threadIdx.x] = valid ? delta[head_offset + row] : 0.0f;
    rowG[threadIdx.x] = valid ? dlse[head_offset + row] : 0.0f;
  }
  __syncthreads();
  const int first = startb[q0] / TILE, last = min(q0 + TILE - 1, length - 1) / TILE;

  float acc[4][D / 16] = {};
  for (int kt = first; kt <= last; ++kt) {
    load_transposed<T, D>(kT, kh, kt * TILE, length);
    load_transposed<T, D>(vT, vh, kt * TILE, length);
    load_rows<T, D>(kS, kh, kt * TILE, length);
    __syncthreads();
    float s[4][4], dp[4][4];
    dot_tile<D>(s, qT, kT, ty * 4, tx * 4);
    dot_tile<D>(dp, doT, vT, ty * 4, tx * 4);
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int r = ty * 4 + a, key = kt * TILE + tx * 4 + c;
        float p, ds;
        probabilities(s[a][c], dp[a][c], rowM[r], rowL[r], rowD[r], rowG[r], scale,
                      allowed(q0 + r, key, rowStart[r], length), &p, &ds);
        dS[r * SROW + tx * 4 + c] = ds;
      }
    __syncthreads();
    accumulate_tile<D>(acc, dS, kS, ty * 4, tx * (D / 16));
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int row = q0 + ty * 4 + a;
    if (row >= length) continue;
#pragma unroll
    for (int e = 0; e < D / 16; ++e)
      dq[(head_offset + row) * D + tx * (D / 16) + e] = from_float<T>(fmul(acc[a][e], scale));
  }
}

template <typename T, int D>
__global__ void __launch_bounds__(THREADS)
    dkv_kernel(const T* q, const T* k, const T* v, const int* starts, const T* dout,
               const float* row_max, const float* row_sum, const float* delta, const float* dlse,
               T* dk, T* dv, int heads, int kv_heads, int length, float scale) {
  extern __shared__ __align__(16) float shared[];
  float* kT = shared;
  float* vT = kT + D * PAD;
  float* qT = vT + D * PAD;
  float* doT = qT + D * PAD;
  float* qS = doT + D * PAD;
  float* doS = qS + TILE * D;
  float* wT = doS + TILE * D;  // [key][query] weights: p, then ds.
  float* rowM = wT + TILE * SROW;
  float* rowL = rowM + TILE;
  float* rowD = rowL + TILE;
  float* rowG = rowD + TILE;
  int* rowStart = reinterpret_cast<int*>(rowG + TILE);

  const int tile = blockIdx.x, hk = blockIdx.y, b = blockIdx.z;
  const int groups = heads / kv_heads;
  const int k0 = tile * TILE, k_last = min(k0 + TILE - 1, length - 1);
  const int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
  const long kv_offset = (static_cast<long>(b) * kv_heads + hk) * length;
  const int* startb = starts + static_cast<long>(b) * length;

  load_transposed<T, D>(kT, k + kv_offset * D, k0, length);
  load_transposed<T, D>(vT, v + kv_offset * D, k0, length);
  float dk_acc[4][D / 16] = {}, dv_acc[4][D / 16] = {};

  for (int g = 0; g < groups; ++g) {
    const int h = hk * groups + g;
    const long head_offset = (static_cast<long>(b) * heads + h) * length;
    for (int qt = k0 / TILE; qt * TILE < length && startb[qt * TILE] <= k_last; ++qt) {
      const int q0 = qt * TILE;
      __syncthreads();
      load_transposed<T, D>(qT, q + head_offset * D, q0, length);
      load_transposed<T, D>(doT, dout + head_offset * D, q0, length);
      load_rows<T, D>(qS, q + head_offset * D, q0, length);
      load_rows<T, D>(doS, dout + head_offset * D, q0, length);
      if (threadIdx.x < TILE) {
        const int row = q0 + threadIdx.x;
        const bool valid = row < length;
        rowStart[threadIdx.x] = valid ? startb[row] : length;
        rowM[threadIdx.x] = valid ? row_max[head_offset + row] : 0.0f;
        rowL[threadIdx.x] = valid ? row_sum[head_offset + row] : 1.0f;
        rowD[threadIdx.x] = valid ? delta[head_offset + row] : 0.0f;
        rowG[threadIdx.x] = valid ? dlse[head_offset + row] : 0.0f;
      }
      __syncthreads();
      // Transposed micro-tile: keys ty * 4 + a, query rows tx * 4 + c.
      float s[4][4], dp[4][4], p[4][4], ds[4][4];
      dot_tile<D>(s, kT, qT, ty * 4, tx * 4);
      dot_tile<D>(dp, vT, doT, ty * 4, tx * 4);
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const int r = tx * 4 + c, key = k0 + ty * 4 + a;
          probabilities(s[a][c], dp[a][c], rowM[r], rowL[r], rowD[r], rowG[r], scale,
                        allowed(q0 + r, key, rowStart[r], length), &p[a][c], &ds[a][c]);
          wT[(ty * 4 + a) * SROW + r] = p[a][c];
        }
      __syncthreads();
      accumulate_tile<D>(dv_acc, wT, doS, ty * 4, tx * (D / 16));
      __syncthreads();
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int c = 0; c < 4; ++c) wT[(ty * 4 + a) * SROW + tx * 4 + c] = ds[a][c];
      __syncthreads();
      accumulate_tile<D>(dk_acc, wT, qS, ty * 4, tx * (D / 16));
    }
  }
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int key = k0 + ty * 4 + a;
    if (key >= length) continue;
#pragma unroll
    for (int e = 0; e < D / 16; ++e) {
      const long index = (kv_offset + key) * D + tx * (D / 16) + e;
      dk[index] = from_float<T>(fmul(dk_acc[a][e], scale));
      dv[index] = from_float<T>(dv_acc[a][e]);
    }
  }
}

constexpr size_t forward_shared(int d) {
  return (2 * d * PAD + TILE * d + TILE * SROW + 3 * TILE) * sizeof(float);
}
constexpr size_t dq_shared(int d) {
  return (4 * d * PAD + TILE * d + TILE * SROW + 5 * TILE) * sizeof(float);
}
constexpr size_t dkv_shared(int d) {
  return (4 * d * PAD + 2 * TILE * d + TILE * SROW + 5 * TILE) * sizeof(float);
}

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                  const torch::Tensor& starts) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && q.is_contiguous(), "q must be contiguous CUDA [B,H,T,D]");
  TORCH_CHECK(k.sizes() == v.sizes() && k.is_contiguous() && v.is_contiguous(), "bad k/v");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "dtype");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) && q.size(3) == k.size(3), "shape");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "query heads must be a multiple of key/value heads");
  TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128, "head dimension must be 64 or 128");
  TORCH_CHECK(starts.scalar_type() == torch::kInt32 && starts.is_contiguous(), "starts int32");
  TORCH_CHECK(starts.size(0) == q.size(0) && starts.size(1) == q.size(2), "starts shape");
}

#define DISPATCH(TYPE, DIM, ...)                                                   \
  [&] {                                                                            \
    if (TYPE == torch::kBFloat16) {                                                \
      using scalar_t = __nv_bfloat16;                                              \
      if (DIM == 128) { constexpr int HEAD = 128; __VA_ARGS__(); }                 \
      else { constexpr int HEAD = 64; __VA_ARGS__(); }                             \
    } else {                                                                       \
      TORCH_CHECK(TYPE == torch::kFloat32, "dtype must be bfloat16 or float32");   \
      using scalar_t = float;                                                      \
      if (DIM == 128) { constexpr int HEAD = 128; __VA_ARGS__(); }                 \
      else { constexpr int HEAD = 64; __VA_ARGS__(); }                             \
    }                                                                              \
  }()

template <typename T>
const T* ptr(const torch::Tensor& t) { return reinterpret_cast<const T*>(t.data_ptr()); }
template <typename T>
T* mut(torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

}  // namespace

std::vector<torch::Tensor> forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                   torch::Tensor starts, double scale) {
  check_inputs(q, k, v, starts);
  const c10::cuda::CUDAGuard guard(q.device());
  const int batch = q.size(0), heads = q.size(1), length = q.size(2), dim = q.size(3);
  auto out = torch::empty_like(q);
  auto stats = q.options().dtype(torch::kFloat32);
  auto lse = torch::empty({batch, heads, length}, stats);
  auto row_max = torch::empty_like(lse), row_sum = torch::empty_like(lse);
  const dim3 grid((length + TILE - 1) / TILE, heads, batch);
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH(q.scalar_type(), dim, [&] {
    auto kernel = forward_kernel<scalar_t, HEAD>;
    constexpr size_t bytes = forward_shared(HEAD);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    kernel<<<grid, THREADS, bytes, stream>>>(
        ptr<scalar_t>(q), ptr<scalar_t>(k), ptr<scalar_t>(v), starts.data_ptr<int>(),
        mut<scalar_t>(out), lse.data_ptr<float>(), row_max.data_ptr<float>(),
        row_sum.data_ptr<float>(), heads, k.size(1), length, static_cast<float>(scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, row_max, row_sum};
}

std::vector<torch::Tensor> backward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                    torch::Tensor starts, torch::Tensor out, torch::Tensor dout,
                                    torch::Tensor dlse, torch::Tensor row_max,
                                    torch::Tensor row_sum, double scale) {
  check_inputs(q, k, v, starts);
  TORCH_CHECK(out.sizes() == q.sizes() && dout.sizes() == q.sizes(), "out/dout shape");
  TORCH_CHECK(out.scalar_type() == q.scalar_type() && dout.scalar_type() == q.scalar_type(),
              "out/dout dtype");
  TORCH_CHECK(out.is_contiguous() && dout.is_contiguous() && dlse.is_contiguous(), "layout");
  TORCH_CHECK(dlse.scalar_type() == torch::kFloat32, "dlse must be float32");
  const c10::cuda::CUDAGuard guard(q.device());
  const int batch = q.size(0), heads = q.size(1), length = q.size(2), dim = q.size(3);
  const int kv_heads = k.size(1);
  auto delta = torch::empty_like(row_max);
  auto dq = torch::empty_like(q), dk = torch::empty_like(k), dv = torch::empty_like(v);
  auto stream = at::cuda::getCurrentCUDAStream();
  const long rows = static_cast<long>(batch) * heads * length;
  DISPATCH(q.scalar_type(), dim, [&] {
    delta_kernel<scalar_t, HEAD><<<(rows + 255) / 256, 256, 0, stream>>>(
        ptr<scalar_t>(out), ptr<scalar_t>(dout), delta.data_ptr<float>(), rows);
    auto dq_fn = dq_kernel<scalar_t, HEAD>;
    constexpr size_t dq_bytes = dq_shared(HEAD);
    cudaFuncSetAttribute(dq_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dq_bytes);
    dq_fn<<<dim3((length + TILE - 1) / TILE, heads, batch), THREADS, dq_bytes, stream>>>(
        ptr<scalar_t>(q), ptr<scalar_t>(k), ptr<scalar_t>(v), starts.data_ptr<int>(),
        ptr<scalar_t>(dout), row_max.data_ptr<float>(), row_sum.data_ptr<float>(),
        delta.data_ptr<float>(), dlse.data_ptr<float>(), mut<scalar_t>(dq), heads, kv_heads,
        length, static_cast<float>(scale));
    auto dkv_fn = dkv_kernel<scalar_t, HEAD>;
    constexpr size_t dkv_bytes = dkv_shared(HEAD);
    cudaFuncSetAttribute(dkv_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dkv_bytes);
    dkv_fn<<<dim3((length + TILE - 1) / TILE, kv_heads, batch), THREADS, dkv_bytes, stream>>>(
        ptr<scalar_t>(q), ptr<scalar_t>(k), ptr<scalar_t>(v), starts.data_ptr<int>(),
        ptr<scalar_t>(dout), row_max.data_ptr<float>(), row_sum.data_ptr<float>(),
        delta.data_ptr<float>(), dlse.data_ptr<float>(), mut<scalar_t>(dk), mut<scalar_t>(dv),
        heads, kv_heads, length, static_cast<float>(scale));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &forward, "Document-causal attention forward");
  module.def("backward", &backward, "Document-causal attention backward");
}
