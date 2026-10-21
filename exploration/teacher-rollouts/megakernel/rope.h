#pragma once

#include "half_type.h"
#include <cuda_runtime.h>
#include <type_traits>

// One lane owns both coordinates. Input and output may be the same buffer;
// preserve both original values before storing either normalized rotation.
// Storage is float for decode scratch and half_t for prefill and KV caches.
template <typename Input, typename Output>
__device__ __forceinline__ void apply_rope_pair(
    const Input *input, Output *output, const half_t *norm_weight,
    int i, int p, float scale, float cv, float sv)
{
    float x0, x1;
    if constexpr (std::is_same_v<Input, float>) {
        x0 = input[i];
        x1 = input[p];
    } else {
        x0 = H2F(input[i]);
        x1 = H2F(input[p]);
    }
    x0 = x0 * scale * (1.0f + H2F(__ldg(norm_weight + i)));
    x1 = x1 * scale * (1.0f + H2F(__ldg(norm_weight + p)));
    x0 = H2F(F2H(x0)); x1 = H2F(F2H(x1));
    cv = H2F(F2H(cv)); sv = H2F(F2H(sv));
    float r0 = H2F(F2H(H2F(F2H(x0*cv)) - H2F(F2H(x1*sv))));
    float r1 = H2F(F2H(H2F(F2H(x0*sv)) + H2F(F2H(x1*cv))));
    if constexpr (std::is_same_v<Output, float>) {
        output[i] = r0;
        output[p] = r1;
    } else {
        output[i] = F2H(r0);
        output[p] = F2H(r1);
    }
}
