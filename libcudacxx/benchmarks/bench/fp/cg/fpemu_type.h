#ifndef FPEMU_TYPE_H
#define FPEMU_TYPE_H

#define MAD_ENABLED  1
#define DOT_ENABLED  1

#include <cuda/fpemu>

#ifndef __METHOD__
  #define __METHOD__ def
#endif

using fp64emu_t = cuda::experimental::fp64emu_t<cuda::experimental::fp64emu_accuracy::__METHOD__>;

template<typename T>
__host__ __device__ __inline__ T own_mad(T a, T b, T c)
{
    #if MAD_ENABLED == 1
    if constexpr (std::is_same_v<T, cuda::experimental::fp64emu_t<cuda::experimental::fp64emu_accuracy::def>>)
    {
        return mad(a, b, c);
    }
    else
    #endif
    {
        return a * b + c;
    }
}

template<typename T>
__host__ __device__ __inline__ T own_dot(T x1, T x2, T y1, T y2)
{
    #if DOT_ENABLED == 1
    if constexpr (std::is_same_v<T, cuda::experimental::fp64emu_t<cuda::experimental::fp64emu_accuracy::def>>)
    {
        return dot(x1, x2, y1, y2);
    }
    else
    #endif
    {
        return x1 * y1 + x2 * y2;
    }
}

// Macro to convert a macro value to a string
#define STRINGIFY(x) #x
#define ABC(x) STRINGIFY(x)

#endif // FPEMU_TYPE_H
