#ifndef __TS_FUNCTIONS_HPP__  
#define __TS_FUNCTIONS_HPP__

// Standalone compilation only
#if defined __DEVICE_FUNC__

    #include "ts.hpp"
    #include "ts_utils.hpp"
    #include "ts_types.hpp"

#endif // __DEVICE_FUNC__

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct fma_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y, Tin z) const
    {
        // The unrounded `fma()` API has no rounding-mode parameter and
        // unconditionally dispatches to the `_rn` builtin (both for native
        // double and for __fp64emu_t<m> via ADL). To honor the rm
        // template arg under ROUNDING={rz,ru,rd}, we switch to the
        // explicitly-rounded entry points, which exist for both Tin=double
        // (CUDA double intrinsics) and Tin=__fp64emu_t<m>
        // (fpemu_impl_fma.h __fma_r{n,z,u,d} templates).
        // For rn we keep the unrounded `fma()` to preserve the original
        // test target (the C-style API), which already routes to _rn.
        #if defined __USE_CUDA_BUILTINS__
            if      constexpr (rm == ts::rounding::rz) return __fma_rz(x, y, z);
            else if constexpr (rm == ts::rounding::ru) return __fma_ru(x, y, z);
            else if constexpr (rm == ts::rounding::rd) return __fma_rd(x, y, z);
            else                                       return fma(x, y, z);
        #else
            return fma(x, y, z);
        #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct dfma_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y, Tin z) const
    {
        #if defined __USE_CUDA_BUILTINS__
            if constexpr (rm == ts::rounding::rn)
                return __fma_rn(x, y, z);
            else if constexpr (rm == ts::rounding::rz) 
                return __fma_rz(x, y, z);
            else if constexpr (rm == ts::rounding::ru)
                return __fma_ru(x, y, z);
            else if constexpr (rm == ts::rounding::rd)
                return __fma_rd(x, y, z);
        #else
            return fma(x, y, z);
        #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct mad_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y, Tin z) const
    {
        if constexpr (std::is_same_v<Tin, double>) 
        {
            #if defined __CUDACC__
                return __dadd_rn(__dmul_rn(x, y), z);
            #else
                return (x * y) + z;
            #endif
        }
        else 
        {
            return mad(x, y, z);
        }
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct dot_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x1, Tin y1, Tin x2, Tin y2) const
    {
        if constexpr (std::is_same_v<Tin, double>) 
        {
            #if defined __CUDACC__
                return __dadd_rn(__dmul_rn(x1, x2), __dmul_rn(y1, y2));
            #else
                return (x1 * x2) + (y1 * y2);
            #endif
        }
        else 
        {
            return dot(x1, y1, x2, y2);
        }
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct cmul_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin xre, Tin xim, Tin yre, Tin yim) const
    {
        if constexpr (std::is_same_v<Tin, double>) 
        {
            #if defined __CUDACC__
                double re =  __dsub_rn(__dmul_rn(xre, yre), __dmul_rn(xim, yim));
                double im =  __dadd_rn(__dmul_rn(xre, yim), __dmul_rn(xim, yre));
            #else
                double re =  (xre * yre) - (xim * yim);
                double im =  (xre * yim) + (xim * yre);
            #endif
            return re + im;
        }
        else 
        {
            out_type re, im;
            cmul(xre, xim, yre, yim, re, im);
            return re + im;
        
        }
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct dmul_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y) const
    {
        #if defined __USE_CUDA_BUILTINS__
            if constexpr (rm == ts::rounding::rn)
                return __dmul_rn(x, y);
            else if constexpr (rm == ts::rounding::rz)
                return __dmul_rz(x, y);
            else if constexpr (rm == ts::rounding::ru)
                return __dmul_ru(x, y);
            else if constexpr (rm == ts::rounding::rd)
                return __dmul_rd(x, y);
        #else
            return x * y;
        #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct mul_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y) const
    {
        return x * y;
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct dadd_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y) const
    {
        #if defined __USE_CUDA_BUILTINS__
            if constexpr (rm == ts::rounding::rn)
                return __dadd_rn(x, y);
            else if constexpr (rm == ts::rounding::rz)
                return __dadd_rz(x, y);
            else if constexpr (rm == ts::rounding::ru)
                return __dadd_ru(x, y);
            else if constexpr (rm == ts::rounding::rd)
                return __dadd_rd(x, y);
        #else
            return x + y;
        #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct add_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y) const
    {
        return x + y;
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct dsub_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y) const
    {
        #if defined __USE_CUDA_BUILTINS__
            if constexpr (rm == ts::rounding::rn)
                return __dsub_rn(x, y);
            else if constexpr (rm == ts::rounding::rz)
                return __dsub_rz(x, y);
            else if constexpr (rm == ts::rounding::ru)
                return __dsub_ru(x, y);
            else if constexpr (rm == ts::rounding::rd)
                return __dsub_rd(x, y);
        #else
            return x - y;
        #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct sub_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y) const
    {
        return x - y;
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct ddiv_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y) const
    {
        #if defined __USE_CUDA_BUILTINS__
            if constexpr (rm == ts::rounding::rn)
                return __ddiv_rn(x, y);
            else if constexpr (rm == ts::rounding::rz)
                return __ddiv_rz(x, y);
            else if constexpr (rm == ts::rounding::ru)
                return __ddiv_ru(x, y);
            else if constexpr (rm == ts::rounding::rd)
                return __ddiv_rd(x, y);
        #else
            return x / y;
        #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct div_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x, Tin y) const
    {
        return x / y;
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct dsqrt_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x) const
    {
        #if defined __USE_CUDA_BUILTINS__
            if constexpr (rm == ts::rounding::rn)
                return __dsqrt_rn(x);
            else if constexpr (rm == ts::rounding::rz)
                return __dsqrt_rz(x);
            else if constexpr (rm == ts::rounding::ru)
                return __dsqrt_ru(x);
            else if constexpr (rm == ts::rounding::rd)
                return __dsqrt_rd(x);
        #else
            return sqrt(x);
        #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct sqrt_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x) const
    {
        return sqrt(x);
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct rsqrt_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x) const
    {
        return rsqrt(x);
    }
};


// #define __USE_FMA_FOR_EXP__
// Default implementation 
template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct exp_function
{
    using inp_type = Tin;
    using out_type = Tout;

    __DEVICE_DECL__ Tout operator()(Tin x) const
    {
        #define LN2_HI   (Tin)0x1.62e42fefa39efp-1 // High part of ln(2)
        #define LN2_LO   (Tin)0x1.abc9e3b39803fp-34 // Low part for extra precision
        #define INV_LN2  0x1.71547652b82fep+0  // 1 / ln(2)    // Handle special cases

        if (x != x) return x;  // NaN
        if (x > 709.782712893384) return Tout(1.0) / 0.0;  // Overflow
        if (x < -745.1332191019411) return 0.0;  // Underflow

        // Range reduction: x = k * ln2 + r,  |r| <= ln2/2
        int k = (int)(x * INV_LN2 + (x >= 0 ? 0.5 : -0.5));
        Tin r   = (x - k * LN2_HI);
        r     = (r - k * LN2_LO);

        // Polynomial approximation of exp(r), r in [-ln2/2, ln2/2]
        #if  defined __CUDA_ARCH__ && defined __USE_FMA_FOR_EXP__
        Tout poly = 0x1.a01a01a01a01ap-9;
        poly = __fma_rn(poly, r, (0x1.6c16c16c16c17p-7));
        poly = __fma_rn(poly, r, (0x1.999999999999ap-5));
        poly = __fma_rn(poly, r, (0x1.5555555555555p-3));
        poly = __fma_rn(poly, r, (0x1p-1));
        poly = __fma_rn(poly, r, (0x1p+0));
        poly = __fma_rn(poly, r, (0x1p+0));
        #else
        Tout poly = 0x1p+0+r*(
                1.0+r*(
                    0x1p-1+r*(
                    0x1.5555555555555p-3+r*(
                        0x1.999999999999ap-5+r*(
                        0x1.6c16c16c16c17p-7+r*(
                            0x1.a01a01a01a01ap-9))))));
        #endif
                
        // Reconstruct exp(x) = 2^k * exp(r)
        int exponent = k + 1023;  // Bias = 1023 for double
        if (exponent <= 0) {  // Subnormal
            if (exponent < -52) return 0.0;
            uint64_t uexp = (uint64_t)(exponent + 52) << 52;
            Tout dexp = ts::bit_cast<Tout>(uexp);
            return poly * dexp * 0x1.0p-52;
        }

        if (exponent >= 2047) return Tout(1.0) / 0.0;

        uint64_t uexp = (uint64_t)exponent << 52;
        Tout dexp = ts::bit_cast<Tout>(uexp);
        return poly * dexp;  
    } // __DEVICE_DECL__ Tout operator()(Tin x) const
};

#define C0 (1.0)
#define C1 (1.0/2.0)
#define C2 (1.0/6.0)
#define C3 (1.0/24.0)
#define C4 (1.0/120.0)
#define C5 (1.0/720.0)
#define C6 (1.0/5040.0)
#define C7 (1.0/40320.0)

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct poly4_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x) const
    {

    #if defined __CUDACC__ && defined USE_BUILTINS
       Tout poly = __dmul_rn(x,C3);
       poly   = __dadd_rn(poly,C2);
       poly   = __dmul_rn(poly,x);
       poly   = __dadd_rn(poly,C1);
       poly   = __dmul_rn(poly,x);
       poly   = __dadd_rn(poly,C0);
       return poly;
    #else
        return C0 + x * (C1 + x * (C2 + x * C3));
    #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct polyf4_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x) const
    {
    #if defined __CUDACC__
       Tout poly = __fma_rn(x,C3, C2);
       poly   = __fma_rn(x,poly, C1);
       poly   = __fma_rn(x,poly, C0);
       return poly;
    #else
        return fma(fma(fma(x,C3, C2), x, C1), x, C0);
    #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct poly8_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x) const
    {
    #if defined __CUDACC__ && defined USE_BUILTINS
       Tout poly = __dmul_rn(x,C7);
       poly   = __dadd_rn(poly,C6);
       poly   = __dmul_rn(poly,x);
       poly   = __dadd_rn(poly,C5);
       poly   = __dmul_rn(poly,x);
       poly   = __dadd_rn(poly,C4);
       poly   = __dmul_rn(poly,x);
       poly   = __dadd_rn(poly,C3);
       poly   = __dmul_rn(poly,x);
       poly   = __dadd_rn(poly,C2);
       poly   = __dmul_rn(poly,x);
       poly   = __dadd_rn(poly,C1);
       poly   = __dmul_rn(poly,x);
       poly   = __dadd_rn(poly,C0);
       return poly;
    #else
        return C0 + x * (C1 + x * (C2 + x * (C3 + x * (C4 + x * (C5 + x * (C6 + x * C7))))));
    #endif
    }
};

template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct polyf8_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x) const
    {
    #if defined __CUDACC__
       Tout poly = __fma_rn(x,C7, C6);
       poly   = __fma_rn(x,poly, C5);
       poly   = __fma_rn(x,poly, C4);
       poly   = __fma_rn(x,poly, C3);
       poly   = __fma_rn(x,poly, C2);
       poly   = __fma_rn(x,poly, C1);
       poly   = __fma_rn(x,poly, C0);
       return poly;
    #else
        return fma(fma(fma(fma(fma(fma(fma(x,C7, C6), x, C5), x, C4), x, C3), x, C2), x, C1), x, C0);
    #endif
    }
};


template<typename Tin, typename Tout, ts::rounding rm = ts::rounding::def>
struct polyu8_function
{
    using inp_type = Tin;
    using out_type = Tout;
    __DEVICE_DECL__ Tout operator()(Tin x) const
    {
        if constexpr ((std::is_same_v<Tin,double>))
        {
            return C0 + x * (C1 + x * (C2 + x * (C3 + x * (C4 + x * (C5 + x * (C6 + x * C7))))));;
        }
        else
        {
            fpbits64_unpacked_t x_unpacked = __fp64emu_unpacked_from_double(x);
            fpbits64_unpacked_t poly = __fp64emu_unpacked_mid_mad(x_unpacked,__fp64emu_unpacked_from_double(C7),
                                                                                       __fp64emu_unpacked_from_double(C6));
            poly = __fp64emu_unpacked_mid_mad(poly,x_unpacked,__fp64emu_unpacked_from_double(C5));
            poly = __fp64emu_unpacked_mid_mad(poly,x_unpacked,__fp64emu_unpacked_from_double(C4));
            poly = __fp64emu_unpacked_mid_mad(poly,x_unpacked,__fp64emu_unpacked_from_double(C3));
            poly = __fp64emu_unpacked_mid_mad(poly,x_unpacked,__fp64emu_unpacked_from_double(C2));
            poly = __fp64emu_unpacked_mid_mad(poly,x_unpacked,__fp64emu_unpacked_from_double(C1));
            poly = __fp64emu_unpacked_mid_mad(poly,x_unpacked,__fp64emu_unpacked_from_double(C0));
            double res = __fp64emu_unpacked_to_double(poly);
            return res;
        }
    }
};


// Standalone compilation of device function
#if defined __DEVICE_FUNC__

    #if defined __NATIVE_IMPL__
      #define ARGTYPE_IMPL ARGTYPE_NATIVE
      #define RESTYPE_IMPL RESTYPE_NATIVE
    #else
      #define ARGTYPE_IMPL ARGTYPE_EMU
      #define RESTYPE_IMPL RESTYPE_EMU
    #endif

    // Pass the rounding mode as the third template arg so the asm dump
    // reflects ROUNDING={rz,ru,rd} (functors like fma_function /
    // dfma_function use the rm template param to pick the right
    // rounded API). Without this, the wrapper instantiated FUNC_NAME
    // with the default rm == rounding::def == rn and silently emitted
    // the rn path regardless of the Makefile's ROUNDING value.
    // ts::rounding::__ROUNDING__ relies on the same idiom as
    // ts_types.hpp:`ts::rounding::__ROUNDING__` (the macro expands to
    // one of rn, rz, ru, rd, all of which are enumerators of ts::rounding).

    #if (FUNC_ARGS == 4)
    extern "C" __device__ RESTYPE_IMPL FUNC_IMPL(ARGTYPE_IMPL x, ARGTYPE_IMPL y, ARGTYPE_IMPL z, ARGTYPE_IMPL w)
    {
        using FunctionType = FUNC_NAME<ARGTYPE_IMPL, RESTYPE_IMPL, ts::rounding::__ROUNDING__>;
        FunctionType function;
        return function(x, y, z, w);
    }
    #elif (FUNC_ARGS == 3)
    extern "C" __device__ RESTYPE_IMPL FUNC_IMPL(ARGTYPE_IMPL x, ARGTYPE_IMPL y, ARGTYPE_IMPL z)
    {
        using FunctionType = FUNC_NAME<ARGTYPE_IMPL, RESTYPE_IMPL, ts::rounding::__ROUNDING__>;
        FunctionType function;
        return function(x, y, z);
    }
    #elif FUNC_ARGS == 2
    extern "C" __device__ RESTYPE_IMPL FUNC_IMPL(ARGTYPE_IMPL x, ARGTYPE_IMPL y)
    {
        using FunctionType = FUNC_NAME<ARGTYPE_IMPL, RESTYPE_IMPL, ts::rounding::__ROUNDING__>;
        FunctionType function;
        return function(x, y);
    }
    #else
    extern "C" __device__ RESTYPE_IMPL FUNC_IMPL(ARGTYPE_IMPL x)
    {
        using FunctionType = FUNC_NAME<ARGTYPE_IMPL, RESTYPE_IMPL, ts::rounding::__ROUNDING__>;
        FunctionType function;
        return function(x);
    }
    #endif

    /*
     * SASS-step kernel wrapper.
     *
     * The TS SASS rule needs ptxas to run as a final code-generator
     * (not in --compile-only mode) so it reports `Used N registers`
     * for FUNC_IMPL.  ptxas only does physical register allocation
     * when it sees a `__global__` entry, so this wrapper provides a
     * minimal launchable kernel that calls FUNC_IMPL.  The wrapper is
     * gated behind __TS_SASS_KERNEL__ so it is invisible to the
     * regular test executable build.  Arity matches FUNC_IMPL's
     * arity-specific definitions above (no defaults available here,
     * unlike the fpmp test suite).
     */
#if defined __TS_SASS_KERNEL__
    #if (FUNC_ARGS == 4)
    extern "C" __global__ void __ts_sass_kernel(ARGTYPE_IMPL x, ARGTYPE_IMPL y, ARGTYPE_IMPL z, ARGTYPE_IMPL w, RESTYPE_IMPL* out)
    {
        *out = FUNC_IMPL(x, y, z, w);
    }
    #elif (FUNC_ARGS == 3)
    extern "C" __global__ void __ts_sass_kernel(ARGTYPE_IMPL x, ARGTYPE_IMPL y, ARGTYPE_IMPL z, RESTYPE_IMPL* out)
    {
        *out = FUNC_IMPL(x, y, z);
    }
    #elif FUNC_ARGS == 2
    extern "C" __global__ void __ts_sass_kernel(ARGTYPE_IMPL x, ARGTYPE_IMPL y, RESTYPE_IMPL* out)
    {
        *out = FUNC_IMPL(x, y);
    }
    #else
    extern "C" __global__ void __ts_sass_kernel(ARGTYPE_IMPL x, RESTYPE_IMPL* out)
    {
        *out = FUNC_IMPL(x);
    }
    #endif
#endif

#endif // __DEVICE_FUNC__

#endif // __TS_FUNCTIONS_HPP__
