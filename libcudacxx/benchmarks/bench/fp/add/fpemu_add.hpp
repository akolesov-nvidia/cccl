#ifndef __FPEMU_ADD_HPP__
#define __FPEMU_ADD_HPP__
/**
 * @file fpemu_add.hpp
 * @brief Standalone emulation of double-precision operations using integer operations
 *
 * Provides standalone integer-only emulation functions based on
 * __internal_fp64emu_mid_dadd with HA accuracy (9 extra precision bits):
 *
 *   __internal_int32_to_fp64 (int32_t i)           — int32 to double conversion
 *   __internal_fp64_add_fp64 (double x, double  y) — double + double
 *   __internal_fp64_add_int32(double x, int32_t i) — double + int32
 *
 * Normal range only: NaN, Inf, and denormals are not specially handled.
 */

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__CUDA_LIBDEVICE__)
  #define __FP64_ADD_INT32_DECL__ static __forceinline__ __host__ __device__
#else
  #define __FP64_ADD_INT32_DECL__ static inline
#endif

namespace __fp64_add_int32_impl
{
    // similar to C++20 std::bit_cast
    template<typename T, typename R>
    __FP64_ADD_INT32_DECL__ T bit_cast(const R value)
    {
        T dst;
    #if defined __DO_NOT_USE_MEMCPY__
        for (unsigned i = 0U; i < sizeof(T); i++)
        {
            unsigned char * ptrSRC = i + (unsigned char *)(&value);
            unsigned char * ptrDST = i + (unsigned char *)(&dst);
            *ptrDST = *ptrSRC;
        }
    #else
        #if !defined(__CUDA_LIBDEVICE__)
            std::memcpy(static_cast<void*>(&dst), static_cast<const void*>(&value),
                        sizeof(T));
        #else
            memcpy(static_cast<void*>(&dst), static_cast<const void*>(&value),
                        sizeof(T));
        #endif
    #endif
        return dst;
    }

    // 64-bit value represented as two 32-bit halves: x[0] = low, x[1] = high
    struct uint32x2_t { uint32_t x[2]; };

    // Count leading zeros in a 32-bit integer
    __FP64_ADD_INT32_DECL__ int32_t __internal_clz(int x)
    {
    #if defined(__CUDA_ARCH__)
        return __clz(x);
    #else
        if (x == 0) return 32;
        int n = 0;
        unsigned int u = (unsigned int)x;
        if ((u >> 16) == 0) { n += 16; u <<= 16; }
        if ((u >> 24) == 0) { n += 8;  u <<= 8;  }
        if ((u >> 28) == 0) { n += 4;  u <<= 4;  }
        if ((u >> 30) == 0) { n += 2;  u <<= 2;  }
        if ((u >> 31) == 0) { n += 1;             }
        return n;
    #endif
    }

    // Count leading zeros in a 64-bit integer
    __FP64_ADD_INT32_DECL__ int32_t __internal_clzll(int64_t x)
    {
    #if defined(__CUDA_ARCH__)
        return __clzll(x);
    #else
        uint64_t ux = (uint64_t)x;
        if (ux == 0) return 64;
        int count = 0;
        for (int i = 63; i >= 0; --i) {
            if (ux & (1ULL << i)) break;
            count++;
        }
        return count;
    #endif
    }

    // Logical left shift of 64-bit value stored as uint32x2_t
    __FP64_ADD_INT32_DECL__ uint32x2_t __shl_64(uint32x2_t man, int shift)
    {
        uint64_t man64 = bit_cast<uint64_t>(man);
        man64 <<= shift;
        return bit_cast<uint32x2_t>(man64);
    }

    // Logical right shift of 64-bit value stored as uint32x2_t
    __FP64_ADD_INT32_DECL__ uint32x2_t __shr_64(uint32x2_t man, int shift)
    {
        uint64_t man64 = bit_cast<uint64_t>(man);
    #ifndef __CUDA_ARCH__
        shift = (shift > 0) ? (shift > 64) ? 64 : shift : 0;
    #endif
        man64 = man64 >> shift;
        return bit_cast<uint32x2_t>(man64);
    }

    // Arithmetic right shift of 64-bit value (sign-extending)
    __FP64_ADD_INT32_DECL__ uint32x2_t __sar_64(uint32x2_t man, int shift)
    {
    #ifndef __CUDA_ARCH__
        shift = (shift > 0) ? (shift > 63) ? 63 : shift : 0;
    #endif
        int64_t man64 = bit_cast<int64_t>(man);
        man64 = man64 >> shift;
        return bit_cast<uint32x2_t>(man64);
    }

    // Two's complement negation of 64-bit value
    __FP64_ADD_INT32_DECL__ uint32x2_t __two_comp(uint32x2_t c)
    {
        uint64_t c64   = bit_cast<uint64_t>(c);
        uint64_t res64 = 0 - c64;
        return bit_cast<uint32x2_t>(res64);
    }

    // Unsigned 64-bit integer addition
    __FP64_ADD_INT32_DECL__ uint32x2_t __iadd_u64(uint32x2_t a, uint32x2_t b)
    {
        uint64_t a64 = bit_cast<uint64_t>(a);
        uint64_t b64 = bit_cast<uint64_t>(b);
        uint64_t res64 = a64 + b64;
        return bit_cast<uint32x2_t>(res64);
    }
} // namespace __fp64_add_int32_impl

// ====================================================================================================
// __internal_int32_to_fp64: convert int32 to double using only integer operations
// ====================================================================================================

/**
 * @brief Convert int32_t to double using only integer operations
 *
 * Produces a bit-identical result to (double)i. All int32 values are exactly
 * representable in IEEE 754 double precision.
 *
 * @param i  Integer value to convert
 * @return   (double)i
 */
__FP64_ADD_INT32_DECL__
double __internal_int32_to_fp64(int32_t i)
{
    using namespace __fp64_add_int32_impl;

    if (i == 0) return bit_cast<double>((uint64_t)0);

    uint64_t sign = (i < 0) ? (1ULL << 63) : 0ULL;
    uint32_t abs_i = (uint32_t)((i < 0) ? -(int64_t)i : (int64_t)i);

    int32_t nz = __internal_clz((int)abs_i);

    // Exponent: bias + MSB position = 1023 + (31 - nz)
    uint64_t exp = (uint64_t)(1023 + 31 - nz);

    // Mantissa: shift abs_i so MSB lands at bit 52, then clear implicit bit
    uint64_t mantissa = ((uint64_t)abs_i << (21 + nz)) & 0x000FFFFFFFFFFFFFull;

    return bit_cast<double>(sign | (exp << 52) | mantissa);
}

// ====================================================================================================
// __internal_fp64_add_fp64: double + double using only integer operations
// ====================================================================================================

/**
 * @brief Add two double precision values using only integer operations
 *
 * Standalone emulation function that performs (x + y) where both operands are
 * doubles. The implementation uses the same integer-only mantissa alignment and
 * addition algorithm as __internal_fp64emu_mid_dadd (HA accuracy, normal range).
 *
 * @param x  First double precision operand
 * @param y  Second double precision operand
 * @return   Result of (x + y) as double
 */
__FP64_ADD_INT32_DECL__
double __internal_fp64_add_fp64(double x, double y)
{
    using namespace __fp64_add_int32_impl;
    using uint32x2_t = __fp64_add_int32_impl::uint32x2_t;

    constexpr int32_t  extra_bits = 9;
    constexpr uint32_t HI_MANT_SHIFT = 20;
    constexpr uint32_t LO_EXP_MASK   = 0x000007FFu;
    constexpr uint32_t HI_SIGN_MASK  = 0x80000000u;
    constexpr uint32_t HI_MANT_MASK  = 0x000FFFFFu;

    // Unpack operand A (double x)
    uint32x2_t a_32x2 = bit_cast<uint32x2_t>(x);

    uint32_t exp_a  = (a_32x2.x[1] >> HI_MANT_SHIFT) & LO_EXP_MASK;
    uint32_t sign_a = a_32x2.x[1] & HI_SIGN_MASK;
    a_32x2.x[1] &= HI_MANT_MASK;
    if (exp_a != 0) a_32x2.x[1] |= (1u << HI_MANT_SHIFT);

    a_32x2 = __shl_64(a_32x2, extra_bits);
    if (sign_a) a_32x2 = __two_comp(a_32x2);

    // Unpack operand B (double y)
    uint32x2_t b_32x2 = bit_cast<uint32x2_t>(y);

    uint32_t exp_b  = (b_32x2.x[1] >> HI_MANT_SHIFT) & LO_EXP_MASK;
    uint32_t sign_b = b_32x2.x[1] & HI_SIGN_MASK;
    b_32x2.x[1] &= HI_MANT_MASK;
    if (exp_b != 0) b_32x2.x[1] |= (1u << HI_MANT_SHIFT);

    b_32x2 = __shl_64(b_32x2, extra_bits);
    if (sign_b) b_32x2 = __two_comp(b_32x2);

    // Align exponents and add mantissas
    int32_t exp_c = (exp_a > exp_b) ? exp_a : exp_b;

    int32_t delta_a = exp_c - exp_a;
    int32_t delta_b = exp_c - exp_b;

    a_32x2 = __sar_64(a_32x2, delta_a);
    b_32x2 = __sar_64(b_32x2, delta_b);

    uint32x2_t c_32x2 = __iadd_u64(a_32x2, b_32x2);

    // Normalize result
    uint32_t sign_c = c_32x2.x[1] & HI_SIGN_MASK;

    if (sign_c) c_32x2 = __two_comp(c_32x2);

    int32_t nzeros =
        __internal_clzll(bit_cast<int64_t>(c_32x2));

    int32_t exp_corr = (nzeros - (11 - 1 - extra_bits));
    exp_c -= exp_corr;

    constexpr uint32x2_t zero_32x2 = {0, 0};
    if (exp_c < 0) { exp_c = 0; c_32x2 = zero_32x2; }
    if ((exp_c < 0x000007ff) && (nzeros == 64)) exp_c = 0;

    c_32x2 = __shl_64(c_32x2, exp_corr);

    // Pack result
    c_32x2 = __shr_64(c_32x2, extra_bits + 1);

    c_32x2.x[1] += (exp_c << HI_MANT_SHIFT);

    c_32x2.x[1] |= sign_c;

    return bit_cast<double>(c_32x2);
}

// ====================================================================================================
// __internal_fp64_add_int32: fused version (no intermediate IEEE 754 pack/unpack)
// ====================================================================================================

/**
 * @brief Add int32 value to double using only integer operations (fused)
 *
 * Fused implementation that constructs the working mantissa directly from the
 * integer, avoiding the pack-to-IEEE-754 / unpack-from-IEEE-754 round-trip
 * that the composed version (__internal_int32_to_fp64 + __internal_fp64_add_fp64)
 * would incur.
 *
 * @param x  Double precision accumulator
 * @param i  Integer value to add
 * @return   Result of (x + i) as double
 */
__FP64_ADD_INT32_DECL__
double __internal_fp64_add_int32(double x, int32_t i)
{
    using namespace __fp64_add_int32_impl;
    using uint32x2_t = __fp64_add_int32_impl::uint32x2_t;

    constexpr int32_t  extra_bits = 9;
    constexpr uint32_t HI_MANT_SHIFT = 20;
    constexpr uint32_t LO_EXP_MASK   = 0x000007FFu;
    constexpr uint32_t HI_SIGN_MASK  = 0x80000000u;
    constexpr uint32_t HI_MANT_MASK  = 0x000FFFFFu;

    // Unpack operand A (double x)
    uint32x2_t a_32x2 = bit_cast<uint32x2_t>(x);

    uint32_t exp_a  = (a_32x2.x[1] >> HI_MANT_SHIFT) & LO_EXP_MASK;
    uint32_t sign_a = a_32x2.x[1] & HI_SIGN_MASK;
    a_32x2.x[1] &= HI_MANT_MASK;
    if (exp_a != 0) a_32x2.x[1] |= (1u << HI_MANT_SHIFT);

    a_32x2 = __shl_64(a_32x2, extra_bits);
    if (sign_a) a_32x2 = __two_comp(a_32x2);

    // Operand B: directly from int32 — no IEEE 754 intermediate
    uint32_t sign_b = 0;
    uint32_t exp_b  = 0;
    uint32x2_t b_32x2 = {0, 0};

    if (i != 0) {
        sign_b = (i < 0) ? HI_SIGN_MASK : 0;
        uint32_t abs_i = (uint32_t)((i < 0) ? -(int64_t)i : (int64_t)i);
        int32_t nz = __internal_clz((int)abs_i);
        exp_b = 1023 + 31 - nz;
        // Place MSB at bit 61 (= bit 52 + extra_bits), matching the shifted working format
        b_32x2 = bit_cast<uint32x2_t>((uint64_t)abs_i << (30 + nz));
        if (sign_b) b_32x2 = __two_comp(b_32x2);
    }

    // Align exponents and add mantissas
    int32_t exp_c = (exp_a > exp_b) ? exp_a : exp_b;

    int32_t delta_a = exp_c - exp_a;
    int32_t delta_b = exp_c - exp_b;

    a_32x2 = __sar_64(a_32x2, delta_a);
    b_32x2 = __sar_64(b_32x2, delta_b);

    uint32x2_t c_32x2 = __iadd_u64(a_32x2, b_32x2);

    // Normalize result
    uint32_t sign_c = c_32x2.x[1] & HI_SIGN_MASK;

    if (sign_c) c_32x2 = __two_comp(c_32x2);

    int32_t nzeros =
        __internal_clzll(bit_cast<int64_t>(c_32x2));

    int32_t exp_corr = (nzeros - (11 - 1 - extra_bits));
    exp_c -= exp_corr;

    constexpr uint32x2_t zero_32x2 = {0, 0};
    if (exp_c < 0) { exp_c = 0; c_32x2 = zero_32x2; }
    if ((exp_c < 0x000007ff) && (nzeros == 64)) exp_c = 0;

    c_32x2 = __shl_64(c_32x2, exp_corr);

    // Pack result
    c_32x2 = __shr_64(c_32x2, extra_bits + 1);

    c_32x2.x[1] += (exp_c << HI_MANT_SHIFT);

    c_32x2.x[1] |= sign_c;

    return bit_cast<double>(c_32x2);
}

#endif // __FPEMU_ADD_HPP__
