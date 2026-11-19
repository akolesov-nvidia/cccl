// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Shared part of the math_sampling_<family>.pass.cpp tests: the gross-error check of the
//  fpmp2 math functions over random normalized pairs.
//
//  A test samples 2^20 random normalized pairs per function, type and accuracy level,
//  evaluates the function on the device, compares with the CUDA double library and fails
//  when the maximum relative error is above the gross-error threshold of the cccl-fpdev
//  ts/fpmp suite -- 1e-7 for fp32mp2 (the extended precision is gone), 1e-15 for fp64mp2
//  (the double reference's own error is about 1e-16) -- or when a result is not finite
//  where the reference is. The sampling is deterministic (a per-sample seed), so a run is
//  bit-reproducible on a given architecture, and the thresholds sit two orders of magnitude
//  above the measured errors, so another architecture cannot trip them.
//
//  This is the CI counterpart of the `Check` column of the fpmp benchmarks
//  (libcudacxx/benchmarks/bench/fp/fpmp), whose function list, input ranges and generator
//  these tests mirror: keep the two in step when a function is added.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CUDA_FP_UNITS_FPMP_MATH_SAMPLING_H
#define TEST_CUDA_FP_UNITS_FPMP_MATH_SAMPLING_H

#include <cuda/fpmp>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <nv/target>

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "test_macros.h"

namespace fps
{
using cuda::std::uint32_t;
using cuda::std::uint64_t;

// ---- input ranges -------------------------------------------------------------------------
// The high limb of an input is uniform in value over [lo, hi), or, with log_scale, uniform in
// the exponent over the magnitudes [lo, hi), negated at random when `with_sign`.
struct input_range
{
  double lo;
  double hi;
  bool log_scale;
  bool with_sign;
};
[[nodiscard]] __host__ __device__ constexpr input_range lin(double lo, double hi)
{
  return {lo, hi, false, false};
}
[[nodiscard]] __host__ __device__ constexpr input_range logr(double lo, double hi, bool with_sign = false)
{
  return {lo, hi, true, with_sign};
}

// ---- limb helpers -------------------------------------------------------------------------
template <class T>
using limb_t = cuda::std::remove_cvref_t<decltype(cuda::std::declval<T>().hi())>;

template <class T>
constexpr bool has_float_limbs = cuda::std::is_same_v<limb_t<T>, float>;

// the gross-error thresholds of the cccl-fpdev ts/fpmp suite on the maximum relative error
template <class T>
inline constexpr double fail_rel = has_float_limbs<T> ? 1e-7 : 1e-15;

template <class T>
[[nodiscard]] __device__ __forceinline__ double wide(const T& x)
{
  return static_cast<double>(x.hi()) + static_cast<double>(x.lo());
}

// ---- random normalized pairs (the benchmark's generator) ----------------------------------
[[nodiscard]] __device__ __forceinline__ uint32_t lcg(uint32_t& s)
{
  s = s * 1664525u + 1013904223u;
  return s;
}
// a random tail: `below` to `below + 4` binades under the head (25 for a float pair, 54 for
// a double pair), random sign and mantissa; zero when it would be subnormal as a float. A
// float pair's head and tail then span at most 53 bits, so that the pair is exactly a
// double and the reference sees the same argument.
[[nodiscard]] __device__ __forceinline__ float rnd_tail(uint32_t& s, int e_head, int below)
{
  const uint32_t r = lcg(s);
  const int e      = e_head - below - static_cast<int>(r % 5u);
  if (e + 127 < 1 || e + 127 > 254)
  {
    return 0.0f;
  }
  return cuda::std::bit_cast<float>(
    (r & 0x80000000u) | (static_cast<uint32_t>(e + 127) << 23) | ((r >> 3) & 0x7FFFFFu));
}
[[nodiscard]] __device__ __forceinline__ double rnd_value(uint32_t& s, const input_range& r)
{
  const double u = static_cast<double>(lcg(s) >> 8) * 0x1p-24; // 24 random bits in [0, 1)
  double v       = r.log_scale ? r.lo * ::exp2(u * ::log2(r.hi / r.lo)) : r.lo + u * (r.hi - r.lo);
  if (r.with_sign && (lcg(s) & 1u))
  {
    v = -v;
  }
  return v;
}
template <class T>
[[nodiscard]] __device__ __forceinline__ T random_pair(uint32_t& s, const input_range& r)
{
  using L        = limb_t<T>;
  const double v = rnd_value(s, r);
  if constexpr (cuda::std::is_same_v<L, float>)
  {
    const float hi = static_cast<float>(v);
    const int e    = static_cast<int>((cuda::std::bit_cast<uint32_t>(hi) >> 23) & 0xFFu) - 127;
    return T(hi, rnd_tail(s, e, 25));
  }
  else
  {
    // fill the mantissa bits a float cannot give with more random bits
    const uint64_t extra = static_cast<uint64_t>(lcg(s)) & ((uint64_t{1} << 29) - 1);
    const double hi      = cuda::std::bit_cast<double>(cuda::std::bit_cast<uint64_t>(v) | extra);
    const int e          = static_cast<int>((cuda::std::bit_cast<uint64_t>(hi) >> 52) & 0x7FFu) - 1023;
    return T(hi, static_cast<double>(rnd_tail(s, e, 54)));
  }
}

// ---- the operation, by arity --------------------------------------------------------------
template <class Op>
[[nodiscard]] __device__ __forceinline__ double eval_ref(double a, double b, double c)
{
  if constexpr (Op::arity == 1)
  {
    return Op::ref(a);
  }
  else if constexpr (Op::arity == 2)
  {
    return Op::ref(a, b);
  }
  else
  {
    return Op::ref(a, b, c);
  }
}
template <class Op, class T>
[[nodiscard]] __device__ __forceinline__ T eval(const T& a, const T& b, const T& c)
{
  if constexpr (Op::arity == 1)
  {
    return Op::apply(a);
  }
  else if constexpr (Op::arity == 2)
  {
    return Op::apply(a, b);
  }
  else
  {
    return Op::apply(a, b, c);
  }
}

// acc[0]: the maximum relative error as the bits of a non-negative double (they order like
// integers, so atomicMax works); acc[1]: the results that are not finite where the reference is
template <class Op, class T>
__global__ void sample_kernel(unsigned long long* acc, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
  {
    return;
  }
  uint32_t s       = 12345u + 7919u * static_cast<uint32_t>(i);
  const T a        = random_pair<T>(s, Op::in0());
  const T b        = random_pair<T>(s, Op::in1());
  const T c        = random_pair<T>(s, Op::in2());
  const T r        = eval<Op, T>(a, b, c);
  const double ref = eval_ref<Op>(wide(a), wide(b), wide(c));
  const double res = wide(r);
  if (!cuda::std::isfinite(ref) || ref == 0.0)
  {
    return;
  }
  if (!cuda::std::isfinite(res))
  {
    atomicAdd(acc + 1, 1ull);
    return;
  }
  const double rel = cuda::std::fabs(res - ref) / cuda::std::fabs(ref);
  atomicMax(acc, cuda::std::bit_cast<unsigned long long>(rel));
}

#define FPS_CUDA_CALL(expr)                                                \
  do                                                                       \
  {                                                                        \
    const cudaError_t fps_err_ = (expr);                                   \
    if (fps_err_ != cudaSuccess)                                           \
    {                                                                      \
      std::printf("%s failed: %s\n", #expr, cudaGetErrorString(fps_err_)); \
      std::abort();                                                        \
    }                                                                      \
  } while (0)

// one cell: the function on one type at one level, over 2^20 samples
template <class Op, class T>
[[nodiscard]] bool check_cell(const char* name, const char* type, const char* level)
{
  constexpr int n           = 1 << 20;
  unsigned long long* d_acc = nullptr;
  unsigned long long acc[2] = {};
  FPS_CUDA_CALL(cudaMalloc(&d_acc, sizeof(acc)));
  FPS_CUDA_CALL(cudaMemset(d_acc, 0, sizeof(acc)));
  sample_kernel<Op, T><<<n / 256, 256>>>(d_acc, n);
  FPS_CUDA_CALL(cudaGetLastError());
  FPS_CUDA_CALL(cudaDeviceSynchronize());
  FPS_CUDA_CALL(cudaMemcpy(acc, d_acc, sizeof(acc), cudaMemcpyDeviceToHost));
  FPS_CUDA_CALL(cudaFree(d_acc));
  const double rel = cuda::std::bit_cast<double>(acc[0]);
  if (acc[1] > 0 || rel > fail_rel<T>)
  {
    std::printf(
      "%s %s %s: maximum relative error %.3g (limit %.0e), %llu results not finite where the reference is\n",
      name,
      type,
      level,
      rel,
      fail_rel<T>,
      acc[1]);
    return false;
  }
  return true;
}

// one function on both pair types at the three accuracy levels
template <class Op>
[[nodiscard]] bool check(const char* name)
{
  using cuda::fpmp2_accuracy;
  bool ok = true;
  ok      = check_cell<Op, cuda::fpmp2<float, fpmp2_accuracy::low>>(name, "fp32mp2", "low") && ok;
  ok      = check_cell<Op, cuda::fpmp2<float, fpmp2_accuracy::mid>>(name, "fp32mp2", "mid") && ok;
  ok      = check_cell<Op, cuda::fpmp2<float, fpmp2_accuracy::high>>(name, "fp32mp2", "high") && ok;
  ok      = check_cell<Op, cuda::fpmp2<double, fpmp2_accuracy::low>>(name, "fp64mp2", "low") && ok;
  ok      = check_cell<Op, cuda::fpmp2<double, fpmp2_accuracy::mid>>(name, "fp64mp2", "mid") && ok;
  ok      = check_cell<Op, cuda::fpmp2<double, fpmp2_accuracy::high>>(name, "fp64mp2", "high") && ok;
  return ok;
}
} // namespace fps

// ---- declaring the functions of a family ---------------------------------------------------
// FPS_OPn(name, pair expression in a[, b[, c]], double expression in a[, b[, c]], the input
// range of each argument); expressions and ranges with commas go in parentheses.
#define FPS_OP_COMMON(ARITY, IN0, IN1, IN2)                                 \
  static constexpr int arity = ARITY;                                       \
  [[nodiscard]] __host__ __device__ static constexpr fps::input_range in0() \
  {                                                                         \
    return IN0;                                                             \
  }                                                                         \
  [[nodiscard]] __host__ __device__ static constexpr fps::input_range in1() \
  {                                                                         \
    return IN1;                                                             \
  }                                                                         \
  [[nodiscard]] __host__ __device__ static constexpr fps::input_range in2() \
  {                                                                         \
    return IN2;                                                             \
  }

#define FPS_OP1(NAME, PAIR_EXPR, REF_EXPR, IN0)          \
  struct op_##NAME                                       \
  {                                                      \
    FPS_OP_COMMON(1, IN0, IN0, IN0)                      \
    template <class T>                                   \
    [[nodiscard]] __device__ static T apply(const T& a)  \
    {                                                    \
      return PAIR_EXPR;                                  \
    }                                                    \
    [[nodiscard]] __device__ static double ref(double a) \
    {                                                    \
      return REF_EXPR;                                   \
    }                                                    \
  };

#define FPS_OP2(NAME, PAIR_EXPR, REF_EXPR, IN0, IN1)                \
  struct op_##NAME                                                  \
  {                                                                 \
    FPS_OP_COMMON(2, IN0, IN1, IN1)                                 \
    template <class T>                                              \
    [[nodiscard]] __device__ static T apply(const T& a, const T& b) \
    {                                                               \
      return PAIR_EXPR;                                             \
    }                                                               \
    [[nodiscard]] __device__ static double ref(double a, double b)  \
    {                                                               \
      return REF_EXPR;                                              \
    }                                                               \
  };

#define FPS_OP3(NAME, PAIR_EXPR, REF_EXPR, IN0, IN1, IN2)                       \
  struct op_##NAME                                                              \
  {                                                                             \
    FPS_OP_COMMON(3, IN0, IN1, IN2)                                             \
    template <class T>                                                          \
    [[nodiscard]] __device__ static T apply(const T& a, const T& b, const T& c) \
    {                                                                           \
      return PAIR_EXPR;                                                         \
    }                                                                           \
    [[nodiscard]] __device__ static double ref(double a, double b, double c)    \
    {                                                                           \
      return REF_EXPR;                                                          \
    }                                                                           \
  };

// the test's main: the checks run on the host, which launches the kernels
#define FPS_MAIN(RUN)                  \
  int main(int, char**)                \
  {                                    \
    NV_IF_TARGET(NV_IS_HOST, (RUN();)) \
    return 0;                          \
  }

#endif // TEST_CUDA_FP_UNITS_FPMP_MATH_SAMPLING_H
