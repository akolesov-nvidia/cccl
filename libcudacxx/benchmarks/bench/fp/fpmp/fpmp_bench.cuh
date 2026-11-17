//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

// Shared machinery of the fpmp benchmarks (bench/fp/fpmp/*.cu): one row per
// (Func, Type, Accuracy) in the CSV output, with
//
//   throughput : 2048 x 256 threads, each a chain of 1024 dependent evaluations;
//                reported as evaluations per clock per SM (ev/clk/SM) next to
//                nvbench's elements per second
//   latency    : one thread, a chain of 4096 dependent evaluations; clocks per
//                evaluation (clk/ev)
//   u2         : the maximum relative error over 2^20 random normalized pairs
//                against a double reference, in units of u^2 = 2^-48 (fp32mp2).
//                The fp64mp2 types would need a binary128 reference, which the
//                device does not have yet: they are checked against double, show
//                "-" here and cap Bits at 53
//   Regs       : registers of the chain kernel (cudaFuncGetAttributes)
//
// The SASS instruction count of one evaluation is not a run-time quantity: a
// __noinline__ "bare" function per cell is compiled into the executable, so that
// nvdisasm lists it as a labelled block of its own, and
// benchmarks/scripts/fp_table.py joins the count into the CSV.
//
// A family file declares its operations with FPB_OP1 / FPB_OP2 / FPB_OP3 (the pair
// expression, the double reference, the accuracy input range of each argument and
// the performance input) and registers them with FPB_REGISTER.

#include <thrust/device_vector.h>

#include <cuda/fpmp>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "nvbench_helper.cuh"

namespace fpb
{
// ---- axes: limb type and accuracy level ----------------------------------------------
struct type_fp32mp2
{
  using limb = float;
};
struct type_fp64mp2
{
  using limb = double;
};

template <cuda::fpmp2_accuracy A>
struct accuracy_tag
{
  static constexpr cuda::fpmp2_accuracy value = A;
};
using acc_low  = accuracy_tag<cuda::fpmp2_accuracy::low>;
using acc_mid  = accuracy_tag<cuda::fpmp2_accuracy::mid>;
using acc_high = accuracy_tag<cuda::fpmp2_accuracy::high>;

template <class Type, class Acc>
using pair_t = cuda::fpmp2<typename Type::limb, Acc::value>;

using fp_types = nvbench::type_list<type_fp32mp2, type_fp64mp2>;
using fp_accs  = nvbench::type_list<acc_low, acc_mid, acc_high>;

// ---- limb helpers ----------------------------------------------------------------------
template <class T>
using limb_t = cuda::std::remove_cvref_t<decltype(cuda::std::declval<T>().hi())>;

template <class T>
constexpr bool has_float_limbs = cuda::std::is_same_v<limb_t<T>, float>;

template <class T>
__device__ __forceinline__ double wide(const T& x)
{
  return static_cast<double>(x.hi()) + static_cast<double>(x.lo());
}

// ---- accuracy input ranges -------------------------------------------------------------
// The high limb of an accuracy input is uniform in value over [lo, hi), or, with log_scale,
// uniform in the exponent over the magnitudes [lo, hi), negated at random when `with_sign`.
struct input_range
{
  double lo;
  double hi;
  bool log_scale;
  bool with_sign;
};
__host__ __device__ constexpr input_range lin(double lo, double hi)
{
  return {lo, hi, false, false};
}
__host__ __device__ constexpr input_range logr(double lo, double hi, bool with_sign = false)
{
  return {lo, hi, true, with_sign};
}

// the performance input of the chain: a = start + tid * stride (+ 2e-4), see chain_kernel
struct perf_input
{
  float start;
  float stride;
};
__host__ __device__ constexpr perf_input perf(float start, float stride)
{
  return {start, stride};
}

// ---- random normalized pairs -----------------------------------------------------------
__device__ __forceinline__ uint32_t lcg(uint32_t& s)
{
  s = s * 1664525u + 1013904223u;
  return s;
}
// a float with its exponent in [emin, emax] and a random mantissa; the sign from a bit
__device__ __forceinline__ float rnd_float(uint32_t& s, int emin, int emax, bool with_sign)
{
  const uint32_t r    = lcg(s);
  const int e         = emin + static_cast<int>((r >> 23) % static_cast<uint32_t>(emax - emin + 1));
  const uint32_t sign = with_sign ? (r & 0x80000000u) : 0u;
  return cuda::std::bit_cast<float>(sign | (static_cast<uint32_t>(e + 127) << 23) | (r & 0x7FFFFFu));
}
// a random tail: `below` to `below + 4` binades under the head (25 for a float pair, 54 for
// a double pair), random sign and mantissa; zero when it would be subnormal as a float. A
// float pair's head and tail then span at most 53 bits, so that the pair is exactly a
// double and the reference sees the same argument: one more binade would make the
// reference round the argument, which near a zero of sin or a pole of erfinv shows up as a
// huge relative error that is the reference's, not the function's.
__device__ __forceinline__ float rnd_tail(uint32_t& s, int e_head, int below)
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
__device__ __forceinline__ double rnd_value(uint32_t& s, const input_range& r)
{
  const double u = static_cast<double>(lcg(s) >> 8) * 0x1p-24; // 24 random bits in [0, 1)
  double v       = r.log_scale ? r.lo * exp2(u * log2(r.hi / r.lo)) : r.lo + u * (r.hi - r.lo);
  if (r.with_sign && (lcg(s) & 1u))
  {
    v = -v;
  }
  return v;
}
template <class T>
__device__ __forceinline__ T random_pair(uint32_t& s, const input_range& r)
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

// ---- the dependency chain of the timing kernels -------------------------------------------
// Consecutive evaluations are made dependent by folding one bit of each limb of the result
// into the input, which moves each limb by at most one ulp (the trick of the cccl-fpdev
// suite). Both limbs take part: with the high limb alone the low limb's arithmetic of the
// `low` level, whose high limb does not depend on it, is dead code.
template <class T>
__device__ __forceinline__ void chain_bits(const T& r, T& a)
{
  using L    = limb_t<T>;
  using U    = cuda::std::conditional_t<sizeof(L) == 4, uint32_t, uint64_t>;
  const U hi = cuda::std::bit_cast<U>(a.hi()) ^ (cuda::std::bit_cast<U>(r.hi()) & U{1});
  const U lo = cuda::std::bit_cast<U>(a.lo()) ^ (cuda::std::bit_cast<U>(r.lo()) & U{1});
  a          = T(cuda::std::bit_cast<L>(hi), cuda::std::bit_cast<L>(lo));
}

// ---- the operation, by arity -------------------------------------------------------------
template <class Op, class T>
__device__ __forceinline__ T eval(const T& a, const T& b, const T& c)
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
template <class Op>
__device__ __forceinline__ double eval_ref(double a, double b, double c)
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

// ---- kernels ---------------------------------------------------------------------------------
// The inputs of the timing chain are the suite's: a = start + tid * stride, b = 1.5 + tid * 7e-4,
// c = 0.5, with the operation's own start and stride where the default would leave its domain.
template <class Op, class T>
__global__ void chain_kernel(float start, float stride, int reps, unsigned char* sink, uint32_t never)
{
  using L       = limb_t<T>;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  T a(static_cast<L>(start + static_cast<float>(tid) * stride + 0.0002f));
  const T b(static_cast<L>(1.5f + static_cast<float>(tid) * 0.0007f));
  const T c(static_cast<L>(0.5f));
  T r = a;
#pragma unroll 16
  for (int i = 0; i < reps; ++i)
  {
    r = eval<Op, T>(a, b, c);
    chain_bits(r, a);
  }
  // the sink reads both limbs, or the compiler drops the low-limb chain of the `low` level
  if (never)
  {
    sink[tid] = static_cast<unsigned char>(cuda::std::bit_cast<uint32_t>(static_cast<float>(r.hi()))
                                           ^ cuda::std::bit_cast<uint32_t>(static_cast<float>(r.lo())));
  }
}

// one evaluation on its own, for the instruction count: __noinline__ keeps it a separate
// labelled block in the SASS, and bare_kernel makes the compiler emit it (it is never launched)
template <class Op, class T>
__device__ __noinline__ T bare(const T& a, const T& b, const T& c)
{
  return eval<Op, T>(a, b, c);
}
template <class Op, class T>
__global__ void bare_kernel(const T* in, T* out)
{
  out[threadIdx.x] = bare<Op, T>(in[3 * threadIdx.x], in[3 * threadIdx.x + 1], in[3 * threadIdx.x + 2]);
}

// the maximum relative error against the double reference, as the bits of a non-negative
// double (which order like integers, so atomicMax works). A second pass with `worst` set
// records the arguments of a sample that reached the maximum.
template <class Op, class T>
__global__ void accuracy_kernel(unsigned long long* max_bits, int n, double* worst)
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
  if (!isfinite(ref) || ref == 0.0 || !isfinite(res))
  {
    return;
  }
  const double rel = fabs(res - ref) / fabs(ref);
  if (worst == nullptr)
  {
    atomicMax(max_bits, static_cast<unsigned long long>(__double_as_longlong(rel)));
  }
  else if (static_cast<unsigned long long>(__double_as_longlong(rel)) == *max_bits)
  {
    worst[0] = wide(a);
    worst[1] = wide(b);
    worst[2] = wide(c);
    worst[3] = res;
    worst[4] = ref;
  }
}

// ---- summaries -----------------------------------------------------------------------------
// add_summary returns a reference into a vector that the next add_summary may reallocate: each
// summary is added and filled in one go, after the measurement
template <class Op, class T>
void add_accuracy_summary(nvbench::state& state)
{
  constexpr int n           = 1 << 20;
  unsigned long long* d_max = nullptr;
  double* d_worst           = nullptr;
  unsigned long long b      = 0ull;
  double worst[5]           = {};
  NVBENCH_CUDA_CALL(cudaMalloc(&d_max, sizeof(unsigned long long)));
  NVBENCH_CUDA_CALL(cudaMalloc(&d_worst, sizeof(worst)));
  NVBENCH_CUDA_CALL(cudaMemset(d_max, 0, sizeof(unsigned long long)));
  NVBENCH_CUDA_CALL(cudaMemset(d_worst, 0, sizeof(worst)));
  accuracy_kernel<Op, T><<<n / 256, 256>>>(d_max, n, nullptr);
  NVBENCH_CUDA_CALL(cudaGetLastError());
  accuracy_kernel<Op, T><<<n / 256, 256>>>(d_max, n, d_worst);
  NVBENCH_CUDA_CALL(cudaGetLastError());
  NVBENCH_CUDA_CALL(cudaDeviceSynchronize());
  NVBENCH_CUDA_CALL(cudaMemcpy(&b, d_max, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  NVBENCH_CUDA_CALL(cudaMemcpy(worst, d_worst, sizeof(worst), cudaMemcpyDeviceToHost));
  NVBENCH_CUDA_CALL(cudaFree(d_max));
  NVBENCH_CUDA_CALL(cudaFree(d_worst));
  const double rel = cuda::std::bit_cast<double>(static_cast<long long>(b));

  {
    auto& u2 = state.add_summary("fp/accuracy/u2");
    u2.set_string("name", "u2");
    u2.set_string("description",
                  "Maximum relative error over 2^20 random normalized pairs, in units of u^2 = 2^-48, against a "
                  "double reference; '-' for the double-double types until a binary128 reference is available");
    if constexpr (has_float_limbs<T>)
    {
      u2.set_float64("value", rel / 0x1p-48);
    }
    else
    {
      u2.set_string("value", "-");
    }
  }
  {
    auto& bits = state.add_summary("fp/accuracy/bits");
    bits.set_string("name", "Bits");
    bits.set_string("description",
                    "Correct bits, -log2 of the maximum relative error; capped at 53 for the double-double types, "
                    "whose reference is a double for now");
    constexpr int cap = has_float_limbs<T> ? 48 : 53;
    const int value   = rel > 0.0 ? static_cast<int>(std::floor(-std::log2(rel))) : cap;
    bits.set_int64("value", value < cap ? value : cap);
  }
  {
    // the arguments of the worst sample, for a look at where an accuracy change comes from
    char buf[160];
    if (Op::arity == 1)
    {
      std::snprintf(buf, sizeof(buf), "%.9g", worst[0]);
    }
    else if (Op::arity == 2)
    {
      std::snprintf(buf, sizeof(buf), "%.9g; %.9g", worst[0], worst[1]);
    }
    else
    {
      std::snprintf(buf, sizeof(buf), "%.9g; %.9g; %.9g", worst[0], worst[1], worst[2]);
    }
    auto& w = state.add_summary("fp/accuracy/worst_args");
    w.set_string("name", "Worst at");
    w.set_string("description",
                 "The arguments (as doubles, ';'-separated) of the sample with the maximum relative error");
    w.set_string("value", rel > 0.0 ? buf : "-");
  }
}

inline void add_regs_summary(nvbench::state& state, const void* kernel)
{
  cudaFuncAttributes attr{};
  if (cudaFuncGetAttributes(&attr, kernel) == cudaSuccess)
  {
    auto& s = state.add_summary("fp/kernel/regs");
    s.set_string("name", "Regs");
    s.set_string("description", "Registers per thread of the chain kernel");
    s.set_int64("value", attr.numRegs);
  }
}

// evaluations per clock per SM (throughput) or clocks per evaluation (latency), from the mean
// cold GPU time and the device's default SM clock
inline void add_clock_summary(nvbench::state& state, std::size_t evals, bool per_sm)
{
  if (state.is_skipped() || !state.get_device().has_value())
  {
    return;
  }
  double t_gpu = -1.0;
  for (const auto& s : state.get_summaries())
  {
    if (s.get_tag() == "nv/cold/time/gpu/mean")
    {
      t_gpu = s.get_float64("value");
    }
  }
  if (t_gpu <= 0.0)
  {
    return;
  }
  const auto& dev  = *state.get_device();
  const double clk = static_cast<double>(dev.get_sm_default_clock_rate());
  const double sms = static_cast<double>(dev.get_number_of_sms());
  if (per_sm)
  {
    auto& s = state.add_summary("fp/throughput/ev_per_clk_sm");
    s.set_string("name", "ev/clk/SM");
    s.set_string("description", "Evaluations per SM clock per SM, from the mean GPU time and the default SM clock");
    s.set_float64("value", static_cast<double>(evals) / (t_gpu * clk * sms));
  }
  else
  {
    auto& s = state.add_summary("fp/latency/clk_per_ev");
    s.set_string("name", "clk/ev");
    s.set_string("description", "SM clocks per evaluation of one dependent chain on one thread");
    s.set_float64("value", t_gpu * clk / static_cast<double>(evals));
  }
}

// ---- benchmarks -----------------------------------------------------------------------------
template <class Op, class Type, class Acc>
void throughput(nvbench::state& state)
{
  using T              = pair_t<Type, Acc>;
  constexpr int blocks = 2048, threads = 256, reps = 1024;
  const std::size_t evals = static_cast<std::size_t>(blocks) * threads * reps;

  add_accuracy_summary<Op, T>(state);
  add_regs_summary(state, reinterpret_cast<const void*>(chain_kernel<Op, T>));
  static_cast<void>(reinterpret_cast<const void*>(bare_kernel<Op, T>)); // emitted, never launched

  thrust::device_vector<unsigned char> sink(static_cast<std::size_t>(blocks) * threads);
  unsigned char* d_sink = thrust::raw_pointer_cast(sink.data());

  state.add_element_count(evals, "Evaluations");
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    chain_kernel<Op, T><<<blocks, threads, 0, launch.get_stream()>>>(Op::perf.start, Op::perf.stride, reps, d_sink, 0u);
  });
  add_clock_summary(state, evals, true);
}

template <class Op, class Type, class Acc>
void latency(nvbench::state& state)
{
  using T            = pair_t<Type, Acc>;
  constexpr int reps = 4096;
  thrust::device_vector<unsigned char> sink(1);
  unsigned char* d_sink = thrust::raw_pointer_cast(sink.data());

  state.add_element_count(reps, "Evaluations");
  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    chain_kernel<Op, T><<<1, 1, 0, launch.get_stream()>>>(Op::perf.start, Op::perf.stride, reps, d_sink, 0u);
  });
  add_clock_summary(state, reps, false);
}
} // namespace fpb

NVBENCH_DECLARE_TYPE_STRINGS(fpb::type_fp32mp2, "fp32mp2", "cuda::fp32mp2 (float pair)");
NVBENCH_DECLARE_TYPE_STRINGS(fpb::type_fp64mp2, "fp64mp2", "cuda::fp64mp2 (double pair)");
NVBENCH_DECLARE_TYPE_STRINGS(fpb::acc_low, "low", "fpmp2_accuracy::low");
NVBENCH_DECLARE_TYPE_STRINGS(fpb::acc_mid, "mid", "fpmp2_accuracy::mid (default)");
NVBENCH_DECLARE_TYPE_STRINGS(fpb::acc_high, "high", "fpmp2_accuracy::high");

// the registration macros build identifiers from the function name, so the benchmark
// functions live at global scope under their own names
template <class Op, class Type, class Acc>
void fpmp_throughput(nvbench::state& state, nvbench::type_list<Op, Type, Acc>)
{
  fpb::throughput<Op, Type, Acc>(state);
}
template <class Op, class Type, class Acc>
void fpmp_latency(nvbench::state& state, nvbench::type_list<Op, Type, Acc>)
{
  fpb::latency<Op, Type, Acc>(state);
}

// ---- declaring the operations of a family --------------------------------------------------
// FPB_OPn(name, pair expression in a[, b[, c]], double expression in a[, b[, c]],
//         accuracy range of each argument, performance input)
// The expressions, the ranges and the performance input go in parentheses when they contain
// commas: (fpb::lin(-1.0, 1.0)), (fpb::perf(-0.9995f, 3.8e-6f)) or FPB_PERF_DEFAULT.
// the ranges are functions: a static constexpr data member of class type has no definition
// the device code could refer to
#define FPB_OP_COMMON(NAME, ARITY, PERF, IN0, IN1, IN2)       \
  static constexpr int arity            = ARITY;              \
  static constexpr fpb::perf_input perf = PERF;               \
  __host__ __device__ static constexpr fpb::input_range in0() \
  {                                                           \
    return IN0;                                               \
  }                                                           \
  __host__ __device__ static constexpr fpb::input_range in1() \
  {                                                           \
    return IN1;                                               \
  }                                                           \
  __host__ __device__ static constexpr fpb::input_range in2() \
  {                                                           \
    return IN2;                                               \
  }

#define FPB_OP1(NAME, PAIR_EXPR, REF_EXPR, IN0, PERF) \
  namespace fpb                                       \
  {                                                   \
  struct op_##NAME                                    \
  {                                                   \
    FPB_OP_COMMON(NAME, 1, PERF, IN0, IN0, IN0)       \
    template <class T>                                \
    __device__ static T apply(const T& a)             \
    {                                                 \
      return PAIR_EXPR;                               \
    }                                                 \
    __device__ static double ref(double a)            \
    {                                                 \
      return REF_EXPR;                                \
    }                                                 \
  };                                                  \
  }                                                   \
  NVBENCH_DECLARE_TYPE_STRINGS(fpb::op_##NAME, #NAME, "cuda::" #NAME)

#define FPB_OP2(NAME, PAIR_EXPR, REF_EXPR, IN0, IN1, PERF) \
  namespace fpb                                            \
  {                                                        \
  struct op_##NAME                                         \
  {                                                        \
    FPB_OP_COMMON(NAME, 2, PERF, IN0, IN1, IN1)            \
    template <class T>                                     \
    __device__ static T apply(const T& a, const T& b)      \
    {                                                      \
      return PAIR_EXPR;                                    \
    }                                                      \
    __device__ static double ref(double a, double b)       \
    {                                                      \
      return REF_EXPR;                                     \
    }                                                      \
  };                                                       \
  }                                                        \
  NVBENCH_DECLARE_TYPE_STRINGS(fpb::op_##NAME, #NAME, "cuda::" #NAME)

#define FPB_OP3(NAME, PAIR_EXPR, REF_EXPR, IN0, IN1, IN2, PERF)   \
  namespace fpb                                                   \
  {                                                               \
  struct op_##NAME                                                \
  {                                                               \
    FPB_OP_COMMON(NAME, 3, PERF, IN0, IN1, IN2)                   \
    template <class T>                                            \
    __device__ static T apply(const T& a, const T& b, const T& c) \
    {                                                             \
      return PAIR_EXPR;                                           \
    }                                                             \
    __device__ static double ref(double a, double b, double c)    \
    {                                                             \
      return REF_EXPR;                                            \
    }                                                             \
  };                                                              \
  }                                                               \
  NVBENCH_DECLARE_TYPE_STRINGS(fpb::op_##NAME, #NAME, "cuda::" #NAME)

// the suite's default performance input: a = 1 + tid * 1e-4 (up to 53.4 over the 2^19 threads)
#define FPB_PERF_DEFAULT (fpb::perf(1.0f, 1e-4f))

// the two benchmarks over the operations OPS (an nvbench::type_list) and every type and level
#define FPB_REGISTER(OPS)                                                                   \
  NVBENCH_BENCH_TYPES(fpmp_throughput, NVBENCH_TYPE_AXES(OPS, fpb::fp_types, fpb::fp_accs)) \
    .set_name("throughput")                                                                 \
    .set_type_axes_names({"Func", "Type", "Accuracy"});                                     \
  NVBENCH_BENCH_TYPES(fpmp_latency, NVBENCH_TYPE_AXES(OPS, fpb::fp_types, fpb::fp_accs))    \
    .set_name("latency")                                                                    \
    .set_type_axes_names({"Func", "Type", "Accuracy"})
