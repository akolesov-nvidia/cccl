// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: FP64 precision-emulation tool (fptool) with a reduced mantissa.
//
//  Builds fptool with a 23-bit (float-like) mantissa and exercises: basic
//  arithmetic, math functions (sqrt, fma), precision-sensitive operations
//  (small differences, catastrophic cancellation), accumulation error,
//  Newton-Raphson convergence, mantissa-truncation bit patterns, and
//  comparisons. Every check confirms the reduced-precision surface behaves as
//  expected relative to native double. The same TEST_HOST_DEVICE_FUNC run_test()
//  runs on the host and, under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

// Reduced precision version (23 mantissa bits like float).
#define CCCL_FP64_TOOL_MANTISSA_BITS 23
#include <cuda/fptool>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

struct TestResults
{
  double add_n, add_r;
  double sub_n, sub_r;
  double mul_n, mul_r;
  double div_n, div_r;
  double neg_n, neg_r;

  double sqrt_n, sqrt_r;
  double fma_n, fma_r;

  double small_diff_n, small_diff_r;
  double cancel_n, cancel_r;
  double mul_prec_n, mul_prec_r;

  double accum_n, accum_r;
  double newton_n, newton_r;

  ::cuda::std::uint64_t bits_orig;
  ::cuda::std::uint64_t bits_native;
  ::cuda::std::uint64_t bits_reduced;

  double cmp_eq, cmp_lt, cmp_gt;
};

// Core computation, runs on both CPU and GPU.
TEST_HOST_DEVICE_FUNC void run_precision_tests(TestResults* r)
{
  const double val_a = 1.12345678123456789;
  const double val_b = 2.12345678123456789;

  // Basic arithmetic.
  {
    double na = val_a, nb = val_b;
    cudax::fp64_tool ra = val_a, rb = val_b;

    r->add_n = (double) (na + nb);
    r->add_r = (double) (ra + rb);
    r->sub_n = (double) (na - nb);
    r->sub_r = (double) (ra - rb);
    r->mul_n = (double) (na * nb);
    r->mul_r = (double) (ra * rb);
    r->div_n = (double) (na / nb);
    r->div_r = (double) (ra / rb);
    r->neg_n = (double) (-na);
    r->neg_r = (double) (-ra);
  }

  // Math functions.
  {
    double nx           = 2.12345678123456789;
    cudax::fp64_tool rx = 2.32145678123456789;

    r->sqrt_n = ::cuda::std::sqrt(nx);
    r->sqrt_r = (double) sqrt(rx);

    double na = val_a, nb = val_b, nc = 0.5;
    cudax::fp64_tool ra = val_a, rb = val_b, rc = 0.5;

    r->fma_n = ::cuda::std::fma(na, nb, nc);
    r->fma_r = (double) fma(ra, rb, rc);
  }

  // Small difference: (1 + 1e-10) - 1.
  {
    double a  = 1.0 + 1e-10;
    double b  = 1.0;
    double na = a, nb = b;
    cudax::fp64_tool ra = a, rb = b;
    r->small_diff_n = (double) (na - nb);
    r->small_diff_r = (double) (ra - rb);
  }

  // Catastrophic cancellation: (a + b) - a.
  {
    double a = 1.0, b = 1e-10;
    double na = a, nb = b;
    cudax::fp64_tool ra = a, rb = b;
    r->cancel_n = (double) ((na + nb) - na);
    r->cancel_r = (double) ((ra + rb) - ra);
  }

  // Multiplication precision.
  {
    double a = 1.0000001, b = 1.0000002;
    double na = a, nb = b;
    cudax::fp64_tool ra = a, rb = b;
    r->mul_prec_n = (double) (na * nb);
    r->mul_prec_r = (double) (ra * rb);
  }

  // Accumulation error (sum of 1/n, n=1..1000).
  {
    double native_sum            = 0.0;
    cudax::fp64_tool reduced_sum = 0.0;
    for (int n = 1; n <= 1000; n++)
    {
      double term = 1.0 / n;
      native_sum += double(term);
      reduced_sum += cudax::fp64_tool(term);
    }
    r->accum_n = (double) native_sum;
    r->accum_r = (double) reduced_sum;
  }

  // Newton-Raphson sqrt(2): x_{n+1} = 0.5 * (x_n + S/x_n).
  {
    double n_x = 1.0, n_S = 2.0, n_half = 0.5;
    cudax::fp64_tool r_x = 1.0, r_S = 2.0, r_half = 0.5;
    for (int i = 0; i < 10; i++)
    {
      n_x = n_half * (n_x + n_S / n_x);
      r_x = r_half * (r_x + r_S / r_x);
    }
    r->newton_n = (double) n_x;
    r->newton_r = (double) r_x;
  }

  // Bit-pattern analysis (mantissa truncation).
  {
    double val                = 1.12345678123456789;
    double n_val              = val;
    cudax::fp64_tool r_val    = val;
    double n_result           = n_val + double(0.0);
    cudax::fp64_tool r_result = r_val + cudax::fp64_tool(0.0);
    double n_out              = (double) n_result;
    double r_out              = (double) r_result;
    ::cuda::std::memcpy(&r->bits_orig, &val, sizeof(::cuda::std::uint64_t));
    ::cuda::std::memcpy(&r->bits_native, &n_out, sizeof(::cuda::std::uint64_t));
    ::cuda::std::memcpy(&r->bits_reduced, &r_out, sizeof(::cuda::std::uint64_t));
  }

  // Comparison operators.
  {
    double na = val_a, nb = val_b;
    r->cmp_eq = (na == na) ? 1.0 : 0.0;
    r->cmp_lt = (na < nb) ? 1.0 : 0.0;
    r->cmp_gt = (na > nb) ? 1.0 : 0.0;
  }
}

// Verify the reduced-precision surface behaves as expected vs native double.
TEST_HOST_DEVICE_FUNC bool verify(const TestResults& r)
{
  bool ok = true;

  // Basic arithmetic: finite, and precision loss vs native where expected.
  ok = ok && ::cuda::std::isfinite(r.add_r) && ::cuda::std::isfinite(r.sub_r) && ::cuda::std::isfinite(r.mul_r)
    && ::cuda::std::isfinite(r.div_r) && ::cuda::std::isfinite(r.neg_r);
  ok = ok && (r.add_r != r.add_n) && (r.mul_r != r.mul_n) && (r.neg_r < 0.0);

  // Math functions finite.
  ok = ok && ::cuda::std::isfinite(r.sqrt_r) && ::cuda::std::isfinite(r.fma_r);

  // Precision-sensitive.
  ok = ok && (r.small_diff_n > 0.0) && (r.small_diff_r == 0.0);
  ok = ok && (r.cancel_n > 0.0) && (r.cancel_r == 0.0);
  ok = ok && (r.mul_prec_r != r.mul_prec_n);

  // Accumulation.
  ok = ok && (::cuda::std::fabs(r.accum_n - r.accum_r) > 0.0) && ::cuda::std::isfinite(r.accum_r);

  // Newton-Raphson.
  const double sqrt2        = ::cuda::std::sqrt(2.0);
  const double newton_err_n = ::cuda::std::fabs(r.newton_n - sqrt2);
  const double newton_err_r = ::cuda::std::fabs(r.newton_r - sqrt2);
  ok                        = ok && (newton_err_n < 1e-14) && (newton_err_r < 1e-5) && (newton_err_r > newton_err_n);

  // Bit patterns: reduced zeroes the low 29 bits, native keeps some.
  const ::cuda::std::uint64_t low_29_mask = (1ULL << 29) - 1;
  ok = ok && ((r.bits_native & low_29_mask) != 0) && ((r.bits_reduced & low_29_mask) == 0);

  // Comparisons.
  ok = ok && (r.cmp_eq == 1.0) && (r.cmp_lt == 1.0) && (r.cmp_gt == 0.0);

  return ok;
}

TEST_HOST_DEVICE_FUNC bool run_test()
{
  TestResults r{};
  run_precision_tests(&r);
  return verify(r);
}

TEST_HOST_DEVICE_FUNC void test()
{
  assert(run_test());
}

int main(int, char**)
{
  test();

  return 0;
}
