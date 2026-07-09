// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: float-float arithmetic API (fp32mp2).
//
//  Compares native double-precision arithmetic against the float-float type
//  fp32mp2 (== fpmp2<float, fpmp2_accuracy::def>) for the basic ops (mul, add,
//  div, sub) and fma. fp32mp2 carries ~46 effective mantissa bits, so its results
//  must track the double reference to a tight relative tolerance. The same
//  _CCCL_HOST_DEVICE run_test() runs on the host directly and, under CUDA, on the
//  device via a plain kernel that writes its bool result back to managed memory
//  (no __host__ __device__ lambda, so no --extended-lambda dependency).
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cmath>

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the multi-precision floating-point type.
using ffloat = fp32mp2;

// Relative-error check against a double reference. fp32mp2 keeps ~46 mantissa
// bits (~1.4e-14 relative), so 1e-10 is a safe, still-meaningful bound.
_CCCL_HOST_DEVICE bool close(double got, double ref)
{
  const double scale = ::cuda::std::fabs(ref) > 1.0 ? ::cuda::std::fabs(ref) : 1.0;
  return ::cuda::std::fabs(got - ref) <= 1e-10 * scale;
}

// Runs each op in float-float precision and verifies it matches the double
// reference within tolerance. Returns true on success.
_CCCL_HOST_DEVICE bool run_test(double dx, double dy, double dz)
{
  // double -> fp32mp2 is a narrowing conversion, so construct explicitly.
  ffloat ex = ffloat(dx);
  ffloat ey = ffloat(dy);
  ffloat ez = ffloat(dz);

  bool ok = true;
  ok      = ok && close((double) (ex * ey), dx * dy);
  ok      = ok && close((double) (ex + ey), dx + dy);
  ok      = ok && close((double) (ex / ey), dx / dy);
  ok      = ok && close((double) (ex - ey), dx - dy);
  ok      = ok && close((double) fma(ex, ey, ez), ::cuda::std::fma(dx, dy, dz));
  return ok;
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out, double dx, double dy, double dz)
{
  *out = run_test(dx, dy, dz);
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp float-float API", "[fpmp]")
{
  // High-precision constants (as in the original example).
  const double dx = 1.123456782345678936;
  const double dy = 2.234567891234567856;
  const double dz = 3.345678901234567892;

  // Host run.
  fp_ran_on_host();
  REQUIRE(run_test(dx, dy, dz));

#if _CCCL_CUDA_COMPILATION()
  // Device run: same run_test() in a kernel, result read back via managed memory.
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok, dx, dy, dz);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
