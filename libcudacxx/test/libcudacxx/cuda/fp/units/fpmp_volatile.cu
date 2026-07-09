// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 volatile constructors / assignment + trivial copyability.
//
//  Verifies that the fp32mp2 / fp64mp2 accuracy variants are trivially copyable
//  (required for cooperative_groups, __shfl, etc.) and that they correctly support
//  construction from volatile, assignment to volatile, and assignment from
//  volatile, preserving hi/lo through volatile round-trips. The same
//  _CCCL_HOST_DEVICE run_test() runs on the host and, under CUDA, on the device.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cmath>

#include <cstdio>
#include <type_traits>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Compile-time: every accuracy variant is trivially copyable.
static_assert(std::is_trivially_copyable<fp32mp2>::value, "fp32mp2 must be trivially copyable");
static_assert(std::is_trivially_copyable<fp32mp2_low>::value, "fp32mp2_low must be trivially copyable");
static_assert(std::is_trivially_copyable<fp32mp2_high>::value, "fp32mp2_high must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64mp2>::value, "fp64mp2 must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64mp2_low>::value, "fp64mp2_low must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64mp2_high>::value, "fp64mp2_high must be trivially copyable");

// Exercise the four volatile paths for one fpmp2 type. Value checks use a
// tolerance (the double-word truncates the source double); the bit-preserving
// round-trip is checked exactly on hi/lo.
template <typename mp_type>
_CCCL_HOST_DEVICE bool vol_ok()
{
  const double v1  = 3.141592653589793;
  const double v2  = 2.718281828459045;
  const double tol = 1e-6;
  bool ok          = true;

  // Construct from volatile.
  {
    volatile mp_type vol;
    const mp_type tmp(v1);
    vol = tmp;
    mp_type non_vol(vol);
    ok = ok && (::cuda::std::fabs((double) non_vol - v1) < tol);
  }

  // Assign to volatile, read back via construct-from-volatile.
  {
    mp_type src(v1);
    volatile mp_type vol;
    vol = src;
    mp_type readback(vol);
    ok = ok && (::cuda::std::fabs((double) readback - v1) < tol);
  }

  // Assign from volatile.
  {
    volatile mp_type vol;
    const mp_type tmp(v2);
    vol = tmp;
    mp_type dst;
    dst = vol;
    ok  = ok && (::cuda::std::fabs((double) dst - v2) < tol);
  }

  // Volatile round-trip preserves hi/lo exactly.
  {
    mp_type src(v1);
    volatile mp_type vol;
    vol = src;
    mp_type dst(vol);
    ok = ok && (src.hi() == dst.hi()) && (src.lo() == dst.lo());
  }

  return ok;
}

_CCCL_HOST_DEVICE bool run_test()
{
  return vol_ok<fp32mp2>() && vol_ok<fp32mp2_low>() && vol_ok<fp32mp2_high>() && vol_ok<fp64mp2>()
      && vol_ok<fp64mp2_low>() && vol_ok<fp64mp2_high>();
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp volatile constructors + assignment", "[fpmp]")
{
  fp_ran_on_host();
  REQUIRE(run_test());

#if _CCCL_CUDA_COMPILATION()
  fp_ran_on_device();
  bool* d_ok = nullptr;
  REQUIRE_CUDART(cudaMallocManaged(&d_ok, sizeof(bool)));
  *d_ok = false;
  run_test_kernel<<<1, 1>>>(d_ok);
  REQUIRE_CUDART(cudaGetLastError());
  REQUIRE_CUDART(cudaDeviceSynchronize());
  REQUIRE(*d_ok);
  REQUIRE_CUDART(cudaFree(d_ok));
#endif // _CCCL_CUDA_COMPILATION()
}
