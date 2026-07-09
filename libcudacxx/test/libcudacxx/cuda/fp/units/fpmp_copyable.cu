// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fp32mp2 trivial copyability + volatile round-trip.
//
//  A compile-time static_assert checks that fp32mp2 is trivially copyable; the
//  runtime run_test() confirms a value survives a round-trip through a volatile
//  object. The same _CCCL_HOST_DEVICE run_test() runs on the host directly and,
//  under CUDA, on the device via a plain kernel that writes its bool result back
//  to managed memory.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <type_traits>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

static_assert(std::is_trivially_copyable<fp32mp2>::value, "fp32mp2 must be trivially copyable");

// Assign through a volatile object and confirm the value is preserved.
_CCCL_HOST_DEVICE bool run_test()
{
  volatile fp32mp2 vx[1];
  fp32mp2 x[1] = {fp32mp2(1.0e+20)};
  vx[0]        = x[0];
  return !(vx[0] != x[0]);
}

#if _CCCL_CUDA_COMPILATION()
__global__ void run_test_kernel(bool* out)
{
  *out = run_test();
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp trivially-copyable + volatile round-trip", "[fpmp]")
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
