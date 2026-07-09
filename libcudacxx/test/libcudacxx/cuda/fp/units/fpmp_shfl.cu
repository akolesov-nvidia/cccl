// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 warp-shuffle overloads.
//
//  Device-only test. For __shfl_sync / __shfl_xor_sync / __shfl_down_sync /
//  __shfl_up_sync, the fpmp2 overload output is compared against the scalar CUDA
//  intrinsics applied independently to the hi/lo lanes; the mismatch count must be
//  zero. On a host-only build (no CUDA) the test is SKIP()ed; under CUDA it runs
//  on the device.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_CUDA_COMPILATION()
template <typename MP2>
__device__ MP2 make_lane_value(int lane)
{
  using FpType = decltype(MP2().hi());
  FpType hi    = static_cast<FpType>(lane) + static_cast<FpType>(0.25);
  FpType lo    = static_cast<FpType>(lane - 8) * static_cast<FpType>(0.03125);
  return MP2(hi, lo);
}

template <typename MP2>
__global__ void kernel_shfl_sync(int* mismatches)
{
  const int lane      = threadIdx.x & 31;
  const unsigned mask = 0xFFFFFFFFu;
  MP2 x               = make_lane_value<MP2>(lane);
  MP2 y               = __shfl_sync(mask, x, 3, 16);
  auto ref_hi         = ::__shfl_sync(mask, x.hi(), 3, 16);
  auto ref_lo         = ::__shfl_sync(mask, x.lo(), 3, 16);
  if (y.hi() != ref_hi || y.lo() != ref_lo)
  {
    atomicAdd(mismatches, 1);
  }
}

template <typename MP2>
__global__ void kernel_shfl_xor_sync(int* mismatches)
{
  const int lane      = threadIdx.x & 31;
  const unsigned mask = 0xFFFFFFFFu;
  MP2 x               = make_lane_value<MP2>(lane);
  MP2 y               = __shfl_xor_sync(mask, x, 5, 16);
  auto ref_hi         = ::__shfl_xor_sync(mask, x.hi(), 5, 16);
  auto ref_lo         = ::__shfl_xor_sync(mask, x.lo(), 5, 16);
  if (y.hi() != ref_hi || y.lo() != ref_lo)
  {
    atomicAdd(mismatches, 1);
  }
}

template <typename MP2>
__global__ void kernel_shfl_down_sync(int* mismatches)
{
  const int lane      = threadIdx.x & 31;
  const unsigned mask = 0xFFFFFFFFu;
  MP2 x               = make_lane_value<MP2>(lane);
  MP2 y               = __shfl_down_sync(mask, x, 2u, 16);
  auto ref_hi         = ::__shfl_down_sync(mask, x.hi(), 2u, 16);
  auto ref_lo         = ::__shfl_down_sync(mask, x.lo(), 2u, 16);
  if (y.hi() != ref_hi || y.lo() != ref_lo)
  {
    atomicAdd(mismatches, 1);
  }
}

template <typename MP2>
__global__ void kernel_shfl_up_sync(int* mismatches)
{
  const int lane      = threadIdx.x & 31;
  const unsigned mask = 0xFFFFFFFFu;
  MP2 x               = make_lane_value<MP2>(lane);
  MP2 y               = __shfl_up_sync(mask, x, 2u, 16);
  auto ref_hi         = ::__shfl_up_sync(mask, x.hi(), 2u, 16);
  auto ref_lo         = ::__shfl_up_sync(mask, x.lo(), 2u, 16);
  if (y.hi() != ref_hi || y.lo() != ref_lo)
  {
    atomicAdd(mismatches, 1);
  }
}

template <typename Kern>
static bool run_op(Kern kern)
{
  int* d_mismatches = nullptr;
  if (cudaMallocManaged(&d_mismatches, sizeof(int)) != cudaSuccess)
  {
    return false;
  }
  *d_mismatches = 0;
  kern<<<1, 32>>>(d_mismatches);
  bool ok = (cudaGetLastError() == cudaSuccess) && (cudaDeviceSynchronize() == cudaSuccess) && (*d_mismatches == 0);
  cudaFree(d_mismatches);
  return ok;
}

template <typename MP2>
static bool run_type_suite()
{
  return run_op(kernel_shfl_sync<MP2>) && run_op(kernel_shfl_xor_sync<MP2>) && run_op(kernel_shfl_down_sync<MP2>)
      && run_op(kernel_shfl_up_sync<MP2>);
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp warp shuffle overloads", "[fpmp]")
{
#if !_CCCL_CUDA_COMPILATION()
  SKIP("warp shuffle on fpmp2 is device-only");
#else
  fp_ran_on_device();
  REQUIRE(run_type_suite<fp32mp2>());
  REQUIRE(run_type_suite<fp64mp2>());
#endif // _CCCL_CUDA_COMPILATION()
}
