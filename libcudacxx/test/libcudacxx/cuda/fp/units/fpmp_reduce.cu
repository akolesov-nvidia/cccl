// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 compatibility with cooperative_groups::reduce.
//
//  Device-only test. Each thread contributes (seed + thread_id) and a tiled
//  sub-warp reduces the values with operator+; the reduced result of each
//  sub-warp is compared against the closed-form arithmetic-series sum. On a
//  host-only build (no CUDA) the test is SKIP()ed; under CUDA it runs on device.
//
//===----------------------------------------------------------------------===//

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifndef _CCCL_FP_STANDALONE_UNIT_TESTS
#  include <c2h/catch2_test_helper.h> // must be included in every C2H file
#endif

#include <cuda/fpmp>

#include "fp_test_targets.h"

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_CUDA_COMPILATION()
#  include <cooperative_groups.h>

#  include <cooperative_groups/reduce.h>

template <int subwarp_size, typename mp_type>
__global__ void test_reduce_kernel(unsigned int seed, mp_type* res)
{
  namespace cg            = cooperative_groups;
  auto this_block         = cg::this_thread_block();
  auto subwarp            = cg::tiled_partition<subwarp_size>(this_block);
  const auto thread_id    = this_block.thread_rank();
  const mp_type to_reduce = seed + thread_id;

  mp_type result = cg::reduce(subwarp, to_reduce, [](const mp_type& a, const mp_type& b) -> mp_type {
    return a + b;
  });
  if (subwarp.thread_rank() == 0)
  {
    res[thread_id / subwarp_size] = result;
  }
}

template <int subwarp_size, typename mp_type>
static bool test_reduce(int num_threads)
{
  const int num_subwarps  = (num_threads + subwarp_size - 1) / subwarp_size;
  const unsigned int seed = 10;

  mp_type* d_res = nullptr;
  if (cudaMalloc(&d_res, num_subwarps * sizeof(mp_type)) != cudaSuccess)
  {
    return false;
  }
  if (cudaMemset(d_res, 0, num_subwarps * sizeof(mp_type)) != cudaSuccess)
  {
    cudaFree(d_res);
    return false;
  }

  test_reduce_kernel<subwarp_size><<<1, num_threads>>>(seed, d_res);
  bool ok = (cudaGetLastError() == cudaSuccess) && (cudaDeviceSynchronize() == cudaSuccess);

  std::vector<mp_type> h_res(num_subwarps);
  ok = ok && (cudaMemcpy(h_res.data(), d_res, num_subwarps * sizeof(mp_type), cudaMemcpyDeviceToHost) == cudaSuccess);

  if (ok)
  {
    for (int i = 0; i < num_subwarps; ++i)
    {
      const double to_double = static_cast<double>(h_res[i]);
      const std::int64_t expected =
        // sum of the lowest subwarp ...
        (2 * std::int64_t{seed} + subwarp_size - 1) * subwarp_size / 2
        // ... plus the offset for higher subwarps.
        + i * subwarp_size * subwarp_size;
      if (std::abs(to_double - static_cast<double>(expected)) > 1e-4)
      {
        ok = false;
      }
    }
  }

  cudaFree(d_res);
  return ok;
}
#endif // _CCCL_CUDA_COMPILATION()

C2H_TEST("fpmp cooperative_groups::reduce", "[fpmp]")
{
#if !_CCCL_CUDA_COMPILATION()
  SKIP("cg::reduce on fpmp2 is device-only");
#else
  fp_ran_on_device();
  REQUIRE(test_reduce<4, fp32mp2>(4));
  REQUIRE(test_reduce<32, fp32mp2>(64));
  REQUIRE(test_reduce<4, fp64mp2>(4));
  REQUIRE(test_reduce<32, fp64mp2>(64));
#endif // _CCCL_CUDA_COMPILATION()
}
