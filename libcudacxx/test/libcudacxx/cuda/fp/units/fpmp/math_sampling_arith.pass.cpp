// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Gross-error check over random normalized pairs: the arithmetic family
//  (add, sub, mul, div, fma, sqrt, rsqrt). See math_sampling.h.
//
//===----------------------------------------------------------------------===//

#include "math_sampling.h"

FPS_OP2(add, a + b, a + b, (fps::logr(1e-3, 1e3)), (fps::logr(1e-3, 1e3)))
FPS_OP2(sub, a - b, a - b, (fps::logr(2.0, 16.0)), (fps::logr(0.125, 0.5)))
FPS_OP2(mul, a* b, a* b, (fps::logr(1e-8, 1e8, true)), (fps::logr(1e-8, 1e8, true)))
FPS_OP2(div, a / b, a / b, (fps::logr(1e-8, 1e8, true)), (fps::logr(1e-8, 1e8, true)))
FPS_OP3(
  fma, (cuda::fma(a, b, c)), (::fma(a, b, c)), (fps::logr(1e-4, 1e4)), (fps::logr(1e-4, 1e4)), (fps::logr(1e-4, 1e4)))
FPS_OP1(sqrt, cuda::sqrt(a), ::sqrt(a), (fps::logr(1e-16, 1e16)))
FPS_OP1(rsqrt, cuda::rsqrt(a), ::rsqrt(a), (fps::logr(1e-16, 1e16)))

void run()
{
  bool ok = true;
  ok      = fps::check<op_add>("add") && ok;
  ok      = fps::check<op_sub>("sub") && ok;
  ok      = fps::check<op_mul>("mul") && ok;
  ok      = fps::check<op_div>("div") && ok;
  ok      = fps::check<op_fma>("fma") && ok;
  ok      = fps::check<op_sqrt>("sqrt") && ok;
  ok      = fps::check<op_rsqrt>("rsqrt") && ok;
  assert(ok);
}

FPS_MAIN(run)
