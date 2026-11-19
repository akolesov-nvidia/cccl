// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Gross-error check over random normalized pairs: the rounding and remainder
//  family (floor, ceil, round, trunc, rint, nearbyint, fmod, remainder).
//  See math_sampling.h.
//
//===----------------------------------------------------------------------===//

#include "math_sampling.h"

FPS_OP1(floor, cuda::floor(a), ::floor(a), (fps::logr(1e-3, 1e6, true)))
FPS_OP1(ceil, cuda::ceil(a), ::ceil(a), (fps::logr(1e-3, 1e6, true)))
FPS_OP1(round, cuda::round(a), ::round(a), (fps::logr(1e-3, 1e6, true)))
FPS_OP1(trunc, cuda::trunc(a), ::trunc(a), (fps::logr(1e-3, 1e6, true)))
FPS_OP1(rint, cuda::rint(a), ::rint(a), (fps::logr(1e-3, 1e6, true)))
FPS_OP1(nearbyint, cuda::nearbyint(a), ::nearbyint(a), (fps::logr(1e-3, 1e6, true)))
FPS_OP2(fmod, (cuda::fmod(a, b)), (::fmod(a, b)), (fps::lin(-1000.0, 1000.0)), (fps::lin(-1000.0, 1000.0)))
FPS_OP2(
  remainder, (cuda::remainder(a, b)), (::remainder(a, b)), (fps::lin(-1000.0, 1000.0)), (fps::lin(-1000.0, 1000.0)))

void run()
{
  bool ok = true;
  ok      = fps::check<op_floor>("floor") && ok;
  ok      = fps::check<op_ceil>("ceil") && ok;
  ok      = fps::check<op_round>("round") && ok;
  ok      = fps::check<op_trunc>("trunc") && ok;
  ok      = fps::check<op_rint>("rint") && ok;
  ok      = fps::check<op_nearbyint>("nearbyint") && ok;
  ok      = fps::check<op_fmod>("fmod") && ok;
  ok      = fps::check<op_remainder>("remainder") && ok;
  assert(ok);
}

FPS_MAIN(run)
