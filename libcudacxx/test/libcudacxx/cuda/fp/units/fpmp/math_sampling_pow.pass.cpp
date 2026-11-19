// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Gross-error check over random normalized pairs: the power family
//  (pow, cbrt, rcbrt, hypot, rhypot). See math_sampling.h.
//
//===----------------------------------------------------------------------===//

#include "math_sampling.h"

FPS_OP2(pow, (cuda::pow(a, b)), (::pow(a, b)), (fps::logr(0.01, 100.0)), (fps::lin(-10.0, 10.0)))
FPS_OP1(cbrt, cuda::cbrt(a), ::cbrt(a), (fps::logr(1e-16, 1e16, true)))
FPS_OP1(rcbrt, cuda::rcbrt(a), ::rcbrt(a), (fps::logr(1e-16, 1e16, true)))
FPS_OP2(hypot, (cuda::hypot(a, b)), (::hypot(a, b)), (fps::logr(1e-8, 1e8, true)), (fps::logr(1e-8, 1e8, true)))
FPS_OP2(rhypot, (cuda::rhypot(a, b)), (::rhypot(a, b)), (fps::logr(1e-8, 1e8, true)), (fps::logr(1e-8, 1e8, true)))

void run()
{
  bool ok = true;
  ok      = fps::check<op_pow>("pow") && ok;
  ok      = fps::check<op_cbrt>("cbrt") && ok;
  ok      = fps::check<op_rcbrt>("rcbrt") && ok;
  ok      = fps::check<op_hypot>("hypot") && ok;
  ok      = fps::check<op_rhypot>("rhypot") && ok;
  assert(ok);
}

FPS_MAIN(run)
