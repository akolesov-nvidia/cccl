// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Gross-error check over random normalized pairs: the exponential and logarithm
//  family (exp, exp2, exp10, expm1, log, log2, log10, log1p). See math_sampling.h.
//
//===----------------------------------------------------------------------===//

#include "math_sampling.h"

FPS_OP1(exp, cuda::exp(a), ::exp(a), (fps::lin(-64.0, 64.0)))
FPS_OP1(exp2, cuda::exp2(a), ::exp2(a), (fps::lin(-90.0, 90.0)))
FPS_OP1(exp10, cuda::exp10(a), ::exp10(a), (fps::lin(-28.0, 38.0)))
FPS_OP1(expm1, cuda::expm1(a), ::expm1(a), (fps::lin(-64.0, 64.0)))
FPS_OP1(log, cuda::log(a), ::log(a), (fps::logr(1e-16, 1e16)))
FPS_OP1(log2, cuda::log2(a), ::log2(a), (fps::logr(1e-16, 1e16)))
FPS_OP1(log10, cuda::log10(a), ::log10(a), (fps::logr(1e-16, 1e16)))
FPS_OP1(log1p, cuda::log1p(a), ::log1p(a), (fps::logr(1e-10, 1e6)))

void run()
{
  bool ok = true;
  ok      = fps::check<op_exp>("exp") && ok;
  ok      = fps::check<op_exp2>("exp2") && ok;
  ok      = fps::check<op_exp10>("exp10") && ok;
  ok      = fps::check<op_expm1>("expm1") && ok;
  ok      = fps::check<op_log>("log") && ok;
  ok      = fps::check<op_log2>("log2") && ok;
  ok      = fps::check<op_log10>("log10") && ok;
  ok      = fps::check<op_log1p>("log1p") && ok;
  assert(ok);
}

FPS_MAIN(run)
