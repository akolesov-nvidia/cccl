// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Gross-error check over random normalized pairs: the error and probability
//  functions (erf, erfc, erfcx, erfinv, erfcinv, normcdf, normcdfinv, boys_f0).
//  See math_sampling.h.
//
//===----------------------------------------------------------------------===//

#include "math_sampling.h"

FPS_OP1(erf, cuda::erf(a), ::erf(a), (fps::lin(-6.0, 6.0)))
FPS_OP1(erfc, cuda::erfc(a), ::erfc(a), (fps::lin(-6.0, 6.0)))
FPS_OP1(erfcx, cuda::erfcx(a), ::erfcx(a), (fps::lin(-9.0, 60.0)))
FPS_OP1(erfinv, cuda::erfinv(a), ::erfinv(a), (fps::lin(-1.0 + 1e-7, 1.0 - 1e-7)))
FPS_OP1(erfcinv, cuda::erfcinv(a), ::erfcinv(a), (fps::lin(1e-7, 2.0 - 1e-7)))
FPS_OP1(normcdf, cuda::normcdf(a), ::normcdf(a), (fps::lin(-11.5, 8.4)))
FPS_OP1(normcdfinv, cuda::normcdfinv(a), ::normcdfinv(a), (fps::lin(1e-6, 1.0 - 1e-6)))
FPS_OP1(
  boys_f0, cuda::boys_f0(a), (0.5 * ::sqrt(3.14159265358979323846 / a) * ::erf(::sqrt(a))), (fps::logr(1e-10, 50.0)))

void run()
{
  bool ok = true;
  ok      = fps::check<op_erf>("erf") && ok;
  ok      = fps::check<op_erfc>("erfc") && ok;
  ok      = fps::check<op_erfcx>("erfcx") && ok;
  ok      = fps::check<op_erfinv>("erfinv") && ok;
  ok      = fps::check<op_erfcinv>("erfcinv") && ok;
  ok      = fps::check<op_normcdf>("normcdf") && ok;
  ok      = fps::check<op_normcdfinv>("normcdfinv") && ok;
  ok      = fps::check<op_boys_f0>("boys_f0") && ok;
  assert(ok);
}

FPS_MAIN(run)
