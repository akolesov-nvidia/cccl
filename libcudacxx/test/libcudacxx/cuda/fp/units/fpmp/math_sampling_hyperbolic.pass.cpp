// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Gross-error check over random normalized pairs: the hyperbolic family
//  (sinh, cosh, tanh, asinh, acosh, atanh). See math_sampling.h.
//
//===----------------------------------------------------------------------===//

#include "math_sampling.h"

FPS_OP1(sinh, cuda::sinh(a), ::sinh(a), (fps::lin(-64.0, 64.0)))
FPS_OP1(cosh, cuda::cosh(a), ::cosh(a), (fps::lin(-64.0, 64.0)))
FPS_OP1(tanh, cuda::tanh(a), ::tanh(a), (fps::lin(-20.0, 20.0)))
FPS_OP1(asinh, cuda::asinh(a), ::asinh(a), (fps::logr(1e-6, 1e6, true)))
FPS_OP1(acosh, cuda::acosh(a), ::acosh(a), (fps::logr(1.0, 1e6)))
FPS_OP1(atanh, cuda::atanh(a), ::atanh(a), (fps::lin(-0.99, 0.99)))

void run()
{
  bool ok = true;
  ok      = fps::check<op_sinh>("sinh") && ok;
  ok      = fps::check<op_cosh>("cosh") && ok;
  ok      = fps::check<op_tanh>("tanh") && ok;
  ok      = fps::check<op_asinh>("asinh") && ok;
  ok      = fps::check<op_acosh>("acosh") && ok;
  ok      = fps::check<op_atanh>("atanh") && ok;
  assert(ok);
}

FPS_MAIN(run)
