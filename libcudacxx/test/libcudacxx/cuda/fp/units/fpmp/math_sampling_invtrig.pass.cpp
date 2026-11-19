// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Gross-error check over random normalized pairs: the inverse trigonometric
//  family (asin, acos, atan, atan2). See math_sampling.h.
//
//===----------------------------------------------------------------------===//

#include "math_sampling.h"

FPS_OP1(asin, cuda::asin(a), ::asin(a), (fps::lin(-1.0, 1.0)))
FPS_OP1(acos, cuda::acos(a), ::acos(a), (fps::lin(-1.0, 1.0)))
FPS_OP1(atan, cuda::atan(a), ::atan(a), (fps::logr(1e-8, 1e8, true)))
FPS_OP2(atan2, (cuda::atan2(a, b)), (::atan2(a, b)), (fps::logr(1e-8, 1e8, true)), (fps::logr(1e-8, 1e8, true)))

void run()
{
  bool ok = true;
  ok      = fps::check<op_asin>("asin") && ok;
  ok      = fps::check<op_acos>("acos") && ok;
  ok      = fps::check<op_atan>("atan") && ok;
  ok      = fps::check<op_atan2>("atan2") && ok;
  assert(ok);
}

FPS_MAIN(run)
