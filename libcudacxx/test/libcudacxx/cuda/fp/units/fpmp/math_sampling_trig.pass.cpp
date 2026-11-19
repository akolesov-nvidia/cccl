// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: nvrtc
// UNSUPPORTED: enable-tile
// UNSUPPORTED: force-tile
// error: device math intrinsics are unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Gross-error check over random normalized pairs: the trigonometric family
//  (sin, cos, tan, sinpi, cospi). See math_sampling.h.
//
//===----------------------------------------------------------------------===//

#include "math_sampling.h"

FPS_OP1(sin, cuda::sin(a), ::sin(a), (fps::lin(-1e4, 1e4)))
FPS_OP1(cos, cuda::cos(a), ::cos(a), (fps::lin(-1e4, 1e4)))
FPS_OP1(tan, cuda::tan(a), ::tan(a), (fps::lin(-1e4, 1e4)))
FPS_OP1(sinpi, cuda::sinpi(a), ::sinpi(a), (fps::lin(-1e4, 1e4)))
FPS_OP1(cospi, cuda::cospi(a), ::cospi(a), (fps::lin(-1e4, 1e4)))

void run()
{
  bool ok = true;
  ok      = fps::check<op_sin>("sin") && ok;
  ok      = fps::check<op_cos>("cos") && ok;
  ok      = fps::check<op_tan>("tan") && ok;
  ok      = fps::check<op_sinpi>("sinpi") && ok;
  ok      = fps::check<op_cospi>("cospi") && ok;
  assert(ok);
}

FPS_MAIN(run)
