CUDA Multi-Precision Floating-Point Library Specification
=========================================================

*Generated: 2026-05-22 11:03:31*

Overview
--------

This document provides the complete specification for the ``cuda_multi_fp`` library,
including accuracy characteristics, special value handling, and GPU performance benchmarks.

Supported Data Types
--------------------

=========== =========================== ============= ========
Type        Description                 Mantissa Bits Range
=========== =========================== ============= ========
``fp32mp2`` Double-float (two floats)   ~46 bits      ±3.4e38
``fp64mp2`` Double-double (two doubles) ~104 bits     ±1.8e308
=========== =========================== ============= ========

Function Families
-----------------

The math functions are organized into families that mirror the CUDA C++
mathematical standard library taxonomy (CUDA C++ Programming Guide,
"Mathematical Functions"). Each family lives in a dedicated implementation
header (``fpmp_math_impl_<family>.h``) that contains both the ``fp32mp2``
implementation and the ``fp64mp2`` specialization for its functions. Shared
kernels and constants live in ``fpmp_math_impl.h``. The public header
``fpmp_math.h`` includes all family headers and provides the overloaded ``fpmp2``
API wrappers (template declarations, ``float``/``double`` specializations, the
freestanding API, and library-mode declarations). Basic arithmetic (``add``,
``sub``, ``mul``, ``div``, ``fma``, ``mad``) and ``sqrt``/``rsqrt`` are part of
the core ``fpmp.h`` header. Functions that CUDA lists as "non-standard" are
folded into their natural standard family.

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Family
     - Implementation header
     - Functions
   * - Common utilities
     - ``fpmp_math_impl.h``
     - error-free transforms, Horner/polynomial evaluation, ``fp32mp2``/``fp64mp2``
       constants, argument reduction (Cody–Waite, Payne–Hanek), exponent
       split/scale kernels
   * - Exponential
     - ``fpmp_math_impl_exp.h``
     - ``exp``, ``exp2``, ``exp10``, ``expm1``, ``log``, ``log2``, ``log10``, ``log1p``
   * - Power
     - ``fpmp_math_impl_pow.h``
     - ``pow``, ``cbrt``, ``rcbrt``, ``hypot``, ``rhypot``, ``norm3d``, ``norm4d``,
       ``rnorm3d``, ``rnorm4d``
   * - Trigonometric
     - ``fpmp_math_impl_trig.h``
     - ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``atan2``,
       ``sincos``, ``sinpi``, ``cospi``, ``sincospi``
   * - Hyperbolic
     - ``fpmp_math_impl_hyperbolic.h``
     - ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, ``atanh``
   * - Error, gamma & special
     - ``fpmp_math_impl_special.h``
     - ``erf``, ``erfc``, ``erfinv``, ``erfcinv``, ``erfcx``, ``tgamma``,
       ``lgamma``, ``normcdf``, ``normcdfinv``, ``boys_f0``, ``icdf``, ``j0``,
       ``j1``, ``jn``, ``y0``, ``y1``, ``yn``, ``cyl_bessel_i0``, ``cyl_bessel_i1``
   * - Nearest integer & remainder
     - ``fpmp_math_impl_nearint.h``
     - ``ceil``, ``floor``, ``trunc``, ``round``, ``nearbyint``, ``rint``,
       ``lrint``, ``llrint``, ``lround``, ``llround``, ``fmod``, ``remainder``,
       ``remquo``
   * - Floating-point manipulation
     - ``fpmp_math_impl_manip.h``
     - ``frexp``, ``ldexp``, ``modf``, ``scalbn``, ``scalbln``, ``ilogb``,
       ``logb``, ``nextafter``, ``copysign``, ``fabs``
   * - Classification & comparison
     - ``fpmp_math_impl_classify.h``
     - ``isfinite``, ``isinf``, ``isnan``, ``signbit``, ``fmax``, ``fmin``,
       ``max``, ``min``, ``fdim``

Table of Contents
-----------------

`Function Families <#function-families>`__

`Arithmetic Operations <#arithmetic-operations>`__

-  `Addition (add) <#addition-add>`__
-  `Subtraction (sub) <#subtraction-sub>`__
-  `Multiplication (mul) <#multiplication-mul>`__
-  `Division (div) <#division-div>`__
-  `Accumulate (acc) <#accumulate-acc>`__
-  `Fused Multiply-Add (fma) <#fused-multiply-add-fma>`__
-  `Multiply-Add (mad) <#multiply-add-mad>`__

`Mathematical Functions <#mathematical-functions>`__

-  `Square Root (sqrt) <#square-root-sqrt>`__
-  `Reciprocal Square Root (rsqrt) <#reciprocal-square-root-rsqrt>`__
-  `Exponential (exp) <#exponential-exp>`__
-  `Natural Logarithm (log) <#natural-logarithm-log>`__
-  `Power (pow) <#power-pow>`__
-  `Cube Root (cbrt) <#cube-root-cbrt>`__
-  `Reciprocal Cube Root (rcbrt) <#reciprocal-cube-root-rcbrt>`__
-  `Sine (sin) <#sine-sin>`__
-  `Cosine (cos) <#cosine-cos>`__
-  `Hyperbolic Tangent (tanh) <#hyperbolic-tangent-tanh>`__
-  `Error Function (erf) <#error-function-erf>`__
-  `Complementary Error Function (erfc) <#complementary-error-function-erfc>`__
-  `Boys Function F0 (boys_f0) <#boys-function-f0-boys_f0>`__
-  `Inverse Normal CDF (normcdfinv) <#inverse-normal-cdf-normcdfinv>`__
-  `Floor (floor) <#floor-floor>`__
-  `Ceiling (ceil) <#ceiling-ceil>`__
-  `Round to Nearest (round) <#round-to-nearest-round>`__
-  `Truncate (trunc) <#truncate-trunc>`__

`Comparison Operations <#comparison-operations>`__

-  `Equal (eq) <#equal-eq>`__
-  `Not Equal (ne) <#not-equal-ne>`__
-  `Less Than (lt) <#less-than-lt>`__
-  `Less Than or Equal (le) <#less-than-or-equal-le>`__
-  `Greater Than (gt) <#greater-than-gt>`__
-  `Greater Than or Equal (ge) <#greater-than-or-equal-ge>`__

`Type Conversions <#type-conversions>`__

-  `To Int32 (mp2int) <#to-int32-mp2int>`__
-  `To UInt32 (mp2uint) <#to-uint32-mp2uint>`__
-  `To Int64 (mp2ll) <#to-int64-mp2ll>`__
-  `To UInt64 (mp2ull) <#to-uint64-mp2ull>`__
-  `From Int32 (int2mp) <#from-int32-int2mp>`__
-  `From UInt32 (uint2mp) <#from-uint32-uint2mp>`__
-  `From Int64 (ll2mp) <#from-int64-ll2mp>`__
-  `From UInt64 (ull2mp) <#from-uint64-ull2mp>`__
-  `To Native Float (mp2fp) <#to-native-float-mp2fp>`__
-  `From Native Float (fp2mp) <#from-native-float-fp2mp>`__

`Appendix: Legends <#appendix-legends>`__

-  `Measured Accuracy Legend <#measured-accuracy-legend>`__
-  `Special Values Legend (Floating Point) <#special-values-legend-floating-point>`__
-  `Special Values Legend (Integer Conversions) <#special-values-legend-integer-conversions>`__
-  `Performance Metrics Legend <#performance-metrics-legend>`__
-  `SASS Instructions Legend <#sass-instructions-legend>`__
-  `SASS Instructions Summary <#sass-instructions-summary>`__

Arithmetic Operations
=====================

Addition (add)
--------------

Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4261021648 100.00% 1.00e-13   1.63e-16   43
input denormal 18         4e-07%  3.95e-13   2.02e-13   41
input near inf 1180       3e-05%  2.17e-10   8.95e-13   32
cancellation   14         3e-07%  1.27e-08   4.27e-09   26
unclassified   133269     3e-03%  1.21e-09   9.65e-13   29
TOTAL          4261156129 100.00%                       
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    3990.8  -0.19x  5.45x   2495.2 -0.19x  -0.36x
ev/clk/SM 8.74    -0.19x  5.45x   8.58   -0.19x  -0.36x
clk/ev    39.6    -0.34x  2.80x   41.6   -0.33x  -0.54x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      11
fp64      0
other     0
**total** **11**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156129 100.00% 7.11e-15   1.06e-16   47
TOTAL       4261156129 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2387.1  -0.11x  3.26x   1484.7 -0.11x  -0.21x
ev/clk/SM 5.23    -0.11x  3.26x   5.11   -0.11x  -0.21x
clk/ev    59.1    -0.22x  1.87x   60.9   -0.23x  -0.37x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      20
fp64      0
other     0
**total** **20**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     4261021648 99.99%  1.00e-13   1.63e-16   43
output special  130965     3e-03%  --         --         --
input special   5          1e-07%  --         --         --
output denormal 15196      4e-04%  0.00e+00   0.00e+00   0
input denormal  18         4e-07%  3.95e-13   2.02e-13   41
input near inf  1180       3e-05%  2.17e-10   8.95e-13   32
cancellation    13         3e-07%  1.27e-08   4.52e-09   26
unclassified    133270     3e-03%  1.21e-09   9.72e-13   29
TOTAL           4261302295 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    4979.5  -0.24x  6.80x   3103.4 -0.24x  -0.44x
ev/clk/SM 10.90   -0.24x  6.80x   10.67  -0.24x  -0.44x
clk/ev    21.7    -0.60x  5.03x   22.2   -0.61x  1.02x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      8
fp64      0
other     1
**total** **9**
========= =====

**Special Values Table:**

========= ==== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======= ==== ====
**a\b**   -INF -maxN    -1       -minN    -maxD    -minD    -0       +0       +minD    +maxD    +minN    +1       +maxN   +INF QNAN
========= ==== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======= ==== ====
**-INF**  nan  nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan     nan  nan
**-maxN** nan  nan      -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 +0      nan  nan
**-1**    nan  -3.4e+38 -2       -1       -1       -1       -1       -1       -1       -1       -1       +0       3.4e+38 nan  nan
**-minN** nan  -3.4e+38 -1       -2.4e-38 -2.4e-38 -1.2e-38 -1.2e-38 -1.2e-38 -1.2e-38 -1.4e-45 +0       1        3.4e+38 nan  nan
**-maxD** nan  -3.4e+38 -1       -2.4e-38 -2.4e-38 -1.2e-38 -1.2e-38 -1.2e-38 -1.2e-38 +0       1.4e-45  1        3.4e+38 nan  nan
**-minD** nan  -3.4e+38 -1       -1.2e-38 -1.2e-38 -2.8e-45 -1.4e-45 -1.4e-45 +0       1.2e-38  1.2e-38  1        3.4e+38 nan  nan
**-0**    nan  -3.4e+38 -1       -1.2e-38 -1.2e-38 -1.4e-45 +0       +0       1.4e-45  1.2e-38  1.2e-38  1        3.4e+38 nan  nan
**+0**    nan  -3.4e+38 -1       -1.2e-38 -1.2e-38 -1.4e-45 +0       +0       1.4e-45  1.2e-38  1.2e-38  1        3.4e+38 nan  nan
**+minD** nan  -3.4e+38 -1       -1.2e-38 -1.2e-38 +0       1.4e-45  1.4e-45  2.8e-45  1.2e-38  1.2e-38  1        3.4e+38 nan  nan
**+maxD** nan  -3.4e+38 -1       -1.4e-45 +0       1.2e-38  1.2e-38  1.2e-38  1.2e-38  2.4e-38  2.4e-38  1        3.4e+38 nan  nan
**+minN** nan  -3.4e+38 -1       +0       1.4e-45  1.2e-38  1.2e-38  1.2e-38  1.2e-38  2.4e-38  2.4e-38  1        3.4e+38 nan  nan
**+1**    nan  -3.4e+38 +0       1        1        1        1        1        1        1        1        2        3.4e+38 nan  nan
**+maxN** nan  +0       3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  nan     nan  nan
**+INF**  nan  nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan     nan  nan
**QNAN**  nan  nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan     nan  nan
========= ==== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======= ==== ====

Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============ ======== ======= ========== ========== ====
Class        Count    Percent Max RelErr Avg RelErr Bits
============ ======== ======= ========== ========== ====
normal (OK)  16760813 100.00% 9.54e-29   7.62e-36   93
unclassified 1        6e-06%  2.31e-27   2.31e-27   88
TOTAL        16760814 100.00%                       
============ ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    66.9    -0.09x  -0.39x   1265.2 -0.18x  7.11x
ev/clk/SM 0.15    -0.09x  -0.39x   4.35   -0.18x  7.11x
clk/ev    769.9   -0.14x  -0.23x   80.7   -0.28x  2.14x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      11
other     0
**total** **11**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760814 100.00% 2.46e-32   -2.94e-37  105
TOTAL       16760814 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    36.8    -0.05x  -0.22x   766.2 -0.11x  4.30x
ev/clk/SM 0.08    -0.05x  -0.22x   2.63  -0.11x  4.30x
clk/ev    1403.5  -0.08x  -0.13x   114.8 -0.20x  1.51x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      20
other     0
**total** **20**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     16760813 100.00% 9.54e-29   7.62e-36   93
output special  2        1e-05%  --         --         --
output denormal 8        5e-05%  0.00e+00   0.00e+00   0
unclassified    1        6e-06%  2.31e-27   2.31e-27   88
TOTAL           16760824 100.00%                       
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    92.0    -0.13x  -0.54x   1675.2 -0.24x  9.40x
ev/clk/SM 0.20    -0.13x  -0.54x   5.76   -0.24x  9.40x
clk/ev    585.4   -0.19x  -0.31x   42.2   -0.53x  4.10x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      8
other     4
**total** **12**
========= ======

**Special Values Table:**

========= ==== ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ======== ==== ====
**a\b**   -INF -maxN     -1        -minN     -maxD     -minD     -0        +0        +minD     +maxD     +minN     +1        +maxN    +INF QNAN
========= ==== ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ======== ==== ====
**-INF**  nan  nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan      nan  nan
**-maxN** nan  nan       -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 +0       nan  nan
**-1**    nan  -1.8e+308 -2        -1        -1        -1        -1        -1        -1        -1        -1        +0        1.8e+308 nan  nan
**-minN** nan  -1.8e+308 -1        -4.5e-308 -4.5e-308 -2.2e-308 -2.2e-308 -2.2e-308 -2.2e-308 -4.9e-324 +0        1         1.8e+308 nan  nan
**-maxD** nan  -1.8e+308 -1        -4.5e-308 -4.5e-308 -2.2e-308 -2.2e-308 -2.2e-308 -2.2e-308 +0        4.9e-324  1         1.8e+308 nan  nan
**-minD** nan  -1.8e+308 -1        -2.2e-308 -2.2e-308 -9.9e-324 -4.9e-324 -4.9e-324 +0        2.2e-308  2.2e-308  1         1.8e+308 nan  nan
**-0**    nan  -1.8e+308 -1        -2.2e-308 -2.2e-308 -4.9e-324 +0        +0        4.9e-324  2.2e-308  2.2e-308  1         1.8e+308 nan  nan
**+0**    nan  -1.8e+308 -1        -2.2e-308 -2.2e-308 -4.9e-324 +0        +0        4.9e-324  2.2e-308  2.2e-308  1         1.8e+308 nan  nan
**+minD** nan  -1.8e+308 -1        -2.2e-308 -2.2e-308 +0        4.9e-324  4.9e-324  9.9e-324  2.2e-308  2.2e-308  1         1.8e+308 nan  nan
**+maxD** nan  -1.8e+308 -1        -4.9e-324 +0        2.2e-308  2.2e-308  2.2e-308  2.2e-308  4.5e-308  4.5e-308  1         1.8e+308 nan  nan
**+minN** nan  -1.8e+308 -1        +0        4.9e-324  2.2e-308  2.2e-308  2.2e-308  2.2e-308  4.5e-308  4.5e-308  1         1.8e+308 nan  nan
**+1**    nan  -1.8e+308 +0        1         1         1         1         1         1         1         1         2         1.8e+308 nan  nan
**+maxN** nan  +0        1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  nan      nan  nan
**+INF**  nan  nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan      nan  nan
**QNAN**  nan  nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan      nan  nan
========= ==== ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ======== ==== ====

--------------

Subtraction (sub)
-----------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4261021648 100.00% 1.00e-13   1.64e-16   43
input denormal 19         4e-07%  4.12e-13   1.76e-13   41
input near inf 1170       3e-05%  1.53e-10   9.65e-13   32
cancellation   12         3e-07%  4.97e-08   9.45e-09   24
unclassified   133285     3e-03%  1.13e-09   9.52e-13   29
TOTAL          4261156134 100.00%                       
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    3990.3  -0.19x  5.45x   2497.1 -0.19x  -0.36x
ev/clk/SM 8.73    -0.19x  5.45x   8.59   -0.19x  -0.36x
clk/ev    39.8    -0.34x  2.74x   41.2   -0.33x  -0.55x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      11
fp64      0
other     0
**total** **11**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156134 100.00% 7.11e-15   1.06e-16   47
TOTAL       4261156134 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2390.7  -0.11x  3.29x   1485.4 -0.11x  -0.21x
ev/clk/SM 5.23    -0.11x  3.29x   5.11   -0.11x  -0.21x
clk/ev    58.8    -0.22x  1.86x   60.9   -0.23x  -0.37x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      20
fp64      0
other     0
**total** **20**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          4261021649 99.99%  1.00e-13   1.64e-16   43
output special       131026     3e-03%  --         --         --
input special        5          1e-07%  --         --         --
output denormal      15041      4e-04%  0.00e+00   0.00e+00   0
input denormal       19         4e-07%  4.12e-13   1.76e-13   41
output near denormal 1          2e-08%  4.97e-08   4.97e-08   24
input near inf       1170       3e-05%  1.53e-10   9.65e-13   32
cancellation         11         3e-07%  1.85e-08   5.79e-09   25
unclassified         133285     3e-03%  1.13e-09   9.52e-13   29
TOTAL                4261302207 100.00%                       
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    4989.7  -0.23x  6.81x   3102.8 -0.23x  -0.44x
ev/clk/SM 10.92   -0.23x  6.81x   10.67  -0.23x  -0.44x
clk/ev    21.1    -0.61x  5.19x   22.0   -0.61x  1.02x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      8
fp64      0
other     1
**total** **9**
========= =====

**Special Values Table:**

========= ==== ======= ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ==== ====
**a\b**   -INF -maxN   -1       -minN    -maxD    -minD    -0       +0       +minD    +maxD    +minN    +1       +maxN    +INF QNAN
========= ==== ======= ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ==== ====
**-INF**  nan  nan     nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan  nan
**-maxN** nan  +0      -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 nan      nan  nan
**-1**    nan  3.4e+38 +0       -1       -1       -1       -1       -1       -1       -1       -1       -2       -3.4e+38 nan  nan
**-minN** nan  3.4e+38 1        +0       -1.4e-45 -1.2e-38 -1.2e-38 -1.2e-38 -1.2e-38 -2.4e-38 -2.4e-38 -1       -3.4e+38 nan  nan
**-maxD** nan  3.4e+38 1        1.4e-45  +0       -1.2e-38 -1.2e-38 -1.2e-38 -1.2e-38 -2.4e-38 -2.4e-38 -1       -3.4e+38 nan  nan
**-minD** nan  3.4e+38 1        1.2e-38  1.2e-38  +0       -1.4e-45 -1.4e-45 -2.8e-45 -1.2e-38 -1.2e-38 -1       -3.4e+38 nan  nan
**-0**    nan  3.4e+38 1        1.2e-38  1.2e-38  1.4e-45  +0       +0       -1.4e-45 -1.2e-38 -1.2e-38 -1       -3.4e+38 nan  nan
**+0**    nan  3.4e+38 1        1.2e-38  1.2e-38  1.4e-45  +0       +0       -1.4e-45 -1.2e-38 -1.2e-38 -1       -3.4e+38 nan  nan
**+minD** nan  3.4e+38 1        1.2e-38  1.2e-38  2.8e-45  1.4e-45  1.4e-45  +0       -1.2e-38 -1.2e-38 -1       -3.4e+38 nan  nan
**+maxD** nan  3.4e+38 1        2.4e-38  2.4e-38  1.2e-38  1.2e-38  1.2e-38  1.2e-38  +0       -1.4e-45 -1       -3.4e+38 nan  nan
**+minN** nan  3.4e+38 1        2.4e-38  2.4e-38  1.2e-38  1.2e-38  1.2e-38  1.2e-38  1.4e-45  +0       -1       -3.4e+38 nan  nan
**+1**    nan  3.4e+38 2        1        1        1        1        1        1        1        1        +0       -3.4e+38 nan  nan
**+maxN** nan  nan     3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  +0       nan  nan
**+INF**  nan  nan     nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan  nan
**QNAN**  nan  nan     nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan  nan
========= ==== ======= ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============ ======== ======= ========== ========== ====
Class        Count    Percent Max RelErr Avg RelErr Bits
============ ======== ======= ========== ========== ====
normal (OK)  16760814 100.00% 8.49e-29   -2.46e-36  93
unclassified 1        6e-06%  1.45e-28   1.45e-28   92
TOTAL        16760815 100.00%                       
============ ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    66.8    -0.09x  -0.53x   1264.9 -0.18x  10.88x
ev/clk/SM 0.15    -0.09x  -0.53x   4.35   -0.18x  10.88x
clk/ev    769.3   -0.14x  -0.53x   80.3   -0.28x  5.03x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      11
other     0
**total** **11**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760815 100.00% 2.47e-32   -8.39e-38  105
TOTAL       16760815 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    36.8    -0.05x  -0.29x   766.1 -0.11x  6.53x
ev/clk/SM 0.08    -0.05x  -0.29x   2.63  -0.11x  6.53x
clk/ev    1403.4  -0.08x  -0.29x   114.8 -0.20x  3.52x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      20
other     0
**total** **20**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     16760814 100.00% 8.49e-29   -2.46e-36  93
output special  2        1e-05%  --         --         --
output denormal 7        4e-05%  0.00e+00   0.00e+00   0
unclassified    1        6e-06%  1.45e-28   1.45e-28   92
TOTAL           16760824 100.00%                       
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    92.0    -0.13x  -0.74x   1674.7 -0.24x  14.26x
ev/clk/SM 0.20    -0.13x  -0.74x   5.76   -0.24x  14.26x
clk/ev    586.0   -0.19x  -0.70x   42.2   -0.53x  9.57x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      8
other     4
**total** **12**
========= ======

**Special Values Table:**

========= ==== ======== ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ==== ====
**a\b**   -INF -maxN    -1        -minN     -maxD     -minD     -0        +0        +minD     +maxD     +minN     +1        +maxN     +INF QNAN
========= ==== ======== ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ==== ====
**-INF**  nan  nan      nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan  nan
**-maxN** nan  +0       -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 nan       nan  nan
**-1**    nan  1.8e+308 +0        -1        -1        -1        -1        -1        -1        -1        -1        -2        -1.8e+308 nan  nan
**-minN** nan  1.8e+308 1         +0        -4.9e-324 -2.2e-308 -2.2e-308 -2.2e-308 -2.2e-308 -4.5e-308 -4.5e-308 -1        -1.8e+308 nan  nan
**-maxD** nan  1.8e+308 1         4.9e-324  +0        -2.2e-308 -2.2e-308 -2.2e-308 -2.2e-308 -4.5e-308 -4.5e-308 -1        -1.8e+308 nan  nan
**-minD** nan  1.8e+308 1         2.2e-308  2.2e-308  +0        -4.9e-324 -4.9e-324 -9.9e-324 -2.2e-308 -2.2e-308 -1        -1.8e+308 nan  nan
**-0**    nan  1.8e+308 1         2.2e-308  2.2e-308  4.9e-324  +0        +0        -4.9e-324 -2.2e-308 -2.2e-308 -1        -1.8e+308 nan  nan
**+0**    nan  1.8e+308 1         2.2e-308  2.2e-308  4.9e-324  +0        +0        -4.9e-324 -2.2e-308 -2.2e-308 -1        -1.8e+308 nan  nan
**+minD** nan  1.8e+308 1         2.2e-308  2.2e-308  9.9e-324  4.9e-324  4.9e-324  +0        -2.2e-308 -2.2e-308 -1        -1.8e+308 nan  nan
**+maxD** nan  1.8e+308 1         4.5e-308  4.5e-308  2.2e-308  2.2e-308  2.2e-308  2.2e-308  +0        -4.9e-324 -1        -1.8e+308 nan  nan
**+minN** nan  1.8e+308 1         4.5e-308  4.5e-308  2.2e-308  2.2e-308  2.2e-308  2.2e-308  4.9e-324  +0        -1        -1.8e+308 nan  nan
**+1**    nan  1.8e+308 2         1         1         1         1         1         1         1         1         +0        -1.8e+308 nan  nan
**+maxN** nan  nan      1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  +0        nan  nan
**+INF**  nan  nan      nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan  nan
**QNAN**  nan  nan      nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan  nan
========= ==== ======== ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ==== ====

--------------

Multiplication (mul)
--------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3127081985 97.99%  1.00e-13   1.68e-15   43
output special       1          3e-08%  --         --         --
output denormal      3569762    0.11%   3.33e-01   1.68e-06   1
input denormal       22277484   0.70%   1.19e-07   9.31e-09   23
output near denormal 1930512    0.06%   1.19e-07   8.44e-08   23
cancellation         36315300   1.14%   5.96e-08   4.23e-09   24
TOTAL                3191175044 100.00%                       
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    4450.9  -0.21x  6.13x   2773.2 -0.21x  -0.40x
ev/clk/SM 9.74    -0.21x  6.13x   9.54   -0.21x  -0.40x
clk/ev    34.7    -0.38x  3.15x   35.9   -0.38x  -0.63x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      9
fp64      0
other     0
**total** **9**
========= =====

*Method: ``accurate``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3127081985 97.99%  1.00e-13   1.68e-15   43
output special       1          3e-08%  --         --         --
output denormal      3569762    0.11%   3.33e-01   1.68e-06   1
input denormal       22277484   0.70%   1.19e-07   9.31e-09   23
output near denormal 1930512    0.06%   1.19e-07   8.44e-08   23
cancellation         36315300   1.14%   5.96e-08   4.23e-09   24
TOTAL                3191175044 100.00%                       
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    4426.0  -0.21x  6.05x   2771.9 -0.21x  -0.40x
ev/clk/SM 9.69    -0.21x  6.05x   9.53   -0.21x  -0.40x
clk/ev    34.3    -0.38x  3.19x   36.2   -0.37x  -0.62x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      9
fp64      0
other     0
**total** **9**
========= =====

*Method: ``fast``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3127081961 83.86%  1.00e-13   1.31e-15   43
output special       537824645  14.42%  --         --         --
input special        5          1e-07%  --         --         --
output denormal      3569762    0.10%   3.33e-01   1.68e-06   1
input denormal       22277508   0.60%   1.19e-07   9.31e-09   23
output near denormal 1930512    0.05%   1.19e-07   8.44e-08   23
cancellation         36315300   0.97%   5.96e-08   4.23e-09   24
TOTAL                3728999693 100.00%                       
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    6083.1  -0.29x  8.33x   3624.0 -0.27x  -0.52x
ev/clk/SM 13.32   -0.29x  8.33x   12.46  -0.27x  -0.52x
clk/ev    21.8    -0.61x  4.99x   22.4   -0.61x  1.00x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      5
fp64      0
other     1
**total** **6**
========= =====

**Special Values Table:**

========= ==== ======== ======== ======== ======== ======== === === ======== ======== ======== ======== ======== ==== ====
**a\b**   -INF -maxN    -1       -minN    -maxD    -minD    -0  +0  +minD    +maxD    +minN    +1       +maxN    +INF QNAN
========= ==== ======== ======== ======== ======== ======== === === ======== ======== ======== ======== ======== ==== ====
**-INF**  nan  nan      nan      nan      nan      nan      nan nan nan      nan      nan      nan      nan      nan  nan
**-maxN** nan  nan      3.4e+38  4        4        4.8e-07  +0  +0  -4.8e-07 -4       -4       -3.4e+38 nan      nan  nan
**-1**    nan  3.4e+38  1        1.2e-38  1.2e-38  1.4e-45  +0  +0  -1.4e-45 -1.2e-38 -1.2e-38 -1       -3.4e+38 nan  nan
**-minN** nan  4        1.2e-38  +0       +0       +0       +0  +0  +0       +0       +0       -1.2e-38 -4       nan  nan
**-maxD** nan  4        1.2e-38  +0       +0       +0       +0  +0  +0       +0       +0       -1.2e-38 -4       nan  nan
**-minD** nan  4.8e-07  1.4e-45  +0       +0       +0       +0  +0  +0       +0       +0       -1.4e-45 -4.8e-07 nan  nan
**-0**    nan  +0       +0       +0       +0       +0       +0  +0  +0       +0       +0       +0       +0       nan  nan
**+0**    nan  +0       +0       +0       +0       +0       +0  +0  +0       +0       +0       +0       +0       nan  nan
**+minD** nan  -4.8e-07 -1.4e-45 +0       +0       +0       +0  +0  +0       +0       +0       1.4e-45  4.8e-07  nan  nan
**+maxD** nan  -4       -1.2e-38 +0       +0       +0       +0  +0  +0       +0       +0       1.2e-38  4        nan  nan
**+minN** nan  -4       -1.2e-38 +0       +0       +0       +0  +0  +0       +0       +0       1.2e-38  4        nan  nan
**+1**    nan  -3.4e+38 -1       -1.2e-38 -1.2e-38 -1.4e-45 +0  +0  1.4e-45  1.2e-38  1.2e-38  1        3.4e+38  nan  nan
**+maxN** nan  nan      -3.4e+38 -4       -4       -4.8e-07 +0  +0  4.8e-07  4        4        3.4e+38  nan      nan  nan
**+INF**  nan  nan      nan      nan      nan      nan      nan nan nan      nan      nan      nan      nan      nan  nan
**QNAN**  nan  nan      nan      nan      nan      nan      nan nan nan      nan      nan      nan      nan      nan  nan
========= ==== ======== ======== ======== ======== ======== === === ======== ======== ======== ======== ======== ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12537828 99.76%  9.99e-29   -2.40e-20  93
output denormal      1757     0.01%   2.05e-13   2.49e-16   42
input denormal       2949     0.02%   2.19e-16   1.10e-17   52
output near denormal 772      6e-03%  2.22e-16   1.97e-16   52
cancellation         24953    0.20%   1.11e-16   4.57e-18   53
TOTAL                12568259 100.00%                       
==================== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    82.7    -0.11x  -0.42x   1452.7 -0.21x  11.36x
ev/clk/SM 0.18    -0.11x  -0.42x   5.00   -0.21x  11.36x
clk/ev    651.2   -0.17x  -0.33x   68.9   -0.32x  3.04x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      9
other     0
**total** **9**
========= =====

*Method: ``accurate``*

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12537828 99.76%  9.99e-29   -2.40e-20  93
output denormal      1757     0.01%   2.05e-13   2.49e-16   42
input denormal       2949     0.02%   2.19e-16   1.10e-17   52
output near denormal 772      6e-03%  2.22e-16   1.97e-16   52
cancellation         24953    0.20%   1.11e-16   4.57e-18   53
TOTAL                12568259 100.00%                       
==================== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    82.7    -0.11x  -0.42x   1453.2 -0.21x  11.35x
ev/clk/SM 0.18    -0.11x  -0.42x   5.00   -0.21x  11.35x
clk/ev    651.2   -0.17x  -0.33x   68.7   -0.32x  3.04x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      9
other     0
**total** **9**
========= =====

*Method: ``fast``*

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12537828 85.50%  9.99e-29   -2.40e-20  93
output special       2095104  14.29%  --         --         --
output denormal      1757     0.01%   2.05e-13   2.49e-16   42
input denormal       2949     0.02%   2.19e-16   1.10e-17   52
output near denormal 772      5e-03%  2.22e-16   1.97e-16   52
cancellation         24953    0.17%   1.11e-16   4.57e-18   53
TOTAL                14663363 100.00%                       
==================== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    156.7   -0.21x  -0.80x   1978.9 -0.29x  15.46x
ev/clk/SM 0.34    -0.21x  -0.80x   6.80   -0.29x  15.46x
clk/ev    393.2   -0.28x  -0.55x   46.0   -0.49x  4.54x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     2
**total** **7**
========= =====

**Special Values Table:**

========= ==== ========= ========= ========= ========= ========= === === ========= ========= ========= ========= ========= ==== ====
**a\b**   -INF -maxN     -1        -minN     -maxD     -minD     -0  +0  +minD     +maxD     +minN     +1        +maxN     +INF QNAN
========= ==== ========= ========= ========= ========= ========= === === ========= ========= ========= ========= ========= ==== ====
**-INF**  nan  nan       nan       nan       nan       nan       nan nan nan       nan       nan       nan       nan       nan  nan
**-maxN** nan  nan       1.8e+308  4         4         8.9e-16   +0  +0  -8.9e-16  -4        -4        -1.8e+308 nan       nan  nan
**-1**    nan  1.8e+308  1         2.2e-308  2.2e-308  4.9e-324  +0  +0  -4.9e-324 -2.2e-308 -2.2e-308 -1        -1.8e+308 nan  nan
**-minN** nan  4         2.2e-308  +0        +0        +0        +0  +0  +0        +0        +0        -2.2e-308 -4        nan  nan
**-maxD** nan  4         2.2e-308  +0        +0        +0        +0  +0  +0        +0        +0        -2.2e-308 -4        nan  nan
**-minD** nan  8.9e-16   4.9e-324  +0        +0        +0        +0  +0  +0        +0        +0        -4.9e-324 -8.9e-16  nan  nan
**-0**    nan  +0        +0        +0        +0        +0        +0  +0  +0        +0        +0        +0        +0        nan  nan
**+0**    nan  +0        +0        +0        +0        +0        +0  +0  +0        +0        +0        +0        +0        nan  nan
**+minD** nan  -8.9e-16  -4.9e-324 +0        +0        +0        +0  +0  +0        +0        +0        4.9e-324  8.9e-16   nan  nan
**+maxD** nan  -4        -2.2e-308 +0        +0        +0        +0  +0  +0        +0        +0        2.2e-308  4         nan  nan
**+minN** nan  -4        -2.2e-308 +0        +0        +0        +0  +0  +0        +0        +0        2.2e-308  4         nan  nan
**+1**    nan  -1.8e+308 -1        -2.2e-308 -2.2e-308 -4.9e-324 +0  +0  4.9e-324  2.2e-308  2.2e-308  1         1.8e+308  nan  nan
**+maxN** nan  nan       -1.8e+308 -4        -4        -8.9e-16  +0  +0  8.9e-16   4         4         1.8e+308  nan       nan  nan
**+INF**  nan  nan       nan       nan       nan       nan       nan nan nan       nan       nan       nan       nan       nan  nan
**QNAN**  nan  nan       nan       nan       nan       nan       nan nan nan       nan       nan       nan       nan       nan  nan
========= ==== ========= ========= ========= ========= ========= === === ========= ========= ========= ========= ========= ==== ====

--------------

Division (div)
--------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3006485741 94.22%  1.00e-13   2.08e-15   43
output special  8355823    0.26%   --         --         --
input special   1          3e-08%  --         --         --
output denormal 19879171   0.62%   1.00e+00   9.96e-01   0
input denormal  155859363  4.88%   1.82e-07   5.23e-09   22
cancellation    192572     6e-03%  1.26e-08   1.11e-12   26
TOTAL           3190772671 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2808.8  -0.86x  27.54x  1791.0 -0.93x  1.52x
ev/clk/SM 6.15    -0.86x  27.54x  6.16   -0.93x  1.52x
clk/ev    50.5    1.21x   11.56x  52.9   1.20x   2.63x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3006485741 94.22%  1.00e-13   2.08e-15   43
output special  8355823    0.26%   --         --         --
input special   1          3e-08%  --         --         --
output denormal 19879171   0.62%   1.00e+00   9.96e-01   0
input denormal  155859363  4.88%   1.82e-07   5.23e-09   22
cancellation    192572     6e-03%  1.26e-08   1.11e-12   26
TOTAL           3190772671 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2810.0  -0.86x  27.55x  1790.3 -0.93x  1.52x
ev/clk/SM 6.15    -0.86x  27.55x  6.16   -0.93x  1.52x
clk/ev    50.5    1.21x   11.57x  52.6   1.21x   2.62x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          1716918527 45.14%  1.00e-13   3.40e-15   43
output special       1178697562 30.99%  --         --         --
input special        5          1e-07%  --         --         --
output denormal      109423287  2.88%   2.00e+00   8.79e-01   0
input denormal       11094134   0.29%   1.19e-07   9.02e-09   23
output near denormal 5057161    0.13%   1.00e+00   6.67e-01   0
input near inf       9408139    0.25%   1.00e+00   7.17e-01   0
cancellation         773236097  20.33%  1.00e+00   6.61e-01   0
TOTAL                3803834912 100.00%                       
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    5070.2  1.56x   49.69x  3317.5 1.73x   2.82x
ev/clk/SM 11.10   1.56x   49.69x  11.41  1.73x   2.82x
clk/ev    21.2    2.89x   27.79x  25.4   2.51x   5.43x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      21
fp64      0
other     1
**total** **22**
========= ======

**Special Values Table:**

========= ==== ======== ======== ======== ======== ===== === === ===== ======== ======== ======== ======== ==== ====
**a\b**   -INF -maxN    -1       -minN    -maxD    -minD -0  +0  +minD +maxD    +minN    +1       +maxN    +INF QNAN
========= ==== ======== ======== ======== ======== ===== === === ===== ======== ======== ======== ======== ==== ====
**-INF**  nan  nan      nan      nan      nan      nan   nan nan nan   nan      nan      nan      nan      nan  nan
**-maxN** nan  1        3.4e+38  nan      nan      nan   nan nan nan   nan      nan      -3.4e+38 -1       nan  nan
**-1**    nan  2.9e-39  1        8.5e+37  8.5e+37  nan   nan nan nan   -8.5e+37 -8.5e+37 -1       -2.9e-39 nan  nan
**-minN** nan  +0       1.2e-38  1        1        nan   nan nan nan   -1       -1       -1.2e-38 -0       nan  nan
**-maxD** nan  +0       1.2e-38  1        1        nan   nan nan nan   -1       -1       -1.2e-38 -0       nan  nan
**-minD** nan  +0       1.4e-45  1.2e-07  1.2e-07  nan   nan nan nan   -1.2e-07 -1.2e-07 -1.4e-45 -0       nan  nan
**-0**    nan  +0       +0       +0       +0       nan   nan nan nan   +0       +0       +0       +0       nan  nan
**+0**    nan  -0       -0       -0       -0       nan   nan nan nan   +0       +0       +0       +0       nan  nan
**+minD** nan  -0       -1.4e-45 -1.2e-07 -1.2e-07 nan   nan nan nan   1.2e-07  1.2e-07  1.4e-45  +0       nan  nan
**+maxD** nan  -0       -1.2e-38 -1       -1       nan   nan nan nan   1        1        1.2e-38  +0       nan  nan
**+minN** nan  -0       -1.2e-38 -1       -1       nan   nan nan nan   1        1        1.2e-38  +0       nan  nan
**+1**    nan  -2.9e-39 -1       -8.5e+37 -8.5e+37 nan   nan nan nan   8.5e+37  8.5e+37  1        2.9e-39  nan  nan
**+maxN** nan  -1       -3.4e+38 nan      nan      nan   nan nan nan   nan      nan      3.4e+38  1        nan  nan
**+INF**  nan  nan      nan      nan      nan      nan   nan nan nan   nan      nan      nan      nan      nan  nan
**QNAN**  nan  nan      nan      nan      nan      nan   nan nan nan   nan      nan      nan      nan      nan  nan
========= ==== ======== ======== ======== ======== ===== === === ===== ======== ======== ======== ======== ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     12480681 99.32%  1.00e-28   -3.89e-20  93
output special  4083     0.03%   --         --         --
output denormal 8        6e-05%  1.86e-15   3.27e-16   48
input denormal  81772    0.65%   2.32e-16   5.87e-18   51
TOTAL           12566544 100.00%                       
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    65.5    -0.62x  -0.71x   1004.2 -0.86x  17.42x
ev/clk/SM 0.14    -0.62x  -0.71x   3.45   -0.86x  17.42x
clk/ev    832.2   -0.68x  -0.62x   106.5  1.30x   4.87x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      1
fp64      18
other     25
**total** **44**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     12480681 99.32%  1.00e-28   -3.89e-20  93
output special  4083     0.03%   --         --         --
output denormal 8        6e-05%  1.86e-15   3.27e-16   48
input denormal  81772    0.65%   2.32e-16   5.87e-18   51
TOTAL           12566544 100.00%                       
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    65.4    -0.62x  -0.71x   1004.5 -0.86x  17.44x
ev/clk/SM 0.14    -0.62x  -0.71x   3.45   -0.86x  17.44x
clk/ev    832.4   -0.68x  -0.62x   106.5  1.29x   4.86x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      1
fp64      18
other     25
**total** **44**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          7341903  49.73%  9.99e-29   -1.50e-04  93
output special       4704220  31.86%  --         --         --
output denormal      106277   0.72%   1.25e+00   9.42e-01   0
input denormal       1390     9e-03%  2.22e-16   9.41e-18   52
output near denormal 2287     0.02%   1.00e+00   8.29e-01   0
input near inf       12236    0.08%   1.00e+00   9.86e-01   0
cancellation         2595339  17.58%  1.00e+00   9.43e-01   0
TOTAL                14763652 100.00%                       
==================== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    167.5   1.59x   1.83x    1760.3 1.50x   30.61x
ev/clk/SM 0.37    1.59x   1.83x    6.05   1.50x   30.61x
clk/ev    383.4   1.52x   1.34x    46.5   2.96x   11.13x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      1
fp64      26
other     25
**total** **52**
========= ======

**Special Values Table:**

========= ==== ========= ========= ========= ========= ===== === === ===== ========= ========= ========= ========= ==== ====
**a\b**   -INF -maxN     -1        -minN     -maxD     -minD -0  +0  +minD +maxD     +minN     +1        +maxN     +INF QNAN
========= ==== ========= ========= ========= ========= ===== === === ===== ========= ========= ========= ========= ==== ====
**-INF**  nan  nan       nan       nan       nan       nan   nan nan nan   nan       nan       nan       nan       nan  nan
**-maxN** nan  1         1.8e+308  nan       nan       nan   nan nan nan   nan       nan       -1.8e+308 -1        nan  nan
**-1**    nan  5.6e-309  1         4.5e+307  4.5e+307  nan   nan nan nan   -4.5e+307 -4.5e+307 -1        -5.6e-309 nan  nan
**-minN** nan  +0        2.2e-308  1         1         nan   nan nan nan   -1        -1        -2.2e-308 -0        nan  nan
**-maxD** nan  +0        2.2e-308  1         1         nan   nan nan nan   -1        -1        -2.2e-308 -0        nan  nan
**-minD** nan  +0        4.9e-324  2.2e-16   2.2e-16   nan   nan nan nan   -2.2e-16  -2.2e-16  -4.9e-324 -0        nan  nan
**-0**    nan  +0        +0        +0        +0        nan   nan nan nan   +0        +0        +0        +0        nan  nan
**+0**    nan  -0        -0        -0        -0        nan   nan nan nan   +0        +0        +0        +0        nan  nan
**+minD** nan  -0        -4.9e-324 -2.2e-16  -2.2e-16  nan   nan nan nan   2.2e-16   2.2e-16   4.9e-324  +0        nan  nan
**+maxD** nan  -0        -2.2e-308 -1        -1        nan   nan nan nan   1         1         2.2e-308  +0        nan  nan
**+minN** nan  -0        -2.2e-308 -1        -1        nan   nan nan nan   1         1         2.2e-308  +0        nan  nan
**+1**    nan  -5.6e-309 -1        -4.5e+307 -4.5e+307 nan   nan nan nan   4.5e+307  4.5e+307  1         5.6e-309  nan  nan
**+maxN** nan  -1        -1.8e+308 nan       nan       nan   nan nan nan   nan       nan       1.8e+308  1         nan  nan
**+INF**  nan  nan       nan       nan       nan       nan   nan nan nan   nan       nan       nan       nan       nan  nan
**QNAN**  nan  nan       nan       nan       nan       nan   nan nan nan   nan       nan       nan       nan       nan  nan
========= ==== ========= ========= ========= ========= ===== === === ===== ========= ========= ========= ========= ==== ====

--------------

Accumulate (acc)
----------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156129 100.00% 7.10e-15   5.63e-17   47
TOTAL       4261156129 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    4321.1  -0.21x  5.89x   2688.0 -0.20x  -0.38x
ev/clk/SM 9.46    -0.21x  5.89x   9.24   -0.20x  -0.38x
clk/ev    39.2    -0.33x  2.79x   41.0   -0.34x  -0.55x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      10
fp64      0
other     0
**total** **10**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156129 100.00% 7.10e-15   5.63e-17   47
TOTAL       4261156129 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    3317.9  -0.16x  4.53x   2085.6 -0.16x  -0.30x
ev/clk/SM 7.26    -0.16x  4.53x   7.17   -0.16x  -0.30x
clk/ev    51.6    -0.25x  2.15x   54.0   -0.26x  -0.42x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     4261156129 100.00% 7.10e-15   5.63e-17   47
output special  130965     3e-03%  --         --         --
input special   5          1e-07%  --         --         --
output denormal 16604      4e-04%  0.00e+00   0.00e+00   0
TOTAL           4261303703 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    5207.5  -0.25x  7.11x   3253.6 -0.25x  -0.47x
ev/clk/SM 11.40   -0.25x  7.11x   11.19  -0.25x  -0.47x
clk/ev    17.7    -0.76x  6.22x   18.3   -0.74x  1.23x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      7
fp64      0
other     1
**total** **8**
========= =====

**Special Values Table:**

========= ==== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======= ==== ====
**a\b**   -INF -maxN    -1       -minN    -maxD    -minD    -0       +0       +minD    +maxD    +minN    +1       +maxN   +INF QNAN
========= ==== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======= ==== ====
**-INF**  nan  nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan     nan  nan
**-maxN** nan  nan      -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 -3.4e+38 +0      nan  nan
**-1**    nan  -3.4e+38 -2       -1       -1       -1       -1       -1       -1       -1       -1       +0       3.4e+38 nan  nan
**-minN** nan  -3.4e+38 -1       -2.4e-38 -2.4e-38 -1.2e-38 -1.2e-38 -1.2e-38 -1.2e-38 -1.4e-45 +0       1        3.4e+38 nan  nan
**-maxD** nan  -3.4e+38 -1       -2.4e-38 -2.4e-38 -1.2e-38 -1.2e-38 -1.2e-38 -1.2e-38 +0       1.4e-45  1        3.4e+38 nan  nan
**-minD** nan  -3.4e+38 -1       -1.2e-38 -1.2e-38 -2.8e-45 -1.4e-45 -1.4e-45 +0       1.2e-38  1.2e-38  1        3.4e+38 nan  nan
**-0**    nan  -3.4e+38 -1       -1.2e-38 -1.2e-38 -1.4e-45 +0       +0       1.4e-45  1.2e-38  1.2e-38  1        3.4e+38 nan  nan
**+0**    nan  -3.4e+38 -1       -1.2e-38 -1.2e-38 -1.4e-45 +0       +0       1.4e-45  1.2e-38  1.2e-38  1        3.4e+38 nan  nan
**+minD** nan  -3.4e+38 -1       -1.2e-38 -1.2e-38 +0       1.4e-45  1.4e-45  2.8e-45  1.2e-38  1.2e-38  1        3.4e+38 nan  nan
**+maxD** nan  -3.4e+38 -1       -1.4e-45 +0       1.2e-38  1.2e-38  1.2e-38  1.2e-38  2.4e-38  2.4e-38  1        3.4e+38 nan  nan
**+minN** nan  -3.4e+38 -1       +0       1.4e-45  1.2e-38  1.2e-38  1.2e-38  1.2e-38  2.4e-38  2.4e-38  1        3.4e+38 nan  nan
**+1**    nan  -3.4e+38 +0       1        1        1        1        1        1        1        1        2        3.4e+38 nan  nan
**+maxN** nan  +0       3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  3.4e+38  nan     nan  nan
**+INF**  nan  nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan     nan  nan
**QNAN**  nan  nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan      nan     nan  nan
========= ==== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======== ======= ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760814 100.00% 1.32e-32   -3.35e-38  105
TOTAL       16760814 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    73.5    -0.10x  -0.43x   1362.7 -0.20x  7.66x
ev/clk/SM 0.16    -0.10x  -0.43x   4.69   -0.20x  7.66x
clk/ev    713.2   -0.15x  -0.25x   80.6   -0.28x  2.15x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      10
other     0
**total** **10**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760814 100.00% 1.32e-32   -3.35e-38  105
TOTAL       16760814 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    56.6    -0.08x  -0.33x   1101.2 -0.16x  6.19x
ev/clk/SM 0.12    -0.08x  -0.33x   3.79   -0.16x  6.19x
clk/ev    942.2   -0.12x  -0.19x   106.2  -0.21x  1.63x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      13
other     0
**total** **13**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     16760814 100.00% 1.32e-32   -3.35e-38  105
output special  2        1e-05%  --         --         --
output denormal 8        5e-05%  0.00e+00   0.00e+00   0
TOTAL           16760824 100.00%                       
=============== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    105.0   -0.14x  -0.61x   1814.3 -0.27x  10.19x
ev/clk/SM 0.23    -0.14x  -0.61x   6.24   -0.27x  10.19x
clk/ev    512.4   -0.22x  -0.35x   36.9   -0.61x  4.70x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      7
other     2
**total** **9**
========= =====

**Special Values Table:**

========= ==== ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ======== ==== ====
**a\b**   -INF -maxN     -1        -minN     -maxD     -minD     -0        +0        +minD     +maxD     +minN     +1        +maxN    +INF QNAN
========= ==== ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ======== ==== ====
**-INF**  nan  nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan      nan  nan
**-maxN** nan  nan       -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 -1.8e+308 +0       nan  nan
**-1**    nan  -1.8e+308 -2        -1        -1        -1        -1        -1        -1        -1        -1        +0        1.8e+308 nan  nan
**-minN** nan  -1.8e+308 -1        -4.5e-308 -4.5e-308 -2.2e-308 -2.2e-308 -2.2e-308 -2.2e-308 -4.9e-324 +0        1         1.8e+308 nan  nan
**-maxD** nan  -1.8e+308 -1        -4.5e-308 -4.5e-308 -2.2e-308 -2.2e-308 -2.2e-308 -2.2e-308 +0        4.9e-324  1         1.8e+308 nan  nan
**-minD** nan  -1.8e+308 -1        -2.2e-308 -2.2e-308 -9.9e-324 -4.9e-324 -4.9e-324 +0        2.2e-308  2.2e-308  1         1.8e+308 nan  nan
**-0**    nan  -1.8e+308 -1        -2.2e-308 -2.2e-308 -4.9e-324 +0        +0        4.9e-324  2.2e-308  2.2e-308  1         1.8e+308 nan  nan
**+0**    nan  -1.8e+308 -1        -2.2e-308 -2.2e-308 -4.9e-324 +0        +0        4.9e-324  2.2e-308  2.2e-308  1         1.8e+308 nan  nan
**+minD** nan  -1.8e+308 -1        -2.2e-308 -2.2e-308 +0        4.9e-324  4.9e-324  9.9e-324  2.2e-308  2.2e-308  1         1.8e+308 nan  nan
**+maxD** nan  -1.8e+308 -1        -4.9e-324 +0        2.2e-308  2.2e-308  2.2e-308  2.2e-308  4.5e-308  4.5e-308  1         1.8e+308 nan  nan
**+minN** nan  -1.8e+308 -1        +0        4.9e-324  2.2e-308  2.2e-308  2.2e-308  2.2e-308  4.5e-308  4.5e-308  1         1.8e+308 nan  nan
**+1**    nan  -1.8e+308 +0        1         1         1         1         1         1         1         1         2         1.8e+308 nan  nan
**+maxN** nan  +0        1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  1.8e+308  nan      nan  nan
**+INF**  nan  nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan      nan  nan
**QNAN**  nan  nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan       nan      nan  nan
========= ==== ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ========= ======== ==== ====

--------------

Fused Multiply-Add (fma)
------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3705991498 99.85%  1.00e-13   8.17e-16   43
output special  18531      5e-04%  --         --         --
output denormal 42617      1e-03%  1.29e-03   1.12e-06   9
input denormal  5260882    0.14%   1.19e-07   2.00e-09   23
input near inf  4306       1e-04%  3.82e-09   1.89e-12   27
cancellation    174428     5e-03%  1.09e-08   1.27e-12   26
unclassified    74732      2e-03%  2.58e-09   9.39e-13   28
TOTAL           3711566994 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2228.7  -0.11x  3.03x   1394.2 -0.11x  -0.20x
ev/clk/SM 4.88    -0.11x  3.03x   4.79   -0.11x  -0.20x
clk/ev    66.7    -0.20x  1.71x   68.8   -0.20x  -0.34x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      19
fp64      0
other     0
**total** **19**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3706072849 99.85%  1.00e-13   5.02e-16   43
output special  18531      5e-04%  --         --         --
output denormal 42617      1e-03%  1.29e-03   1.12e-06   9
input denormal  5251287    0.14%   1.19e-07   2.00e-09   23
input near inf  3100       8e-05%  1.22e-09   1.36e-12   29
cancellation    125702     3e-03%  1.22e-08   1.32e-12   26
unclassified    52908      1e-03%  3.70e-09   1.00e-12   28
TOTAL           3711566994 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    1205.6  -0.06x  1.65x   747.6 -0.06x  -0.11x
ev/clk/SM 2.64    -0.06x  1.65x   2.57  -0.06x  -0.11x
clk/ev    93.1    -0.14x  1.25x   94.6  -0.15x  -0.24x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      37
fp64      0
other     0
**total** **37**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3705991498 87.35%  1.00e-13   8.17e-16   43
output special  531120823  12.52%  --         --         --
input special   41         1e-06%  --         --         --
output denormal 64932      2e-03%  1.29e-03   7.37e-07   9
input denormal  5260882    0.12%   1.19e-07   2.00e-09   23
input near inf  4306       1e-04%  3.82e-09   1.89e-12   27
cancellation    174428     4e-03%  1.09e-08   1.27e-12   26
unclassified    74732      2e-03%  2.58e-09   9.39e-13   28
TOTAL           4242691642 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2567.7  -0.12x  3.51x   1602.5 -0.12x  -0.23x
ev/clk/SM 5.62    -0.12x  3.51x   5.51   -0.12x  -0.23x
clk/ev    33.5    -0.39x  3.42x   34.9   -0.40x  -0.66x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      16
fp64      0
other     1
**total** **17**
========= ======


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    39.6    -0.05x  -0.91x   772.5 -0.11x  18.24x
ev/clk/SM 0.09    -0.05x  -0.91x   2.66  -0.11x  18.24x
clk/ev    1337.6  -0.08x  -0.95x   133.4 -0.17x  8.52x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      19
other     0
**total** **19**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    20.1    -0.03x  -0.46x   432.6 -0.06x  10.25x
ev/clk/SM 0.04    -0.03x  -0.46x   1.49  -0.06x  10.25x
clk/ev    2517.5  -0.05x  -0.51x   215.6 -0.11x  5.27x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      37
other     0
**total** **37**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    47.4    -0.07x  1.09x    859.8 -0.13x  20.30x
ev/clk/SM 0.10    -0.07x  1.09x    2.96  -0.13x  20.30x
clk/ev    1124.3  -0.10x  1.13x    66.9  -0.34x  16.99x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      16
other     2
**total** **18**
========= ======

--------------

Multiply-Add (mad)
------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3705975209 99.85%  1.00e-13   7.88e-16   43
output special  18531      5e-04%  --         --         --
output denormal 42617      1e-03%  1.29e-03   1.12e-06   9
input denormal  5263349    0.14%   1.19e-07   1.99e-09   23
input near inf  4527       1e-04%  1.38e-09   1.31e-12   29
cancellation    184095     5e-03%  1.22e-08   1.30e-12   26
unclassified    78666      2e-03%  3.70e-09   9.60e-13   28
TOTAL           3711566994 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2717.2  -0.13x  3.71x   1589.9 -0.12x  -0.23x
ev/clk/SM 5.95    -0.13x  3.71x   5.47   -0.12x  -0.23x
clk/ev    47.5    -0.28x  2.44x   48.6   -0.28x  -0.47x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      16
fp64      0
other     0
**total** **16**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3705962819 99.85%  1.00e-13   9.67e-16   43
output special  18531      5e-04%  --         --         --
output denormal 42617      1e-03%  1.29e-03   1.12e-06   9
input denormal  5263622    0.14%   1.19e-07   1.99e-09   23
input near inf  4776       1e-04%  1.38e-09   1.31e-12   29
cancellation    192586     5e-03%  7.89e-09   1.28e-12   26
unclassified    82043      2e-03%  2.83e-09   9.66e-13   28
TOTAL           3711566994 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    1587.7  -0.08x  2.17x   997.7 -0.08x  -0.14x
ev/clk/SM 3.48    -0.08x  2.17x   3.43  -0.08x  -0.14x
clk/ev    80.7    -0.16x  1.41x   86.1  -0.16x  -0.27x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      29
fp64      0
other     0
**total** **29**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3705975209 87.35%  1.00e-13   7.88e-16   43
output special  531120823  12.52%  --         --         --
input special   41         1e-06%  --         --         --
output denormal 68114      2e-03%  1.29e-03   7.03e-07   9
input denormal  5263349    0.12%   1.19e-07   1.99e-09   23
input near inf  4527       1e-04%  1.38e-09   1.31e-12   29
cancellation    184095     4e-03%  1.22e-08   1.30e-12   26
unclassified    78666      2e-03%  3.70e-09   9.60e-13   28
TOTAL           4242694824 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    3074.6  -0.15x  4.19x   1947.0 -0.15x  -0.28x
ev/clk/SM 6.73    -0.15x  4.19x   6.69   -0.15x  -0.28x
clk/ev    31.8    -0.41x  3.62x   33.0   -0.41x  -0.69x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    46.9    -0.06x  1.08x    899.1 -0.13x  21.21x
ev/clk/SM 0.10    -0.06x  1.08x    3.09  -0.13x  21.21x
clk/ev    1114.5  -0.10x  1.13x    91.1  -0.25x  12.48x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      16
other     0
**total** **16**
========= ======

*Method: ``accurate``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    25.5    -0.03x  -0.59x   539.8 -0.08x  12.76x
ev/clk/SM 0.06    -0.03x  -0.59x   1.86  -0.08x  12.76x
clk/ev    1985.3  -0.06x  -0.64x   164.8 -0.14x  6.90x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      29
other     0
**total** **29**
========= ======

*Method: ``fast``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    58.0    -0.08x  1.34x    1053.1 -0.15x  24.93x
ev/clk/SM 0.13    -0.08x  1.34x    3.62   -0.15x  24.93x
clk/ev    908.5   -0.13x  1.39x    63.4   -0.36x  17.93x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      13
other     0
**total** **13**
========= ======

Mathematical Functions
======================

Square Root (sqrt)
------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    1992354374 93.14%  1.00e-13   2.30e-15   43
output special 8388605    0.39%   --         --         --
input denormal 138352058  6.47%   2.98e-08   1.25e-09   25
TOTAL          2139095037 100.00%                       
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2236.9  -0.54x  22.92x  1409.5 -0.54x  1.26x
ev/clk/SM 4.90    -0.54x  22.92x  4.85   -0.54x  1.26x
clk/ev    73.0    -0.91x  8.21x   75.1   -0.92x  1.44x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      18
fp64      0
other     1
**total** **19**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ======= ======= ========== ========== ====
Class          Count   Percent Max RelErr Avg RelErr Bits
============== ======= ======= ========== ========== ====
normal (OK)    8305868 99.06%  1.00e-28   -2.09e-20  93
input denormal 78644   0.94%   1.58e-16   2.27e-18   52
TOTAL          8384512 100.00%                       
============== ======= ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    34.2    -0.34x  -0.25x   570.6 -0.51x  8.08x
ev/clk/SM 0.07    -0.34x  -0.25x   1.96  -0.51x  8.08x
clk/ev    1523.2  -0.40x  -0.51x   181.3 -0.59x  3.84x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      23
other     24
**total** **47**
========= ======

**Special Values Table:**

========= =========
**Input** Value
========= =========
**-INF**  -inf
**-maxN** -1.8e+308
**-1**    -1
**-minN** -2.2e-308
**-maxD** -2.2e-308
**-minD** -4.9e-324
**-0**    -0
**+0**    +0
**+minD** 4.9e-324
**+maxD** 2.2e-308
**+minN** 2.2e-308
**+1**    1
**+maxN** 1.8e+308
**+INF**  +inf
**QNAN**  nan
========= =========

--------------

Reciprocal Square Root (rsqrt)
------------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2130706432 99.61%  5.06e-14   2.36e-15   44
output special 8388605    0.39%   --         --         --
input special  1          5e-08%  --         --         --
TOTAL          2139095038 100.00%                       
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2200.4  -0.36x  14.52x  1417.4 -0.40x  -0.90x
ev/clk/SM 4.82    -0.36x  14.52x  4.87   -0.40x  -0.90x
clk/ev    74.0    -0.61x  4.69x   76.3   -0.60x  1.05x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      17
fp64      0
other     0
**total** **17**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 8384512 100.00% 2.45e-32   -4.81e-33  105
TOTAL       8384512 100.00%                       
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    36.3    -0.23x  -0.58x   593.0 -0.38x  18.65x
ev/clk/SM 0.08    -0.23x  -0.58x   2.04  -0.38x  18.65x
clk/ev    1435.7  -0.24x  1.17x    178.2 -0.45x  9.69x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      22
other     22
**total** **44**
========= ======

**Special Values Table:**

========= =========
**Input** Value
========= =========
**-INF**  -inf
**-maxN** -1.8e+308
**-1**    -1
**-minN** -2.2e-308
**-maxD** -2.2e-308
**-minD** -4.9e-324
**-0**    -0
**+0**    +0
**+minD** 4.9e-324
**+maxD** 2.2e-308
**+minN** 2.2e-308
**+1**    1
**+maxN** 1.8e+308
**+INF**  +inf
**QNAN**  nan
========= =========

--------------

Exponential (exp)
-----------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2233993655 68.53%  1.00e-13   6.19e-16   43
output special  1020169796 31.29%  --         --         --
input special   1          3e-08%  --         --         --
output denormal 2180472    0.07%   1.00e+00   1.00e+00   0
output near inf 66521      2e-03%  3.53e-13   1.73e-13   41
unclassified    3608715    0.11%   2.83e-08   1.94e-13   25
TOTAL           3260019160 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    317.5   -0.06x  6.26x   201.8 -0.07x  -0.24x
ev/clk/SM 0.69    -0.06x  6.26x   0.69  -0.07x  -0.24x
clk/ev    331.8   -0.16x  2.98x   344.8 -0.17x  -0.49x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      136
fp64      0
other     14
**total** **150**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Natural Logarithm (log)
-----------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2139095037 49.80%  3.99e-14   8.06e-16   44
output special 2139095043 49.80%  --         --         --
input special  16777216   0.39%   --         --         --
TOTAL          4294967296 100.00%                       
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    291.1   -0.14x  11.19x  170.6 -0.14x  -0.52x
ev/clk/SM 0.64    -0.14x  11.19x  0.59  -0.14x  -0.52x
clk/ev    350.7   -0.26x  5.64x   366.3 -0.26x  -0.80x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      150
fp64      0
other     20
**total** **170**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Power (pow)
-----------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     1066799648 54.96%  1.00e-13   9.15e-16   43
output special  867368869  44.68%  --         --         --
input special   2          1e-07%  --         --         --
output denormal 1029046    0.05%   1.00e+00   1.00e+00   0
input denormal  647132     0.03%   3.40e-09   2.15e-13   28
output near inf 52356      3e-03%  1.89e-12   2.57e-13   38
input near inf  46089      2e-03%  4.60e-11   1.96e-13   34
cancellation    2725722    0.14%   5.68e-09   2.19e-13   27
unclassified    2458925    0.13%   2.89e-12   1.98e-13   38
TOTAL           1941127789 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    134.5   -0.19x  14.85x  68.0  -0.16x  -0.52x
ev/clk/SM 0.29    -0.19x  14.85x  0.23  -0.16x  -0.52x
clk/ev    830.8   -0.29x  6.92x   859.4 -0.29x  -0.94x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      320
fp64      0
other     69
**total** **389**
========= =======

**Special Values Table:**

========= ==== ===== ======== ===== ===== ===== == == ===== ===== ===== ======== ===== ==== ====
**a\b**   -INF -maxN -1       -minN -maxD -minD -0 +0 +minD +maxD +minN +1       +maxN +INF QNAN
========= ==== ===== ======== ===== ===== ===== == == ===== ===== ===== ======== ===== ==== ====
**-INF**  +0   +0    -0       nan   nan   nan   1  1  nan   nan   nan   -inf     +inf  +inf nan
**-maxN** +0   nan   -0       nan   nan   nan   1  1  nan   nan   nan   -3.4e+38 nan   +inf nan
**-1**    1    1     -1       nan   nan   nan   1  1  nan   nan   nan   -1       1     1    nan
**-minN** +inf nan   -8.5e+37 nan   nan   nan   1  1  nan   nan   nan   -0       nan   +0   nan
**-maxD** +inf nan   -8.5e+37 nan   nan   nan   1  1  nan   nan   nan   -0       nan   +0   nan
**-minD** +inf nan   -inf     nan   nan   nan   1  1  nan   nan   nan   -0       nan   +0   nan
**-0**    +inf +inf  +inf     +inf  +inf  +inf  1  1  +0    +0    +0    +0       +0    +0   nan
**+0**    +inf +inf  +inf     +inf  +inf  +inf  1  1  +0    +0    +0    +0       +0    +0   nan
**+minD** +inf nan   +inf     1     1     1     1  1  1     1     1     +0       nan   +0   nan
**+maxD** +inf nan   8.5e+37  1     1     1     1  1  1     1     1     +0       nan   +0   nan
**+minN** +inf nan   8.5e+37  1     1     1     1  1  1     1     1     +0       nan   +0   nan
**+1**    1    1     1        1     1     1     1  1  1     1     1     1        1     1    1
**+maxN** +0   nan   +0       1     1     1     1  1  1     1     1     3.4e+38  nan   +inf nan
**+INF**  +0   +0    +0       +0    +0    +0    1  1  +inf  +inf  +inf  +inf     +inf  +inf nan
**QNAN**  nan  nan   nan      nan   nan   nan   1  1  nan   nan   nan   nan      nan   nan  nan
========= ==== ===== ======== ===== ===== ===== == == ===== ===== ===== ======== ===== ==== ====

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Cube Root (cbrt)
----------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4278190075 100.00% 3.17e-14   1.37e-15   44
input special 3          7e-08%  --         --         --
TOTAL         4278190078 100.00%                       
============= ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    497.9   -0.17x  8.13x   301.1 -0.17x  -0.52x
ev/clk/SM 1.09    -0.17x  8.13x   1.04  -0.17x  -0.52x
clk/ev    292.6   -0.30x  3.69x   309.7 -0.29x  -0.83x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      75
fp64      0
other     27
**total** **102**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Reciprocal Cube Root (rcbrt)
----------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4278190075 100.00% 1.06e-14   1.19e-15   46
output special 5          1e-07%  --         --         --
TOTAL          4278190080 100.00%                       
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    412.9   -0.23x  9.23x   241.7 -0.22x  -0.52x
ev/clk/SM 0.90    -0.23x  9.23x   0.83  -0.22x  -0.52x
clk/ev    342.6   -0.51x  4.05x   356.5 -0.49x  -0.97x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      95
fp64      0
other     36
**total** **131**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Sine (sin)
----------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4233002838 99.33%  1.00e-13   2.64e-15   43
input near inf 441308     0.01%   1.24e-12   1.44e-13   39
unclassified   27968718   0.66%   1.75e-08   1.45e-13   25
TOTAL          4261412864 100.00%                       
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    202.7   -0.10x  4.66x   108.7 -0.09x  -0.21x
ev/clk/SM 0.44    -0.10x  4.66x   0.37  -0.09x  -0.21x
clk/ev    505.7   -0.21x  2.44x   536.5 -0.22x  -0.52x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      226
fp64      0
other     235
**total** **461**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Cosine (cos)
------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4249101016 99.32%  1.00e-13   2.61e-15   43
input near inf 441836     0.01%   2.72e-12   1.44e-13   38
unclassified   28647228   0.67%   1.99e-09   1.46e-13   28
TOTAL          4278190080 100.00%                       
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    203.1   -0.11x  4.73x   109.0 -0.10x  -0.23x
ev/clk/SM 0.44    -0.11x  4.73x   0.37  -0.10x  -0.23x
clk/ev    505.0   -0.21x  2.49x   536.8 -0.22x  -0.56x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      226
fp64      0
other     235
**total** **461**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Hyperbolic Tangent (tanh)
-------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261412864 100.00% 2.15e-14   4.65e-17   45
TOTAL       4261412864 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    241.1   -0.08x  7.17x   134.2 -0.08x  -0.30x
ev/clk/SM 0.53    -0.08x  7.17x   0.46  -0.08x  -0.30x
clk/ev    334.4   -0.21x  2.72x   352.3 -0.21x  -0.47x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      325
fp64      0
other     29
**total** **354**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Error Function (erf)
--------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     4165995376 97.72%  1.00e-13   5.74e-16   43
output denormal 53642      1e-03%  3.45e-05   2.38e-07   14
input denormal  97326090   2.28%   1.19e-07   1.56e-09   23
TOTAL           4263375108 100.00%                       
=============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    125.7   -0.06x  7.34x   63.6  -0.05x  -0.20x
ev/clk/SM 0.28    -0.06x  7.34x   0.22  -0.05x  -0.20x
clk/ev    666.3   -0.11x  4.49x   697.5 -0.11x  -0.60x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      560
fp64      0
other     22
**total** **582**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Complementary Error Function (erfc)
-----------------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          1465624462 45.36%  1.00e-13   1.28e-14   43
output denormal      35988      1e-03%  1.00e+00   2.12e-02   0
input denormal       436207615  13.50%  1.03e-13   1.03e-13   43
output near denormal 13032      4e-04%  1.19e-07   8.59e-08   23
unclassified         1329021553 41.13%  5.96e-08   9.72e-13   24
TOTAL                3230902650 100.00%                       
==================== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    80.2    -0.07x  6.39x   39.0  -0.06x  -0.22x
ev/clk/SM 0.18    -0.07x  6.39x   0.13  -0.06x  -0.22x
clk/ev    779.5   -0.17x  5.53x   855.4 -0.17x  -0.51x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      555
fp64      0
other     21
**total** **576**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Boys Function F0 (boys_f0)
--------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============ ========= ======= ========== ========== ====
Class        Count     Percent Max RelErr Avg RelErr Bits
============ ========= ======= ========== ========== ====
normal (OK)  325851359 100.00% 9.88e-14   1.84e-14   43
unclassified 9         3e-06%  1.09e-13   1.04e-13   43
TOTAL        325851368 100.00%                       
============ ========= ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    164.8   15.62x  15.19x  86.0  -0.64x  -0.63x
ev/clk/SM 0.36    15.62x  15.19x  0.30  -0.64x  -0.63x
clk/ev    353.2   14.14x  13.78x  376.7 2.24x   2.16x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      749
fp64      0
other     11
**total** **760**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Inverse Normal CDF (normcdfinv)
-------------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============ ========= ======= ========== ========== ====
Class        Count     Percent Max RelErr Avg RelErr Bits
============ ========= ======= ========== ========== ====
normal (OK)  162681097 97.20%  1.00e-13   7.38e-15   43
unclassified 4683644   2.80%   4.37e-13   1.47e-13   41
TOTAL        167364741 100.00%                       
============ ========= ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    103.4   -0.05x  7.51x   52.6  -0.04x  -0.28x
ev/clk/SM 0.23    -0.05x  7.51x   0.18  -0.04x  -0.28x
clk/ev    889.3   -0.12x  4.15x   936.1 -0.13x  -0.57x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      750
fp64      0
other     36
**total** **786**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

Floor (floor)
-------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 3212836860 100.00% 0.00e+00   0.00e+00   48
TOTAL       3212836860 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    3051.0  -0.44x  3.69x   1885.8 -0.47x  -0.50x
ev/clk/SM 6.68    -0.44x  3.69x   6.48   -0.47x  -0.50x
clk/ev    104.0   -0.24x  1.02x   107.3  -0.26x  -0.28x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      28
fp64      0
other     13
**total** **41**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Ceiling (ceil)
--------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 3212836861 100.00% 0.00e+00   0.00e+00   48
TOTAL       3212836861 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    3050.9  -0.44x  3.69x   1884.3 -0.47x  -0.50x
ev/clk/SM 6.68    -0.44x  3.69x   6.48   -0.47x  -0.50x
clk/ev    103.9   -0.23x  1.02x   107.3  -0.26x  -0.28x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      28
fp64      0
other     13
**total** **41**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Round to Nearest (round)
------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2164260863 100.00% 3.55e-15   9.54e-18   48
TOTAL       2164260863 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ===== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200  vs fp32 vs fp64
========= ======= ======= ======= ===== ======= =======
GFLOPS    1433.8  -0.21x  3.63x   885.9 -0.22x  -0.26x
ev/clk/SM 3.14    -0.21x  3.63x   3.05  -0.22x  -0.26x
clk/ev    207.7   -0.18x  -0.81x  215.0 -0.17x  -0.22x
========= ======= ======= ======= ===== ======= =======

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      71
fp64      0
other     29
**total** **100**
========= =======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

Truncate (trunc)
----------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2147483646 100.00% 0.00e+00   0.00e+00   48
TOTAL       2147483646 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2999.2  -0.43x  3.63x   1948.6 -0.49x  -0.52x
ev/clk/SM 6.57    -0.43x  3.63x   6.70   -0.49x  -0.52x
clk/ev    105.3   -0.24x  1.01x   108.0  -0.25x  -0.28x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      35
fp64      0
other     23
**total** **58**
========= ======

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

Comparison Operations
=====================

Equal (eq)
----------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    8873.3  -0.62x  12.36x  3696.9 -0.66x  -0.76x
ev/clk/SM 19.42   -0.62x  12.36x  12.71  -0.66x  -0.76x
clk/ev    21.9    -0.83x  5.32x   19.2   -0.83x  1.17x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     1
**total** **3**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  1    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**-maxN** 0    1     0  0     0     0     0  0  0     0     0     0  0     0    0
**-1**    0    0     1  0     0     0     0  0  0     0     0     0  0     0    0
**-minN** 0    0     0  1     0     0     0  0  0     0     0     0  0     0    0
**-maxD** 0    0     0  0     1     0     0  0  0     0     0     0  0     0    0
**-minD** 0    0     0  0     0     1     0  0  0     0     0     0  0     0    0
**-0**    0    0     0  0     0     0     1  1  0     0     0     0  0     0    0
**+0**    0    0     0  0     0     0     1  1  0     0     0     0  0     0    0
**+minD** 0    0     0  0     0     0     0  0  1     0     0     0  0     0    0
**+maxD** 0    0     0  0     0     0     0  0  0     1     0     0  0     0    0
**+minN** 0    0     0  0     0     0     0  0  0     0     1     0  0     0    0
**+1**    0    0     0  0     0     0     0  0  0     0     0     1  0     0    0
**+maxN** 0    0     0  0     0     0     0  0  0     0     0     0  1     0    0
**+INF**  0    0     0  0     0     0     0  0  0     0     0     0  0     1    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    366.8   -0.51x  -0.38x   2843.9 -0.59x  6.10x
ev/clk/SM 0.80    -0.51x  -0.38x   9.78   -0.59x  6.10x
clk/ev    187.3   -0.63x  -0.61x   29.3   -0.76x  3.96x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      2
other     1
**total** **3**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  1    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**-maxN** 0    1     0  0     0     0     0  0  0     0     0     0  0     0    0
**-1**    0    0     1  0     0     0     0  0  0     0     0     0  0     0    0
**-minN** 0    0     0  1     0     0     0  0  0     0     0     0  0     0    0
**-maxD** 0    0     0  0     1     0     0  0  0     0     0     0  0     0    0
**-minD** 0    0     0  0     0     1     0  0  0     0     0     0  0     0    0
**-0**    0    0     0  0     0     0     1  1  0     0     0     0  0     0    0
**+0**    0    0     0  0     0     0     1  1  0     0     0     0  0     0    0
**+minD** 0    0     0  0     0     0     0  0  1     0     0     0  0     0    0
**+maxD** 0    0     0  0     0     0     0  0  0     1     0     0  0     0    0
**+minN** 0    0     0  0     0     0     0  0  0     0     1     0  0     0    0
**+1**    0    0     0  0     0     0     0  0  0     0     0     1  0     0    0
**+maxN** 0    0     0  0     0     0     0  0  0     0     0     0  1     0    0
**+INF**  0    0     0  0     0     0     0  0  0     0     0     0  0     1    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====

--------------

Not Equal (ne)
--------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    8992.8  -0.63x  12.28x  3689.6 -0.66x  -0.76x
ev/clk/SM 19.68   -0.63x  12.28x  12.69  -0.66x  -0.76x
clk/ev    21.8    -0.82x  5.33x   19.2   -0.82x  1.17x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     1
**total** **3**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  0    1     1  1     1     1     1  1  1     1     1     1  1     1    1
**-maxN** 1    0     1  1     1     1     1  1  1     1     1     1  1     1    1
**-1**    1    1     0  1     1     1     1  1  1     1     1     1  1     1    1
**-minN** 1    1     1  0     1     1     1  1  1     1     1     1  1     1    1
**-maxD** 1    1     1  1     0     1     1  1  1     1     1     1  1     1    1
**-minD** 1    1     1  1     1     0     1  1  1     1     1     1  1     1    1
**-0**    1    1     1  1     1     1     0  0  1     1     1     1  1     1    1
**+0**    1    1     1  1     1     1     0  0  1     1     1     1  1     1    1
**+minD** 1    1     1  1     1     1     1  1  0     1     1     1  1     1    1
**+maxD** 1    1     1  1     1     1     1  1  1     0     1     1  1     1    1
**+minN** 1    1     1  1     1     1     1  1  1     1     0     1  1     1    1
**+1**    1    1     1  1     1     1     1  1  1     1     1     0  1     1    1
**+maxN** 1    1     1  1     1     1     1  1  1     1     1     1  0     1    1
**+INF**  1    1     1  1     1     1     1  1  1     1     1     1  1     0    1
**QNAN**  1    1     1  1     1     1     1  1  1     1     1     1  1     1    1
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    366.3   -0.53x  -0.38x   2841.9 -0.59x  6.07x
ev/clk/SM 0.80    -0.53x  -0.38x   9.77   -0.59x  6.07x
clk/ev    187.2   -0.60x  -0.59x   29.1   -0.78x  3.83x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      2
other     1
**total** **3**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  0    1     1  1     1     1     1  1  1     1     1     1  1     1    1
**-maxN** 1    0     1  1     1     1     1  1  1     1     1     1  1     1    1
**-1**    1    1     0  1     1     1     1  1  1     1     1     1  1     1    1
**-minN** 1    1     1  0     1     1     1  1  1     1     1     1  1     1    1
**-maxD** 1    1     1  1     0     1     1  1  1     1     1     1  1     1    1
**-minD** 1    1     1  1     1     0     1  1  1     1     1     1  1     1    1
**-0**    1    1     1  1     1     1     0  0  1     1     1     1  1     1    1
**+0**    1    1     1  1     1     1     0  0  1     1     1     1  1     1    1
**+minD** 1    1     1  1     1     1     1  1  0     1     1     1  1     1    1
**+maxD** 1    1     1  1     1     1     1  1  1     0     1     1  1     1    1
**+minN** 1    1     1  1     1     1     1  1  1     1     0     1  1     1    1
**+1**    1    1     1  1     1     1     1  1  1     1     1     0  1     1    1
**+maxN** 1    1     1  1     1     1     1  1  1     1     1     1  0     1    1
**+INF**  1    1     1  1     1     1     1  1  1     1     1     1  1     0    1
**QNAN**  1    1     1  1     1     1     1  1  1     1     1     1  1     1    1
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====

--------------

Less Than (lt)
--------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    6948.2  -0.49x  9.79x   2757.9 -0.50x  -0.57x
ev/clk/SM 15.21   -0.49x  9.79x   9.48   -0.50x  -0.57x
clk/ev    37.0    -0.50x  3.14x   33.9   -0.47x  -0.66x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      3
fp64      0
other     2
**total** **5**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  0    1     1  1     1     1     1  1  1     1     1     1  1     1    0
**-maxN** 0    0     1  1     1     1     1  1  1     1     1     1  1     1    0
**-1**    0    0     0  1     1     1     1  1  1     1     1     1  1     1    0
**-minN** 0    0     0  0     1     1     1  1  1     1     1     1  1     1    0
**-maxD** 0    0     0  0     0     1     1  1  1     1     1     1  1     1    0
**-minD** 0    0     0  0     0     0     1  1  1     1     1     1  1     1    0
**-0**    0    0     0  0     0     0     0  0  1     1     1     1  1     1    0
**+0**    0    0     0  0     0     0     0  0  1     1     1     1  1     1    0
**+minD** 0    0     0  0     0     0     0  0  0     1     1     1  1     1    0
**+maxD** 0    0     0  0     0     0     0  0  0     0     1     1  1     1    0
**+minN** 0    0     0  0     0     0     0  0  0     0     0     1  1     1    0
**+1**    0    0     0  0     0     0     0  0  0     0     0     0  1     1    0
**+maxN** 0    0     0  0     0     0     0  0  0     0     0     0  0     1    0
**+INF**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    253.8   -0.35x  -0.27x   2570.2 -0.54x  5.80x
ev/clk/SM 0.56    -0.35x  -0.27x   8.84   -0.54x  5.80x
clk/ev    228.6   -0.50x  -0.45x   47.1   -0.48x  2.36x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     2
**total** **5**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  0    1     1  1     1     1     1  1  1     1     1     1  1     1    0
**-maxN** 0    0     1  1     1     1     1  1  1     1     1     1  1     1    0
**-1**    0    0     0  1     1     1     1  1  1     1     1     1  1     1    0
**-minN** 0    0     0  0     1     1     1  1  1     1     1     1  1     1    0
**-maxD** 0    0     0  0     0     1     1  1  1     1     1     1  1     1    0
**-minD** 0    0     0  0     0     0     1  1  1     1     1     1  1     1    0
**-0**    0    0     0  0     0     0     0  0  1     1     1     1  1     1    0
**+0**    0    0     0  0     0     0     0  0  1     1     1     1  1     1    0
**+minD** 0    0     0  0     0     0     0  0  0     1     1     1  1     1    0
**+maxD** 0    0     0  0     0     0     0  0  0     0     1     1  1     1    0
**+minN** 0    0     0  0     0     0     0  0  0     0     0     1  1     1    0
**+1**    0    0     0  0     0     0     0  0  0     0     0     0  1     1    0
**+maxN** 0    0     0  0     0     0     0  0  0     0     0     0  0     1    0
**+INF**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====

--------------

Less Than or Equal (le)
-----------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    6932.7  -0.48x  9.46x   2755.6 -0.49x  -0.57x
ev/clk/SM 15.18   -0.48x  9.46x   9.48   -0.49x  -0.57x
clk/ev    37.0    -0.48x  3.13x   34.1   -0.46x  -0.66x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      3
fp64      0
other     2
**total** **5**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  1    1     1  1     1     1     1  1  1     1     1     1  1     1    0
**-maxN** 0    1     1  1     1     1     1  1  1     1     1     1  1     1    0
**-1**    0    0     1  1     1     1     1  1  1     1     1     1  1     1    0
**-minN** 0    0     0  1     1     1     1  1  1     1     1     1  1     1    0
**-maxD** 0    0     0  0     1     1     1  1  1     1     1     1  1     1    0
**-minD** 0    0     0  0     0     1     1  1  1     1     1     1  1     1    0
**-0**    0    0     0  0     0     0     1  1  1     1     1     1  1     1    0
**+0**    0    0     0  0     0     0     1  1  1     1     1     1  1     1    0
**+minD** 0    0     0  0     0     0     0  0  1     1     1     1  1     1    0
**+maxD** 0    0     0  0     0     0     0  0  0     1     1     1  1     1    0
**+minN** 0    0     0  0     0     0     0  0  0     0     1     1  1     1    0
**+1**    0    0     0  0     0     0     0  0  0     0     0     1  1     1    0
**+maxN** 0    0     0  0     0     0     0  0  0     0     0     0  1     1    0
**+INF**  0    0     0  0     0     0     0  0  0     0     0     0  0     1    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    253.3   -0.36x  -0.27x   2569.6 -0.54x  5.80x
ev/clk/SM 0.55    -0.36x  -0.27x   8.84   -0.54x  5.80x
clk/ev    228.3   -0.52x  -0.53x   47.4   -0.47x  2.53x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     2
**total** **5**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  1    1     1  1     1     1     1  1  1     1     1     1  1     1    0
**-maxN** 0    1     1  1     1     1     1  1  1     1     1     1  1     1    0
**-1**    0    0     1  1     1     1     1  1  1     1     1     1  1     1    0
**-minN** 0    0     0  1     1     1     1  1  1     1     1     1  1     1    0
**-maxD** 0    0     0  0     1     1     1  1  1     1     1     1  1     1    0
**-minD** 0    0     0  0     0     1     1  1  1     1     1     1  1     1    0
**-0**    0    0     0  0     0     0     1  1  1     1     1     1  1     1    0
**+0**    0    0     0  0     0     0     1  1  1     1     1     1  1     1    0
**+minD** 0    0     0  0     0     0     0  0  1     1     1     1  1     1    0
**+maxD** 0    0     0  0     0     0     0  0  0     1     1     1  1     1    0
**+minN** 0    0     0  0     0     0     0  0  0     0     1     1  1     1    0
**+1**    0    0     0  0     0     0     0  0  0     0     0     1  1     1    0
**+maxN** 0    0     0  0     0     0     0  0  0     0     0     0  1     1    0
**+INF**  0    0     0  0     0     0     0  0  0     0     0     0  0     1    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====

--------------

Greater Than (gt)
-----------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    6967.3  -0.49x  9.62x   2747.3 -0.49x  -0.57x
ev/clk/SM 15.25   -0.49x  9.62x   9.45   -0.49x  -0.57x
clk/ev    37.2    -0.48x  3.11x   34.1   -0.47x  -0.66x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      3
fp64      0
other     2
**total** **5**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**-maxN** 1    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**-1**    1    1     0  0     0     0     0  0  0     0     0     0  0     0    0
**-minN** 1    1     1  0     0     0     0  0  0     0     0     0  0     0    0
**-maxD** 1    1     1  1     0     0     0  0  0     0     0     0  0     0    0
**-minD** 1    1     1  1     1     0     0  0  0     0     0     0  0     0    0
**-0**    1    1     1  1     1     1     0  0  0     0     0     0  0     0    0
**+0**    1    1     1  1     1     1     0  0  0     0     0     0  0     0    0
**+minD** 1    1     1  1     1     1     1  1  0     0     0     0  0     0    0
**+maxD** 1    1     1  1     1     1     1  1  1     0     0     0  0     0    0
**+minN** 1    1     1  1     1     1     1  1  1     1     0     0  0     0    0
**+1**    1    1     1  1     1     1     1  1  1     1     1     0  0     0    0
**+maxN** 1    1     1  1     1     1     1  1  1     1     1     1  0     0    0
**+INF**  1    1     1  1     1     1     1  1  1     1     1     1  1     0    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    245.2   -0.35x  -0.26x   2443.9 -0.51x  5.52x
ev/clk/SM 0.54    -0.35x  -0.26x   8.40   -0.51x  5.52x
clk/ev    239.7   -0.47x  -0.49x   47.3   -0.47x  2.45x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     2
**total** **5**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**-maxN** 1    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**-1**    1    1     0  0     0     0     0  0  0     0     0     0  0     0    0
**-minN** 1    1     1  0     0     0     0  0  0     0     0     0  0     0    0
**-maxD** 1    1     1  1     0     0     0  0  0     0     0     0  0     0    0
**-minD** 1    1     1  1     1     0     0  0  0     0     0     0  0     0    0
**-0**    1    1     1  1     1     1     0  0  0     0     0     0  0     0    0
**+0**    1    1     1  1     1     1     0  0  0     0     0     0  0     0    0
**+minD** 1    1     1  1     1     1     1  1  0     0     0     0  0     0    0
**+maxD** 1    1     1  1     1     1     1  1  1     0     0     0  0     0    0
**+minN** 1    1     1  1     1     1     1  1  1     1     0     0  0     0    0
**+1**    1    1     1  1     1     1     1  1  1     1     1     0  0     0    0
**+maxN** 1    1     1  1     1     1     1  1  1     1     1     1  0     0    0
**+INF**  1    1     1  1     1     1     1  1  1     1     1     1  1     0    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====

--------------

Greater Than or Equal (ge)
--------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    6928.4  -0.48x  9.54x   2748.7 -0.49x  -0.57x
ev/clk/SM 15.17   -0.48x  9.54x   9.45   -0.49x  -0.57x
clk/ev    36.7    -0.50x  3.16x   33.7   -0.47x  -0.67x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      3
fp64      0
other     2
**total** **5**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  1    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**-maxN** 1    1     0  0     0     0     0  0  0     0     0     0  0     0    0
**-1**    1    1     1  0     0     0     0  0  0     0     0     0  0     0    0
**-minN** 1    1     1  1     0     0     0  0  0     0     0     0  0     0    0
**-maxD** 1    1     1  1     1     0     0  0  0     0     0     0  0     0    0
**-minD** 1    1     1  1     1     1     0  0  0     0     0     0  0     0    0
**-0**    1    1     1  1     1     1     1  1  0     0     0     0  0     0    0
**+0**    1    1     1  1     1     1     1  1  0     0     0     0  0     0    0
**+minD** 1    1     1  1     1     1     1  1  1     0     0     0  0     0    0
**+maxD** 1    1     1  1     1     1     1  1  1     1     0     0  0     0    0
**+minN** 1    1     1  1     1     1     1  1  1     1     1     0  0     0    0
**+1**    1    1     1  1     1     1     1  1  1     1     1     1  0     0    0
**+maxN** 1    1     1  1     1     1     1  1  1     1     1     1  1     0    0
**+INF**  1    1     1  1     1     1     1  1  1     1     1     1  1     1    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    245.0   -0.33x  -0.26x   2443.5 -0.51x  5.52x
ev/clk/SM 0.54    -0.33x  -0.26x   8.40   -0.51x  5.52x
clk/ev    239.7   -0.47x  -0.43x   47.1   -0.48x  2.27x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     2
**total** **5**
========= =====

**Special Values Table:**

========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**a\b**   -INF -maxN -1 -minN -maxD -minD -0 +0 +minD +maxD +minN +1 +maxN +INF QNAN
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====
**-INF**  1    0     0  0     0     0     0  0  0     0     0     0  0     0    0
**-maxN** 1    1     0  0     0     0     0  0  0     0     0     0  0     0    0
**-1**    1    1     1  0     0     0     0  0  0     0     0     0  0     0    0
**-minN** 1    1     1  1     0     0     0  0  0     0     0     0  0     0    0
**-maxD** 1    1     1  1     1     0     0  0  0     0     0     0  0     0    0
**-minD** 1    1     1  1     1     1     0  0  0     0     0     0  0     0    0
**-0**    1    1     1  1     1     1     1  1  0     0     0     0  0     0    0
**+0**    1    1     1  1     1     1     1  1  0     0     0     0  0     0    0
**+minD** 1    1     1  1     1     1     1  1  1     0     0     0  0     0    0
**+maxD** 1    1     1  1     1     1     1  1  1     1     0     0  0     0    0
**+minN** 1    1     1  1     1     1     1  1  1     1     1     0  0     0    0
**+1**    1    1     1  1     1     1     1  1  1     1     1     1  0     0    0
**+maxN** 1    1     1  1     1     1     1  1  1     1     1     1  1     0    0
**+INF**  1    1     1  1     1     1     1  1  1     1     1     1  1     1    0
**QNAN**  0    0     0  0     0     0     0  0  0     0     0     0  0     0    0
========= ==== ===== == ===== ===== ===== == == ===== ===== ===== == ===== ==== ====

Type Conversions
================

To Int32 (mp2int)
-----------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2533359616 100.00% 0.00e+00   0.00e+00   32
TOTAL       2533359616 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2241.5  -0.33x  -0.99x  1389.6 -0.35x  -1.00x
ev/clk/SM 4.91    -0.33x  -0.99x  4.78   -0.35x  -1.00x
clk/ev    73.3    -0.34x  -0.99x  73.2   -0.38x  1.00x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     5
**total** **7**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 8577024 100.00% 0.00e+00   0.00e+00   32
TOTAL       8577024 100.00%                       
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    153.3   -0.19x  -1.00x   1331.7 -0.36x  -1.00x
ev/clk/SM 0.34    -0.19x  -1.00x   4.58   -0.36x  -1.00x
clk/ev    391.8   -0.28x  1.01x    80.2   -0.43x  1.00x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     3
**total** **8**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

To UInt32 (mp2uint)
-------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 1266679810 100.00% 0.00e+00   0.00e+00   32
TOTAL       1266679810 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    2236.7  -0.33x  -0.99x  1389.3 -0.35x  -1.00x
ev/clk/SM 4.90    -0.33x  -0.99x  4.78   -0.35x  -1.00x
clk/ev    73.5    -0.34x  -0.99x  73.2   -0.38x  1.00x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     5
**total** **7**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 4288512 100.00% 0.00e+00   0.00e+00   32
TOTAL       4288512 100.00%                       
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    153.4   -0.19x  1.00x    1332.1 -0.36x  -1.00x
ev/clk/SM 0.34    -0.19x  1.00x    4.58   -0.36x  -1.00x
clk/ev    389.0   -0.29x  1.00x    80.4   -0.42x  1.00x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     3
**total** **8**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

To Int64 (mp2ll)
----------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2533359616 100.00% 0.00e+00   0.00e+00   64
TOTAL       2533359616 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    258.2   -0.32x  1.00x   1403.7 -0.35x  -1.00x
ev/clk/SM 0.57    -0.32x  1.00x   4.83   -0.35x  -1.00x
clk/ev    273.2   -0.41x  1.00x   63.5   -0.43x  1.00x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     6
**total** **8**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 8577024 100.00% 0.00e+00   0.00e+00   64
TOTAL       8577024 100.00%                       
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    151.9   -0.19x  1.00x    1332.3 -0.36x  -1.00x
ev/clk/SM 0.33    -0.19x  1.00x    4.58   -0.36x  -1.00x
clk/ev    396.9   -0.28x  1.00x    77.1   -0.39x  1.01x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     3
**total** **8**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

To UInt64 (mp2ull)
------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 1266679810 100.00% 0.00e+00   0.00e+00   64
TOTAL       1266679810 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    258.1   -0.32x  -1.00x  1404.4 -0.35x  -1.00x
ev/clk/SM 0.56    -0.32x  -1.00x  4.83   -0.35x  -1.00x
clk/ev    273.8   -0.40x  1.00x   63.4   -0.44x  1.01x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     6
**total** **8**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 4288512 100.00% 0.00e+00   0.00e+00   64
TOTAL       4288512 100.00%                       
=========== ======= ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    151.9   -0.19x  1.00x    1332.1 -0.36x  -1.00x
ev/clk/SM 0.33    -0.19x  1.00x    4.58   -0.36x  -1.00x
clk/ev    391.4   -0.28x  1.00x    77.1   -0.39x  1.00x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     3
**total** **8**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-LMAX** -9.2e+18
**-IMAX** -2.1e+09
**-1**    -1
**-0**    -0
**+0**    +0
**+1**    1
**+IMAX** 2.1e+09
**+UMAX** 4.3e+09
**+LMAX** 9.2e+18
**+INF**  +inf
**QNAN**  nan
========= ========

--------------

From Int32 (int2mp)
-------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967295 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967295 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    13313.9 -0.98x  -0.99x  8247.4 -1.00x  -0.99x
ev/clk/SM 29.14   -0.98x  -0.99x  28.36  -1.00x  -0.99x
clk/ev    10.8    -1.00x  1.02x   11.5   1.01x   -1.00x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     3
**total** **5**
========= =====

**Special Values Table:**

========= ===========
**Input** Value
========= ===========
**0**     0
**+1**    1
**-1**    -1
**+2**    2
**-2**    -2
**+MAX**  2147483647
**-MAX**  -2147483648
**+Mx-1** 2147483646
**-Mx+1** -2147483647
**+half** 1073741823
**-half** -1073741824
**+100**  100
**-100**  -100
**+1M**   1000000
**-1M**   -1000000
**~MAX**  2147483632
========= ===========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    799.4   3.04x   -1.00x   3973.5 2.92x   -1.00x
ev/clk/SM 1.75    3.04x   -1.00x   13.66  2.92x   -1.00x
clk/ev    110.1   2.27x   -0.99x   27.5   2.48x   1.01x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     3
**total** **6**
========= =====

**Special Values Table:**

========= ===========
**Input** Value
========= ===========
**0**     0
**+1**    1
**-1**    -1
**+2**    2
**-2**    -2
**+MAX**  2147483647
**-MAX**  -2147483648
**+Mx-1** 2147483646
**-Mx+1** -2147483647
**+half** 1073741823
**-half** -1073741824
**+100**  100
**-100**  -100
**+1M**   1000000
**-1M**   -1000000
**~MAX**  2147483632
========= ===========

--------------

From UInt32 (uint2mp)
---------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967295 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967295 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    13203.9 -0.98x  -0.97x  8251.4 -1.00x  -0.99x
ev/clk/SM 28.90   -0.98x  -0.97x  28.37  -1.00x  -0.99x
clk/ev    10.3    1.09x   1.10x   11.8   1.00x   -0.99x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      2
fp64      0
other     3
**total** **5**
========= =====

**Special Values Table:**

========= ==========
**Input** Value
========= ==========
**0**     0
**1**     1
**2**     2
**MAX**   4294967295
**Mx-1**  4294967294
**half**  2147483647
**hlf+1** 2147483648
**100**   100
**1K**    1000
**1M**    1000000
**1B**    1000000000
**MSB**   2147483648
**~MSB**  2147483647
**hi16**  4294901760
**lo16**  65535
**0xAA**  2863311530
========= ==========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    799.0   3.04x   -1.00x   3974.8 2.92x   -1.00x
ev/clk/SM 1.75    3.04x   -1.00x   13.67  2.92x   -1.00x
clk/ev    111.0   2.24x   1.00x    27.4   2.49x   1.01x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     3
**total** **6**
========= =====

**Special Values Table:**

========= ==========
**Input** Value
========= ==========
**0**     0
**1**     1
**2**     2
**MAX**   4294967295
**Mx-1**  4294967294
**half**  2147483647
**hlf+1** 2147483648
**100**   100
**1K**    1000
**1M**    1000000
**1B**    1000000000
**MSB**   2147483648
**~MSB**  2147483647
**hi16**  4294901760
**lo16**  65535
**0xAA**  2863311530
========= ==========

--------------

From Int64 (ll2mp)
------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967296 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967296 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    271.5   -0.34x  1.00x   1286.2 -0.39x  -1.00x
ev/clk/SM 0.59    -0.34x  1.00x   4.42   -0.39x  -1.00x
clk/ev    316.0   -0.34x  1.00x   81.1   -0.48x  -1.00x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      0
other     6
**total** **6**
========= =====

**Special Values Table:**

========= ====================
**Input** Value
========= ====================
**0**     0
**+1**    1
**-1**    -1
**+2**    2
**-2**    -2
**+MAX**  9223372036854775807
**-MAX**  -9223372036854775808
**+Mx-1** 9223372036854775806
**-Mx+1** -9223372036854775806
**+half** 4611686018427387903
**-half** -4611686018427387904
**+100**  100
**-100**  -100
**+1T**   1000000000000
**-1T**   -1000000000000
**~MAX**  9223372036854775792
========= ====================


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    799.2   6.08x   -1.00x   3758.1 4.70x   -1.00x
ev/clk/SM 1.75    6.08x   -1.00x   12.92  4.70x   -1.00x
clk/ev    110.7   4.34x   -0.99x   30.0   3.79x   -1.00x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     3
**total** **6**
========= =====

**Special Values Table:**

========= ====================
**Input** Value
========= ====================
**0**     0
**+1**    1
**-1**    -1
**+2**    2
**-2**    -2
**+MAX**  9223372036854775807
**-MAX**  -9223372036854775808
**+Mx-1** 9223372036854775806
**-Mx+1** -9223372036854775806
**+half** 4611686018427387903
**-half** -4611686018427387904
**+100**  100
**-100**  -100
**+1T**   1000000000000
**-1T**   -1000000000000
**~MAX**  9223372036854775792
========= ====================

--------------

From UInt64 (ull2mp)
--------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967296 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967296 100.00%                       
=========== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    271.0   -0.34x  -1.00x  1286.7 -0.39x  -1.00x
ev/clk/SM 0.59    -0.34x  -1.00x  4.42   -0.39x  -1.00x
clk/ev    316.8   -0.34x  -1.00x  81.4   -0.48x  -1.00x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      0
other     6
**total** **6**
========= =====

**Special Values Table:**

========= ====================
**Input** Value
========= ====================
**0**     0
**1**     1
**2**     2
**MAX**   18446744073709551615
**Mx-1**  18446744073709551614
**half**  9223372036854775807
**hlf+1** 9223372036854775808
**100**   100
**1T**    1000000000000
**MSB**   9223372036854775808
**~MSB**  9223372036854775807
**hi32**  18446744069414584320
**lo32**  4294967295
**0xAA**  12297829382473034410
**0x55**  6148914691236517205
**~MAX**  18446744073709551360
========= ====================


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%                       
=========== ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    797.8   6.08x   -1.00x   3761.2 4.70x   -1.00x
ev/clk/SM 1.75    6.08x   -1.00x   12.93  4.70x   -1.00x
clk/ev    110.6   4.40x   -1.00x   29.7   3.83x   1.01x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     3
**total** **6**
========= =====

**Special Values Table:**

========= ====================
**Input** Value
========= ====================
**0**     0
**1**     1
**2**     2
**MAX**   18446744073709551615
**Mx-1**  18446744073709551614
**half**  9223372036854775807
**hlf+1** 9223372036854775808
**100**   100
**1T**    1000000000000
**MSB**   9223372036854775808
**~MSB**  9223372036854775807
**hi32**  18446744069414584320
**lo32**  4294967295
**0xAA**  12297829382473034410
**0x55**  6148914691236517205
**~MAX**  18446744073709551360
========= ====================

--------------

To Native Float (mp2fp)
-----------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4278190075 100.00% 0.00e+00   0.00e+00   53
input special 3          7e-08%  --         --         --
TOTAL         4278190078 100.00%                       
============= ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    262.3   -0.33x  -1.00x  2081.3 -0.52x  -1.00x
ev/clk/SM 0.57    -0.33x  -1.00x  7.16   -0.52x  -1.00x
clk/ev    247.5   -0.44x  1.00x   45.3   -0.60x  1.00x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      3
other     2
**total** **5**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============= ======== ======= ========== ========== ====
Class         Count    Percent Max RelErr Avg RelErr Bits
============= ======== ======= ========== ========== ====
normal (OK)   16769024 99.95%  0.00e+00   0.00e+00   113
input special 8192     0.05%   --         --         --
TOTAL         16777216 100.00%                       
============= ======== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ====== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200   vs fp64 vs fp128
========= ======= ======= ======== ====== ======= ========
GFLOPS    19.5    -0.51x  -0.99x   72.6   -0.16x  -1.00x
ev/clk/SM 0.04    -0.51x  -0.99x   0.25   -0.16x  -1.00x
clk/ev    3582.7  -0.43x  1.03x    1014.0 -0.26x  -1.00x
========= ======= ======= ======== ====== ======= ========

**SASS Instructions:**

========= ========
Class     Count
========= ========
fp32      0
fp64      86
other     1006
**total** **1092**
========= ========

**Special Values Table:**

========= =========
**Input** Value
========= =========
**-INF**  -inf
**-maxN** -1.8e+308
**-1**    -1
**-minN** -2.2e-308
**-maxD** -2.2e-308
**-minD** -4.9e-324
**-0**    -0
**+0**    +0
**+minD** 4.9e-324
**+maxD** 2.2e-308
**+minN** 2.2e-308
**+1**    1
**+maxN** 1.8e+308
**+INF**  +inf
**QNAN**  nan
========= =========

--------------

From Native Float (fp2mp)
-------------------------


Type: fp32mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    532676608  22.09%  0.00e+00   0.00e+00   48
output special 1879048192 77.91%  --         --         --
TOTAL          2411724800 100.00%                       
============== ========== ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======= ====== ======= =======
Metric    rtx6000 vs fp32 vs fp64 b200   vs fp32 vs fp64
========= ======= ======= ======= ====== ======= =======
GFLOPS    195.5   -0.24x  -1.00x  1293.3 -0.35x  -1.00x
ev/clk/SM 0.43    -0.24x  -1.00x  4.45   -0.35x  -1.00x
clk/ev    340.1   -0.32x  1.01x   80.2   -0.42x  1.00x
========= ======= ======= ======= ====== ======= =======

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      4
other     2
**total** **6**
========= =====

**Special Values Table:**

========= ========
**Input** Value
========= ========
**-INF**  -inf
**-maxN** -3.4e+38
**-1**    -1
**-minN** -1.2e-38
**-maxD** -1.2e-38
**-minD** -1.4e-45
**-0**    -0
**+0**    +0
**+minD** 1.4e-45
**+maxD** 1.2e-38
**+minN** 1.2e-38
**+1**    1
**+maxN** 3.4e+38
**+INF**  +inf
**QNAN**  nan
========= ========


Type: fp64mp2
~~~~~~~~~~~~~

*Method: ``def``*

**Measured Accuracy:**

=============== ======= ======= ========== ========== ====
Class           Count   Percent Max RelErr Avg RelErr Bits
=============== ======= ======= ========== ========== ====
normal (OK)     1047552 11.75%  0.00e+00   0.00e+00   106
input special   7864320 88.24%  --         --         --
output denormal 255     3e-03%  0.00e+00   0.00e+00   0
TOTAL           8912127 100.00%                       
=============== ======= ======= ========== ========== ====

**Measured Performance:**

========= ======= ======= ======== ===== ======= ========
Metric    rtx6000 vs fp64 vs fp128 b200  vs fp64 vs fp128
========= ======= ======= ======== ===== ======= ========
GFLOPS    14.8    -0.31x  -1.00x   61.5  -0.23x  -1.00x
ev/clk/SM 0.03    -0.31x  -1.00x   0.21  -0.23x  -1.00x
clk/ev    4259.0  -0.30x  -1.00x   936.2 -0.26x  -1.00x
========= ======= ======= ======== ===== ======= ========

**SASS Instructions:**

========= ========
Class     Count
========= ========
fp32      0
fp64      103
other     1073
**total** **1176**
========= ========

**Special Values Table:**

========= =========
**Input** Value
========= =========
**-INF**  -inf
**-maxN** -1.8e+308
**-1**    -1
**-minN** -2.2e-308
**-maxD** -2.2e-308
**-minD** -4.9e-324
**-0**    -0
**+0**    +0
**+minD** 4.9e-324
**+maxD** 2.2e-308
**+minN** 2.2e-308
**+1**    1
**+maxN** 1.8e+308
**+INF**  +inf
**QNAN**  nan
========= =========

--------------

Appendix: Legends
=================

Measured Accuracy Legend
------------------------

Pattern Dataset
~~~~~~~~~~~~~~~

Accuracy measurements use the **pattern dataset**, which provides exhaustive bit-pattern
coverage across the IEEE-754 floating-point representation space. The dataset is designed
to systematically test:

-  **Sign bit**: Both positive and negative values
-  **Exponent field**: Full range from denormals through maximum normal values
-  **Mantissa bits**: Both most significant and least significant bits

For multi-precision types (fp32mp2, fp64mp2), both the high and low components are
generated with coordinated exponents to ensure proper normalization (\|lo\| < ulp(hi)/2).

**Sample distribution:**

====================================== =========================================================
Region                                 Coverage
====================================== =========================================================
Denormal inputs                        Systematically tested via exponent field patterns
Near-denormal (smallest normal binade) Covered by exponent boundary patterns
Normal range                           Dense coverage with sign/exponent/mantissa bit variations
Near-infinity (largest normal binade)  Covered by exponent boundary patterns
Special values (INF, NaN)              Tested separately in the Special Values Table
====================================== =========================================================

For binary functions, bits are divided equally between arguments. For example, with 32-bit
rigor and a binary function, each argument receives 16 bits of pattern control, yielding
~4 billion test combinations (2^32 total samples).

*Note: For unary functions with a single 32-bit argument (e.g., ``int2mp``, ``uint2mp``),
the pattern dataset provides*\ **exhaustive coverage**\ *of all 2^32 possible input values.*

Classification Categories
~~~~~~~~~~~~~~~~~~~~~~~~~

The accuracy classification table shows only deviations from expected behavior:
cases that exceed the warning threshold or have special value mismatches.
If only ``normal (OK)`` is shown, all test cases passed within acceptable accuracy bounds.

==================== =======================================================================
Category             Description
==================== =======================================================================
normal (OK)          Result within warning threshold (acceptable accuracy)
output special       Output special value mismatch (INF or NaN)
input special        Special input (INF or NaN) causes result mismatch
output denormal      Output is denormal, accuracy loss
input denormal       Denormal input causes accuracy loss
output near denormal Output is in smallest normal binade and exceeds warning threshold
input near denormal  Input is in smallest normal binade and result exceeds warning threshold
output near inf      Output is in largest normal binade and exceeds warning threshold
input near inf       Input is in largest normal binade and result exceeds warning threshold
cancellation         Cancellation case: result is much smaller than inputs
unclassified         Warning without identified cause (between warning and error threshold)
error (FAIL)         Exceeds error threshold without any mitigating factor
==================== =======================================================================

**Table columns:**

========== ===============================================================
Column     Description
========== ===============================================================
Count      Number of test cases in this category
Percent    Percentage of total test cases
Max RelErr Maximum relative error observed in this category
Avg RelErr Average relative error in this category
Bits       Minimum correct mantissa bits (derived from max relative error)
========== ===============================================================

Special Values Legend (Floating Point)
--------------------------------------

====== ===========================================================
Symbol Description
====== ===========================================================
-INF   Negative infinity
+INF   Positive infinity
-maxN  Negative maximum normal (largest finite negative)
+maxN  Positive maximum normal (largest finite positive)
-minN  Negative minimum normal (smallest normal negative)
+minN  Positive minimum normal (smallest normal positive)
-maxD  Negative maximum denormal
+maxD  Positive maximum denormal
-minD  Negative minimum denormal (smallest representable negative)
+minD  Positive minimum denormal (smallest representable positive)
-0     Negative zero
+0     Positive zero
-1     Negative one
+1     Positive one
QNAN   Quiet NaN (Not a Number)
nan    Result is NaN
====== ===========================================================

Special Values Legend (Integer Conversions)
-------------------------------------------

====== ============================================
Symbol Description
====== ============================================
-INF   Negative infinity (saturates to min integer)
+INF   Positive infinity (saturates to max integer)
-LONG  Minimum 64-bit signed integer (-2^63)
+LONG  Maximum 64-bit signed integer (2^63-1)
-INT   Minimum 32-bit signed integer (-2^31)
+INT   Maximum 32-bit signed integer (2^31-1)
+UINT  Maximum 32-bit unsigned integer (2^32-1)
+ULONG Maximum 64-bit unsigned integer (2^64-1)
QNAN   Quiet NaN (converts to 0)
====== ============================================

Performance Metrics Legend
--------------------------

========= ========== ==================================================================================================
Metric    Type       Description
========= ========== ==================================================================================================
GFLOPS    Throughput Giga floating-point operations per second
ev/clk/SM Throughput Evaluations per clock cycle per SM
clk/ev    Latency    Clock cycles per evaluation
vs fp32   Ratio      Performance ratio compared to native fp32 operations
vs fp64   Ratio      Performance ratio compared to native fp64 operations (for fp32mp2) or reference fp64 (for fp64mp2)
vs fp128  Ratio      Performance ratio compared to reference fp128 operations (for fp64mp2)
========= ========== ==================================================================================================

*Note: Negative ratios indicate slower performance than the baseline.*

SASS Instructions Legend
------------------------

Counts are extracted from the per-test ``.sass.mp`` dumps emitted by
``ts/fpmp/Makefile`` when built with ``ASM=y`` (via ``cuobjdump --dump-sass``).
Only the primary ``<op>_device_impl`` function body, up to (but not
including) its ``RET`` epilogue, is counted; ``NOP`` scheduling slots,
trailing self-loop ``BRA`` padding, and any helper functions emitted
in the same dump (libdevice slowpaths, outlined denormal handlers,
etc.) are excluded — their cost is included only when they're
actually called.

===== ================================================================================================================================================================================================================================================================================================================================================================================= =============================================================================================================================================================================================================================================================================================================
Class Mnemonics                                                                                                                                                                                                                                                                                                                                                                         Notes
===== ================================================================================================================================================================================================================================================================================================================================================================================= =============================================================================================================================================================================================================================================================================================================
fp32  ``FADD``, ``FMUL``, ``FFMA``, ``FSET``, ``FSETP``, ``FMNMX``, ``FCMP``, ``FCHK``, ``FRND``, ``FSWZADD``, ``MUFU.{RCP,SQRT,RSQ,SIN,COS,EX2,LG2}``, plus conversion family (``F2F``, ``F2I``, ``I2F``, ``I2FP``, ``F2FP``, ``F2IP``, ``F2DP``, ``D2FP``) when the precision suffix is ``F32`` (e.g. ``F2I.F32.TRUNC``, ``I2FP.F32.S32``)                                            Single-precision IEEE-754 ops. ``MUFU`` is the Multi-Function Unit (transcendentals & reciprocals). ``FSEL`` is *not* in this class — despite the ``F`` prefix it's a predicated 32-bit register MOV with no FP semantics, freely emitted by ptxas in pure bit-manipulation code, so it falls into ``other``.
fp64  ``DADD``, ``DMUL``, ``DFMA``, ``DSET``, ``DSETP``, ``DMNMX``, ``DCMP``, ``F2D``, ``D2F``, ``I2D``, ``UI2D``, ``D2I``, ``D2UI``, ``MUFU.*64H`` (e.g. ``MUFU.RCP64H``, ``MUFU.RSQ64H`` — high-half helpers of fp64 transcendental sequences), plus any conversion-family op with ``F64`` in the suffix (e.g. ``F2I.F64.TRUNC``, ``I2F.F64.S32``, ``F2F.F64.F32``, ``I2FP.F64.S32``) Double-precision IEEE-754 ops. F32↔F64 conversions are counted as fp64 since they exercise the fp64 datapath.
other everything else (integer ALU, memory, control flow, predicate/uniform, fp16 ``H*2``, ``FSEL``, etc.)                                                                                                                                                                                                                                                                              Scheduling slots (``NOP``) are excluded entirely.
===== ================================================================================================================================================================================================================================================================================================================================================================================= =============================================================================================================================================================================================================================================================================================================

*Note: SASS counts are taken from the accuracy run directory (the
``--acc`` argument) and reflect the GPU architecture that directory was
built for. Different GPUs may compile to different instruction counts;
the spec shows one representative count per (function, type, method).*

SASS Instructions Summary
-------------------------

Per-function instruction counts (NOPs excluded) for every
``(type, method)`` combination that exposes a *dedicated*
implementation. Empty cells (``—``) mean either:

-  the combination wasn't built / tested, or
-  the combination is a wrapper over higher-precision system
   math — specifically the ``fp64mp2`` path of
   ``exp``/``log``/``pow``/``sin``/``cos``/``tanh``/``cbrt``/``rcbrt``/
   ``erf``/``erfc``/``boys_f0``/``normcdfinv``/``floor``/``ceil``/
   ``round``/``trunc``, which falls back to system ``fp64`` /
   reference ``fp128`` math and drags in arbitrary ``libm``
   sub-routines whose count isn't comparable to the
   dedicated implementations (see
   `Mathematical Functions <#mathematical-functions>`__).

.. raw:: html

   <table>
   <thead>
   <tr><th rowspan="3">Function</th><th colspan="9">fp32mp2</th><th colspan="9">fp64mp2</th></tr>
   <tr><th colspan="3">def</th><th colspan="3">accurate</th><th colspan="3">fast</th><th colspan="3">def</th><th colspan="3">accurate</th><th colspan="3">fast</th></tr>
   <tr><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th></tr>
   </thead>
   <tbody>
   <tr><td><code>add</code></td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td><td align="right">0</td><td align="right">8</td><td align="right">0</td><td align="right">1</td><td align="right">0</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td><td align="right">0</td><td align="right">8</td><td align="right">4</td></tr>
   <tr><td><code>sub</code></td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td><td align="right">0</td><td align="right">8</td><td align="right">0</td><td align="right">1</td><td align="right">0</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td><td align="right">0</td><td align="right">8</td><td align="right">4</td></tr>
   <tr><td><code>mul</code></td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">5</td><td align="right">0</td><td align="right">1</td><td align="right">0</td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">5</td><td align="right">2</td></tr>
   <tr><td><code>div</code></td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">21</td><td align="right">0</td><td align="right">1</td><td align="right">1</td><td align="right">18</td><td align="right">25</td><td align="right">1</td><td align="right">18</td><td align="right">25</td><td align="right">1</td><td align="right">26</td><td align="right">25</td></tr>
   <tr><td><code>acc</code></td><td align="right">10</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">7</td><td align="right">0</td><td align="right">1</td><td align="right">0</td><td align="right">10</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">7</td><td align="right">2</td></tr>
   <tr><td><code>fma</code></td><td align="right">19</td><td align="right">0</td><td align="right">0</td><td align="right">37</td><td align="right">0</td><td align="right">0</td><td align="right">16</td><td align="right">0</td><td align="right">1</td><td align="right">0</td><td align="right">19</td><td align="right">0</td><td align="right">0</td><td align="right">37</td><td align="right">0</td><td align="right">0</td><td align="right">16</td><td align="right">2</td></tr>
   <tr><td><code>mad</code></td><td align="right">16</td><td align="right">0</td><td align="right">0</td><td align="right">29</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">16</td><td align="right">0</td><td align="right">0</td><td align="right">29</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td></tr>
   <tr><td><code>sqrt</code></td><td align="right">18</td><td align="right">0</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">23</td><td align="right">24</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>rsqrt</code></td><td align="right">17</td><td align="right">0</td><td align="right">0</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">22</td><td align="right">22</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>exp</code></td><td align="right">136</td><td align="right">0</td><td align="right">14</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>log</code></td><td align="right">150</td><td align="right">0</td><td align="right">20</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>pow</code></td><td align="right">320</td><td align="right">0</td><td align="right">69</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>cbrt</code></td><td align="right">75</td><td align="right">0</td><td align="right">27</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>rcbrt</code></td><td align="right">95</td><td align="right">0</td><td align="right">36</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>sin</code></td><td align="right">226</td><td align="right">0</td><td align="right">235</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>cos</code></td><td align="right">226</td><td align="right">0</td><td align="right">235</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>tanh</code></td><td align="right">325</td><td align="right">0</td><td align="right">29</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>erf</code></td><td align="right">560</td><td align="right">0</td><td align="right">22</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>erfc</code></td><td align="right">555</td><td align="right">0</td><td align="right">21</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>boys_f0</code></td><td align="right">749</td><td align="right">0</td><td align="right">11</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>normcdfinv</code></td><td align="right">750</td><td align="right">0</td><td align="right">36</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>floor</code></td><td align="right">28</td><td align="right">0</td><td align="right">13</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ceil</code></td><td align="right">28</td><td align="right">0</td><td align="right">13</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>round</code></td><td align="right">71</td><td align="right">0</td><td align="right">29</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>trunc</code></td><td align="right">35</td><td align="right">0</td><td align="right">23</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>eq</code></td><td align="right">2</td><td align="right">0</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">2</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ne</code></td><td align="right">2</td><td align="right">0</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">2</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>lt</code></td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>le</code></td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>gt</code></td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ge</code></td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2int</code></td><td align="right">2</td><td align="right">0</td><td align="right">5</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2uint</code></td><td align="right">2</td><td align="right">0</td><td align="right">5</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2ll</code></td><td align="right">2</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2ull</code></td><td align="right">2</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>int2mp</code></td><td align="right">2</td><td align="right">0</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>uint2mp</code></td><td align="right">2</td><td align="right">0</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ll2mp</code></td><td align="right">0</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ull2mp</code></td><td align="right">0</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2fp</code></td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">86</td><td align="right">1006</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>fp2mp</code></td><td align="right">0</td><td align="right">4</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">103</td><td align="right">1073</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   </tbody>
   </table>
