.. _libcudacxx-extended-api-fp-fpmp-spec:

.. _fpmp-specification--accuracy-special-values-and-performance:

FPMP Specification — Accuracy, Special Values and Performance
=============================================================

*Generated: 2026-09-09 22:26:09*

.. _libcudacxx-extended-api-fp-fpmp-spec-overview:

Overview
--------

This document is the per-function reference for the ``fpmp`` multi-precision types of the CCCL FP component: measured accuracy, special-value behavior and GPU performance, for every function the two types implement.

The types live in namespace ``cuda::experimental``, abbreviated ``cudax::`` here and in the examples. Arithmetic and the math functions both come in with ``<cuda/fpmp>``. Names appear unqualified in the tables below for width; every one of them is a ``cudax::`` name.

Each function is reported at three accuracy levels, ``low``, ``def`` and ``high``. ``def`` is the default selector and is equal to ``mid``, not to ``high``; ``mid`` is the name to use in code (``cudax::fpmp2_accuracy::mid``).

.. _libcudacxx-extended-api-fp-fpmp-spec-test-platforms:

Test Platforms
--------------

============================================ =========== === =========
GPU                                          Clock (MHz) SMs FP64:FP32
============================================ =========== === =========
NVIDIA RTX PRO 6000 Blackwell Server Edition 2430        188 1:32
NVIDIA B300 SXM6 AC                          2032        148 1:32
NVIDIA B200                                  1965        148 1:2
============================================ =========== === =========

FP64:FP32 is the rate at which the part runs double precision against single, measured here and rounded to the nearest power of two. It is the single number that decides which of these types is worth using: ``fp32mp2`` is built on FP32 and ``fp64mp2`` on FP64, so each one wins on the hardware where the other's arithmetic is the scarce resource.

.. _libcudacxx-extended-api-fp-fpmp-spec-supported-data-types:

Supported Data Types
--------------------

+-------------+----------------+---------------------+--------------------+----------+
| Type        | Representation | Mantissa            | Range              | Size     |
+=============+================+=====================+====================+==========+
| ``fp32mp2`` | float-float    | 46 bits = 2×24 − 2  | float's, ~±10^38   | 8 bytes  |
+-------------+----------------+---------------------+--------------------+----------+
| ``fp64mp2`` | double-double  | 104 bits = 2×53 − 2 | double's, ~±10^308 | 16 bytes |
+-------------+----------------+---------------------+--------------------+----------+

Significand counts include the implicit leading bit, the convention ``cuda::std::numeric_limits::digits`` uses: ``float`` has 24 and ``double`` has 53. A non-overlapping pair guarantees ``2p − 2`` of them, the two subtracted bits being what keeping the halves disjoint costs. Widening the mantissa does not widen the exponent: each type keeps the range of the format it is built from.

.. _libcudacxx-extended-api-fp-fpmp-spec-function-families:

Function Families
-----------------

The math functions are organized into families that mirror the CUDA C++ mathematical standard library taxonomy (CUDA C++ Programming Guide, "Mathematical Functions"). Each family lives in a dedicated implementation header (``fpmp_math_impl_<family>.h``) that contains both the ``fp32mp2`` implementation and the ``fp64mp2`` specialization for its functions. Shared kernels and constants live in ``fpmp_math_impl.h``. Include ``<cuda/fpmp>``, which pulls in all family headers and provides the overloaded ``fpmp2`` API wrappers (template declarations, ``float``/``double`` specializations, the freestanding API, and library-mode declarations), along with the basic arithmetic (``add``, ``sub``, ``mul``, ``div``, ``fma``, ``mad``) and ``sqrt``/``rsqrt``. Functions that CUDA lists as "non-standard" are folded into their natural standard family.

The implementation headers named below are internal and sit under ``cuda/__fp/``; they are listed to show how the implementation is partitioned, not as headers to include directly.

+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Family                      | Implementation header           | Functions                                                                                                                                                                                                               |
+=============================+=================================+=========================================================================================================================================================================================================================+
| Common utilities            | ``fpmp_math_impl.h``            | error-free transforms, Horner/polynomial evaluation, ``fp32mp2``/``fp64mp2`` constants, argument reduction (Cody–Waite, Payne–Hanek), exponent split/scale kernels                                                      |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Exponential                 | ``fpmp_math_impl_exp.h``        | ``exp``, ``exp2``, ``exp10``, ``expm1``, ``log``, ``log2``, ``log10``, ``log1p``                                                                                                                                        |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Power                       | ``fpmp_math_impl_pow.h``        | ``pow``, ``cbrt``, ``rcbrt``, ``hypot``, ``rhypot``, ``norm3d``, ``norm4d``, ``rnorm3d``, ``rnorm4d``                                                                                                                   |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Trigonometric               | ``fpmp_math_impl_trig.h``       | ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``atan2``, ``sincos``, ``sinpi``, ``cospi``, ``sincospi``                                                                                                      |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Hyperbolic                  | ``fpmp_math_impl_hyperbolic.h`` | ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, ``atanh``                                                                                                                                                           |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Error, gamma & special      | ``fpmp_math_impl_special.h``    | ``erf``, ``erfc``, ``erfinv``, ``erfcinv``, ``erfcx``, ``tgamma``, ``lgamma``, ``normcdf``, ``normcdfinv``, ``boys_f0``, ``icdf``, ``j0``, ``j1``, ``jn``, ``y0``, ``y1``, ``yn``, ``cyl_bessel_i0``, ``cyl_bessel_i1`` |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Nearest integer & remainder | ``fpmp_math_impl_nearint.h``    | ``ceil``, ``floor``, ``trunc``, ``round``, ``nearbyint``, ``rint``, ``lrint``, ``llrint``, ``lround``, ``llround``, ``fmod``, ``remainder``, ``remquo``                                                                 |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Floating-point manipulation | ``fpmp_math_impl_manip.h``      | ``frexp``, ``ldexp``, ``modf``, ``scalbn``, ``scalbln``, ``ilogb``, ``logb``, ``nextafter``, ``copysign``, ``fabs``                                                                                                     |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Classification & comparison | ``fpmp_math_impl_classify.h``   | ``isfinite``, ``isinf``, ``isnan``, ``signbit``, ``fmax``, ``fmin``, ``max``, ``min``, ``fdim``                                                                                                                         |
+-----------------------------+---------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. _libcudacxx-extended-api-fp-fpmp-spec-table-of-contents:

Table of Contents
-----------------

:ref:`Function Families <libcudacxx-extended-api-fp-fpmp-spec-function-families>`

:ref:`Arithmetic Operations <libcudacxx-extended-api-fp-fpmp-spec-arithmetic-operations>`

-  :ref:`Addition (add) <libcudacxx-extended-api-fp-fpmp-spec-addition-add>`
-  :ref:`Subtraction (sub) <libcudacxx-extended-api-fp-fpmp-spec-subtraction-sub>`
-  :ref:`Multiplication (mul) <libcudacxx-extended-api-fp-fpmp-spec-multiplication-mul>`
-  :ref:`Division (div) <libcudacxx-extended-api-fp-fpmp-spec-division-div>`
-  :ref:`Accumulate (acc) <libcudacxx-extended-api-fp-fpmp-spec-accumulate-acc>`
-  :ref:`Fused Multiply-Add (fma) <libcudacxx-extended-api-fp-fpmp-spec-fused-multiply-add-fma>`
-  :ref:`Multiply-Add (mad) <libcudacxx-extended-api-fp-fpmp-spec-multiply-add-mad>`

:ref:`Mathematical Functions <libcudacxx-extended-api-fp-fpmp-spec-mathematical-functions>`

-  :ref:`Square Root (sqrt) <libcudacxx-extended-api-fp-fpmp-spec-square-root-sqrt>`
-  :ref:`Reciprocal Square Root (rsqrt) <libcudacxx-extended-api-fp-fpmp-spec-reciprocal-square-root-rsqrt>`
-  :ref:`Exponential (exp) <libcudacxx-extended-api-fp-fpmp-spec-exponential-exp>`
-  :ref:`Natural Logarithm (log) <libcudacxx-extended-api-fp-fpmp-spec-natural-logarithm-log>`
-  :ref:`Power (pow) <libcudacxx-extended-api-fp-fpmp-spec-power-pow>`
-  :ref:`Cube Root (cbrt) <libcudacxx-extended-api-fp-fpmp-spec-cube-root-cbrt>`
-  :ref:`Reciprocal Cube Root (rcbrt) <libcudacxx-extended-api-fp-fpmp-spec-reciprocal-cube-root-rcbrt>`
-  :ref:`Sine (sin) <libcudacxx-extended-api-fp-fpmp-spec-sine-sin>`
-  :ref:`Cosine (cos) <libcudacxx-extended-api-fp-fpmp-spec-cosine-cos>`
-  :ref:`Hyperbolic Tangent (tanh) <libcudacxx-extended-api-fp-fpmp-spec-hyperbolic-tangent-tanh>`
-  :ref:`Error Function (erf) <libcudacxx-extended-api-fp-fpmp-spec-error-function-erf>`
-  :ref:`Complementary Error Function (erfc) <libcudacxx-extended-api-fp-fpmp-spec-complementary-error-function-erfc>`
-  :ref:`Boys Function F0 (boys_f0) <libcudacxx-extended-api-fp-fpmp-spec-boys-function-f0-boys_f0>`
-  :ref:`Inverse Normal CDF (normcdfinv) <libcudacxx-extended-api-fp-fpmp-spec-inverse-normal-cdf-normcdfinv>`
-  :ref:`Floor (floor) <libcudacxx-extended-api-fp-fpmp-spec-floor-floor>`
-  :ref:`Ceiling (ceil) <libcudacxx-extended-api-fp-fpmp-spec-ceiling-ceil>`
-  :ref:`Round to Nearest (round) <libcudacxx-extended-api-fp-fpmp-spec-round-to-nearest-round>`
-  :ref:`Truncate (trunc) <libcudacxx-extended-api-fp-fpmp-spec-truncate-trunc>`

:ref:`Comparison Operations <libcudacxx-extended-api-fp-fpmp-spec-comparison-operations>`

-  :ref:`Equal (eq) <libcudacxx-extended-api-fp-fpmp-spec-equal-eq>`
-  :ref:`Not Equal (ne) <libcudacxx-extended-api-fp-fpmp-spec-not-equal-ne>`
-  :ref:`Less Than (lt) <libcudacxx-extended-api-fp-fpmp-spec-less-than-lt>`
-  :ref:`Less Than or Equal (le) <libcudacxx-extended-api-fp-fpmp-spec-less-than-or-equal-le>`
-  :ref:`Greater Than (gt) <libcudacxx-extended-api-fp-fpmp-spec-greater-than-gt>`
-  :ref:`Greater Than or Equal (ge) <libcudacxx-extended-api-fp-fpmp-spec-greater-than-or-equal-ge>`

:ref:`Type Conversions <libcudacxx-extended-api-fp-fpmp-spec-type-conversions>`

-  :ref:`To Int32 (mp2int) <libcudacxx-extended-api-fp-fpmp-spec-to-int32-mp2int>`
-  :ref:`To UInt32 (mp2uint) <libcudacxx-extended-api-fp-fpmp-spec-to-uint32-mp2uint>`
-  :ref:`To Int64 (mp2ll) <libcudacxx-extended-api-fp-fpmp-spec-to-int64-mp2ll>`
-  :ref:`To UInt64 (mp2ull) <libcudacxx-extended-api-fp-fpmp-spec-to-uint64-mp2ull>`
-  :ref:`From Int32 (int2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-int32-int2mp>`
-  :ref:`From UInt32 (uint2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-uint32-uint2mp>`
-  :ref:`From Int64 (ll2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-int64-ll2mp>`
-  :ref:`From UInt64 (ull2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-uint64-ull2mp>`
-  :ref:`To Native Float (mp2fp) <libcudacxx-extended-api-fp-fpmp-spec-to-native-float-mp2fp>`
-  :ref:`From Native Float (fp2mp) <libcudacxx-extended-api-fp-fpmp-spec-from-native-float-fp2mp>`

:ref:`Other Functions <libcudacxx-extended-api-fp-fpmp-spec-other-functions>`

-  :ref:`ACOS <libcudacxx-extended-api-fp-fpmp-spec-acos>`
-  :ref:`ACOSH <libcudacxx-extended-api-fp-fpmp-spec-acosh>`
-  :ref:`ASIN <libcudacxx-extended-api-fp-fpmp-spec-asin>`
-  :ref:`ASINH <libcudacxx-extended-api-fp-fpmp-spec-asinh>`
-  :ref:`ATAN <libcudacxx-extended-api-fp-fpmp-spec-atan>`
-  :ref:`ATAN2 <libcudacxx-extended-api-fp-fpmp-spec-atan2>`
-  :ref:`ATANH <libcudacxx-extended-api-fp-fpmp-spec-atanh>`
-  :ref:`COSH <libcudacxx-extended-api-fp-fpmp-spec-cosh>`
-  :ref:`EXP10 <libcudacxx-extended-api-fp-fpmp-spec-exp10>`
-  :ref:`EXP2 <libcudacxx-extended-api-fp-fpmp-spec-exp2>`
-  :ref:`EXPM1 <libcudacxx-extended-api-fp-fpmp-spec-expm1>`
-  :ref:`FMOD <libcudacxx-extended-api-fp-fpmp-spec-fmod>`
-  :ref:`FREXP <libcudacxx-extended-api-fp-fpmp-spec-frexp>`
-  :ref:`LDEXP <libcudacxx-extended-api-fp-fpmp-spec-ldexp>`
-  :ref:`LOG10 <libcudacxx-extended-api-fp-fpmp-spec-log10>`
-  :ref:`LOG1P <libcudacxx-extended-api-fp-fpmp-spec-log1p>`
-  :ref:`LOG2 <libcudacxx-extended-api-fp-fpmp-spec-log2>`
-  :ref:`REMAINDER <libcudacxx-extended-api-fp-fpmp-spec-remainder>`
-  :ref:`SCALBLN <libcudacxx-extended-api-fp-fpmp-spec-scalbln>`
-  :ref:`SCALBN <libcudacxx-extended-api-fp-fpmp-spec-scalbn>`
-  :ref:`SINH <libcudacxx-extended-api-fp-fpmp-spec-sinh>`
-  :ref:`TAN <libcudacxx-extended-api-fp-fpmp-spec-tan>`

:ref:`Appendix: Legends <libcudacxx-extended-api-fp-fpmp-spec-appendix-legends>`

-  :ref:`Measured Accuracy Legend <libcudacxx-extended-api-fp-fpmp-spec-measured-accuracy-legend>`
-  :ref:`Special Values Legend (Floating Point) <libcudacxx-extended-api-fp-fpmp-spec-special-values-legend-floating-point>`
-  :ref:`Special Values Legend (Integer Conversions) <libcudacxx-extended-api-fp-fpmp-spec-special-values-legend-integer-conversions>`
-  :ref:`Performance Metrics Legend <libcudacxx-extended-api-fp-fpmp-spec-performance-metrics-legend>`
-  :ref:`SASS Instructions Legend <libcudacxx-extended-api-fp-fpmp-spec-sass-instructions-legend>`
-  :ref:`SASS Instructions Summary <libcudacxx-extended-api-fp-fpmp-spec-sass-instructions-summary>`

.. _libcudacxx-extended-api-fp-fpmp-spec-arithmetic-operations:

Arithmetic Operations
---------------------

.. _libcudacxx-extended-api-fp-fpmp-spec-addition-add:

Addition (add)
~~~~~~~~~~~~~~

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

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
cancellation    14         3e-07%  1.27e-08   4.27e-09   26
unclassified    133269     3e-03%  1.21e-09   9.65e-13   29
TOTAL           4261302295 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 5180.9       | -0.23x  | 6.82x   | 3477.8       | -0.23x  | 6.97x   | 3298.5 | -0.23x  | -0.44x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 11.34        | -0.23x  | 6.82x   | 11.56        | -0.23x  | 6.97x   | 11.34  | -0.23x  | -0.44x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 20.6         | -0.61x  | 5.15x   | 21.2         | -0.62x  | 5.00x   | 20.7   | -0.62x  | 1.04x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      8
fp64      0
other     1
**total** **9**
========= =====

*Accuracy: ``def``*

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

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 4175.1       | -0.19x  | 5.50x   | 2782.7       | -0.18x  | 5.58x   | 2646.5 | -0.19x  | -0.35x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 9.14         | -0.19x  | 5.50x   | 9.25         | -0.18x  | 5.58x   | 9.10   | -0.19x  | -0.35x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 38.4         | -0.31x  | 2.78x   | 39.3         | -0.33x  | 2.70x   | 38.7   | -0.34x  | -0.56x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      11
fp64      0
other     0
**total** **11**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156129 100.00% 7.11e-15   1.06e-16   47
TOTAL       4261156129 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2501.7       | -0.11x  | 3.30x   | 1645.7       | -0.11x  | 3.30x   | 1579.1 | -0.11x  | -0.21x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 5.48         | -0.11x  | 3.30x   | 5.47         | -0.11x  | 3.30x   | 5.43   | -0.11x  | -0.21x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 56.5         | -0.23x  | 1.88x   | 57.1         | -0.23x  | 1.86x   | 57.4   | -0.23x  | -0.37x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      20
fp64      0
other     0
**total** **20**
========= ======

**Special Values Table:**

+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **a\\b**  | -INF | -maxN    | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN   | +INF | QNAN |
+===========+======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=========+======+======+
| **-INF**  | -inf | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf    | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-maxN** | -inf | -inf     | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | +0      | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-1**    | -inf | -3.4e+38 | -2       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | +0       | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-minN** | -inf | -3.4e+38 | -1       | -2.4e-38 | -2.4e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.4e-45 | +0       | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-maxD** | -inf | -3.4e+38 | -1       | -2.4e-38 | -2.4e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | +0       | 1.4e-45  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-minD** | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -2.8e-45 | -1.4e-45 | -1.4e-45 | +0       | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-0**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | -0       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+0**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | +0       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+minD** | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | +0       | 1.4e-45  | 1.4e-45  | 2.8e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+maxD** | -inf | -3.4e+38 | -1       | -1.4e-45 | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | 2.4e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+minN** | -inf | -3.4e+38 | -1       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | 2.4e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+1**    | -inf | -3.4e+38 | +0       | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+maxN** | -inf | +0       | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf    | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+INF**  | nan  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf    | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **QNAN**  | nan  | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan     | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

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

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 95.1         | -0.13x  | -0.44x   | 62.5         | -0.13x  | -0.34x   | 1783.6 | -0.24x  | 10.18x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.21         | -0.13x  | -0.44x   | 0.21         | -0.13x  | -0.34x   | 6.13   | -0.24x  | 10.18x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 570.3        | -0.19x  | -0.31x   | 563.4        | -0.19x  | -0.29x   | 39.7   | -0.54x  | 4.14x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      8
other     4
**total** **12**
========= ======

*Accuracy: ``def``*

**Measured Accuracy:**

============ ======== ======= ========== ========== ====
Class        Count    Percent Max RelErr Avg RelErr Bits
============ ======== ======= ========== ========== ====
normal (OK)  16760813 100.00% 9.54e-29   7.62e-36   93
unclassified 1        6e-06%  2.31e-27   2.31e-27   88
TOTAL        16760814 100.00%
============ ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 69.2         | -0.09x  | -0.32x   | 45.5         | -0.09x  | -0.24x   | 1343.5 | -0.18x  | 7.68x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.15         | -0.09x  | -0.32x   | 0.15         | -0.09x  | -0.24x   | 4.62   | -0.18x  | 7.68x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 749.7        | -0.14x  | -0.24x   | 786.9        | -0.14x  | -0.21x   | 75.5   | -0.28x  | 2.18x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      11
other     0
**total** **11**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760814 100.00% 2.46e-32   -2.94e-37  105
TOTAL       16760814 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 38.1         | -0.05x  | -0.18x   | 25.0         | -0.05x  | -0.13x   | 813.0 | -0.11x  | 4.68x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.08         | -0.05x  | -0.18x   | 0.08         | -0.05x  | -0.13x   | 2.80  | -0.11x  | 4.68x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 1368.3       | -0.08x  | -0.13x   | 1342.0       | -0.08x  | -0.12x   | 107.4 | -0.20x  | 1.53x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      20
other     0
**total** **20**
========= ======

**Special Values Table:**

+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **a\\b**  | -INF | -maxN     | -1        | -minN     | -maxD     | -minD     | -0        | +0        | +minD     | +maxD     | +minN     | +1        | +maxN    | +INF | QNAN |
+===========+======+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+==========+======+======+
| **-INF**  | -inf | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf     | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-maxN** | -inf | -inf      | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | +0       | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-1**    | -inf | -1.8e+308 | -2        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | +0        | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-minN** | -inf | -1.8e+308 | -1        | -4.5e-308 | -4.5e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.9e-324 | +0        | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-maxD** | -inf | -1.8e+308 | -1        | -4.5e-308 | -4.5e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | +0        | 4.9e-324  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-minD** | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -9.9e-324 | -4.9e-324 | -4.9e-324 | +0        | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-0**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | -0        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+0**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | +0        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+minD** | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | +0        | 4.9e-324  | 4.9e-324  | 9.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+maxD** | -inf | -1.8e+308 | -1        | -4.9e-324 | +0        | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+minN** | -inf | -1.8e+308 | -1        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+1**    | -inf | -1.8e+308 | +0        | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 2         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+maxN** | -inf | +0        | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | +inf     | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+INF**  | nan  | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf     | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **QNAN**  | nan  | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan      | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-subtraction-sub:

Subtraction (sub)
~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-1:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-1:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     4261021649 99.99%  1.00e-13   1.64e-16   43
output special  131026     3e-03%  --         --         --
input special   5          1e-07%  --         --         --
output denormal 15041      4e-04%  0.00e+00   0.00e+00   0
input denormal  19         4e-07%  4.12e-13   1.76e-13   41
input near inf  1170       3e-05%  1.53e-10   9.65e-13   32
cancellation    12         3e-07%  4.97e-08   9.45e-09   24
unclassified    133285     3e-03%  1.13e-09   9.52e-13   29
TOTAL           4261302207 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 5187.6       | -0.23x  | 6.83x   | 3473.4       | -0.23x  | 6.96x   | 3298.7 | -0.23x  | -0.44x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 11.36        | -0.23x  | 6.83x   | 11.55        | -0.23x  | 6.96x   | 11.34  | -0.23x  | -0.44x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 20.2         | -0.61x  | 5.25x   | 21.2         | -0.62x  | 5.00x   | 21.0   | -0.63x  | 1.01x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      8
fp64      0
other     1
**total** **9**
========= =====

*Accuracy: ``def``*

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

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 4177.3       | -0.19x  | 5.50x   | 2783.6       | -0.18x  | 5.58x   | 2652.4 | -0.19x  | -0.35x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 9.14         | -0.19x  | 5.50x   | 9.26         | -0.18x  | 5.58x   | 9.12   | -0.19x  | -0.35x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 38.5         | -0.32x  | 2.77x   | 39.3         | -0.33x  | 2.70x   | 38.8   | -0.33x  | -0.56x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      11
fp64      0
other     0
**total** **11**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156134 100.00% 7.11e-15   1.06e-16   47
TOTAL       4261156134 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2500.1       | -0.11x  | 3.29x   | 1643.6       | -0.11x  | 3.29x   | 1579.3 | -0.11x  | -0.21x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 5.47         | -0.11x  | 3.29x   | 5.47         | -0.11x  | 3.29x   | 5.43   | -0.11x  | -0.21x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 56.7         | -0.22x  | 1.87x   | 57.5         | -0.23x  | 1.85x   | 57.0   | -0.22x  | -0.38x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      20
fp64      0
other     0
**total** **20**
========= ======

**Special Values Table:**

+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **a\\b**  | -INF | -maxN   | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN    | +INF | QNAN |
+===========+======+=========+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+======+======+
| **-INF**  | nan  | -inf    | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-maxN** | +inf | +0      | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -inf     | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-1**    | +inf | 3.4e+38 | +0       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -2       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-minN** | +inf | 3.4e+38 | 1        | +0       | -1.4e-45 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -2.4e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-maxD** | +inf | 3.4e+38 | 1        | 1.4e-45  | +0       | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -2.4e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-minD** | +inf | 3.4e+38 | 1        | 1.2e-38  | 1.2e-38  | +0       | -1.4e-45 | -1.4e-45 | -2.8e-45 | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **-0**    | +inf | 3.4e+38 | 1        | 1.2e-38  | 1.2e-38  | 1.4e-45  | +0       | -0       | -1.4e-45 | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+0**    | +inf | 3.4e+38 | 1        | 1.2e-38  | 1.2e-38  | 1.4e-45  | +0       | +0       | -1.4e-45 | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+minD** | +inf | 3.4e+38 | 1        | 1.2e-38  | 1.2e-38  | 2.8e-45  | 1.4e-45  | 1.4e-45  | +0       | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+maxD** | +inf | 3.4e+38 | 1        | 2.4e-38  | 2.4e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | +0       | -1.4e-45 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+minN** | +inf | 3.4e+38 | 1        | 2.4e-38  | 2.4e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.4e-45  | +0       | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+1**    | +inf | 3.4e+38 | 2        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | +0       | -3.4e+38 | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+maxN** | +inf | +inf    | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +0       | -inf | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **+INF**  | +inf | +inf    | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | nan  | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+
| **QNAN**  | nan  | nan     | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan  | nan  |
+-----------+------+---------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+------+------+

.. _type-fp64mp2-1:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-1:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

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

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 95.1         | -0.13x  | -0.84x   | 62.5         | -0.13x  | -0.55x   | 1777.2 | -0.24x  | 15.95x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.21         | -0.13x  | -0.84x   | 0.21         | -0.13x  | -0.55x   | 6.11   | -0.24x  | 15.95x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 570.0        | -0.19x  | -0.70x   | 563.1        | -0.19x  | -0.68x   | 40.1   | -0.53x  | 9.45x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      8
other     4
**total** **12**
========= ======

*Accuracy: ``def``*

**Measured Accuracy:**

============ ======== ======= ========== ========== ====
Class        Count    Percent Max RelErr Avg RelErr Bits
============ ======== ======= ========== ========== ====
normal (OK)  16760814 100.00% 8.49e-29   -2.46e-36  93
unclassified 1        6e-06%  1.45e-28   1.45e-28   92
TOTAL        16760815 100.00%
============ ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 69.2         | -0.09x  | -0.61x   | 45.5         | -0.09x  | -0.40x   | 1342.6 | -0.18x  | 12.05x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.15         | -0.09x  | -0.61x   | 0.15         | -0.09x  | -0.40x   | 4.62   | -0.18x  | 12.05x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 749.9        | -0.14x  | -0.53x   | 787.3        | -0.14x  | -0.48x   | 75.6   | -0.28x  | 5.01x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      11
other     0
**total** **11**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760815 100.00% 2.47e-32   -8.39e-38  105
TOTAL       16760815 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 38.1         | -0.05x  | -0.34x   | 25.0         | -0.05x  | -0.22x   | 812.7 | -0.11x  | 7.30x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.08         | -0.05x  | -0.34x   | 0.08         | -0.05x  | -0.22x   | 2.79  | -0.11x  | 7.30x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 1368.5       | -0.08x  | -0.29x   | 1341.9       | -0.08x  | -0.28x   | 107.3 | -0.20x  | 3.53x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      20
other     0
**total** **20**
========= ======

**Special Values Table:**

+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **a\\b**  | -INF | -maxN    | -1        | -minN     | -maxD     | -minD     | -0        | +0        | +minD     | +maxD     | +minN     | +1        | +maxN     | +INF | QNAN |
+===========+======+==========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+======+======+
| **-INF**  | nan  | -inf     | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-maxN** | +inf | +0       | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -inf      | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-1**    | +inf | 1.8e+308 | +0        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -2        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-minN** | +inf | 1.8e+308 | 1         | +0        | -4.9e-324 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.5e-308 | -4.5e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-maxD** | +inf | 1.8e+308 | 1         | 4.9e-324  | +0        | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.5e-308 | -4.5e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-minD** | +inf | 1.8e+308 | 1         | 2.2e-308  | 2.2e-308  | +0        | -4.9e-324 | -4.9e-324 | -9.9e-324 | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **-0**    | +inf | 1.8e+308 | 1         | 2.2e-308  | 2.2e-308  | 4.9e-324  | +0        | -0        | -4.9e-324 | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+0**    | +inf | 1.8e+308 | 1         | 2.2e-308  | 2.2e-308  | 4.9e-324  | +0        | +0        | -4.9e-324 | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+minD** | +inf | 1.8e+308 | 1         | 2.2e-308  | 2.2e-308  | 9.9e-324  | 4.9e-324  | 4.9e-324  | +0        | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+maxD** | +inf | 1.8e+308 | 1         | 4.5e-308  | 4.5e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | +0        | -4.9e-324 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+minN** | +inf | 1.8e+308 | 1         | 4.5e-308  | 4.5e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.9e-324  | +0        | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+1**    | +inf | 1.8e+308 | 2         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | +0        | -1.8e+308 | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+maxN** | +inf | +inf     | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | +0        | -inf | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **+INF**  | +inf | +inf     | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | nan  | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+
| **QNAN**  | nan  | nan      | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan  | nan  |
+-----------+------+----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-multiplication-mul:

Multiplication (mul)
~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-2:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-2:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3127081961 83.86%  1.00e-13   1.31e-15   43
output special       537824645  14.42%  --         --         --
input special        5          1e-07%  --         --         --
output denormal      3569762    0.10%   3.33e-01   1.68e-06   1
input denormal       20988831   0.56%   5.96e-08   4.70e-09   24
output near denormal 3219189    0.09%   1.19e-07   8.44e-08   23
cancellation         36315300   0.97%   5.96e-08   4.23e-09   24
TOTAL                3728999693 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 6335.2       | -0.29x  | 8.35x   | 4069.1       | -0.27x  | 8.16x   | 3847.8 | -0.27x  | -0.52x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 13.87        | -0.29x  | 8.35x   | 13.53        | -0.27x  | 8.16x   | 13.23  | -0.27x  | -0.52x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 20.8         | -0.62x  | 5.12x   | 21.5         | -0.61x  | 4.94x   | 21.4   | -0.61x  | 1.01x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      5
fp64      0
other     1
**total** **6**
========= =====

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3127081985 97.99%  1.00e-13   1.68e-15   43
output special       1          3e-08%  --         --         --
output denormal      3569762    0.11%   3.33e-01   1.68e-06   1
input denormal       20988807   0.66%   5.96e-08   4.70e-09   24
output near denormal 3219189    0.10%   1.19e-07   8.44e-08   23
cancellation         36315300   1.14%   5.96e-08   4.23e-09   24
TOTAL                3191175044 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 4615.5       | -0.21x  | 6.08x   | 3098.6       | -0.21x  | 6.21x   | 2942.1 | -0.21x  | -0.40x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 10.10        | -0.21x  | 6.08x   | 10.30        | -0.21x  | 6.21x   | 10.12  | -0.21x  | -0.40x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 33.4         | -0.36x  | 3.18x   | 34.0         | -0.38x  | 3.12x   | 34.0   | -0.39x  | -0.62x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      9
fp64      0
other     0
**total** **9**
========= =====

*Accuracy: ``high``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3127081985 97.99%  1.00e-13   1.68e-15   43
output special       1          3e-08%  --         --         --
output denormal      3569762    0.11%   3.33e-01   1.68e-06   1
input denormal       20988807   0.66%   5.96e-08   4.70e-09   24
output near denormal 3219189    0.10%   1.19e-07   8.44e-08   23
cancellation         36315300   1.14%   5.96e-08   4.23e-09   24
TOTAL                3191175044 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 4631.6       | -0.21x  | 6.10x   | 3093.9       | -0.21x  | 6.20x   | 2942.7 | -0.21x  | -0.40x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 10.14        | -0.21x  | 6.10x   | 10.29        | -0.21x  | 6.20x   | 10.12  | -0.21x  | -0.40x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 33.3         | -0.38x  | 3.19x   | 34.1         | -0.39x  | 3.11x   | 33.9   | -0.39x  | -0.64x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      9
fp64      0
other     0
**total** **9**
========= =====

**Special Values Table:**

+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **a\\b**  | -INF | -maxN    | -1       | -minN    | -maxD    | -minD    | -0  | +0  | +minD    | +maxD    | +minN    | +1       | +maxN    | +INF | QNAN |
+===========+======+==========+==========+==========+==========+==========+=====+=====+==========+==========+==========+==========+==========+======+======+
| **-INF**  | +inf | +inf     | +inf     | +inf     | +inf     | +inf     | nan | nan | -inf     | -inf     | -inf     | -inf     | -inf     | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-maxN** | +inf | +inf     | 3.4e+38  | 4        | 4        | 4.8e-07  | +0  | -0  | -4.8e-07 | -4       | -4       | -3.4e+38 | -inf     | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-1**    | +inf | 3.4e+38  | 1        | 1.2e-38  | 1.2e-38  | 1.4e-45  | +0  | -0  | -1.4e-45 | -1.2e-38 | -1.2e-38 | -1       | -3.4e+38 | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-minN** | +inf | 4        | 1.2e-38  | +0       | +0       | +0       | +0  | -0  | -0       | -0       | -0       | -1.2e-38 | -4       | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-maxD** | +inf | 4        | 1.2e-38  | +0       | +0       | +0       | +0  | -0  | -0       | -0       | -0       | -1.2e-38 | -4       | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-minD** | +inf | 4.8e-07  | 1.4e-45  | +0       | +0       | +0       | +0  | -0  | -0       | -0       | -0       | -1.4e-45 | -4.8e-07 | -inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **-0**    | nan  | +0       | +0       | +0       | +0       | +0       | +0  | -0  | -0       | -0       | -0       | -0       | -0       | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+0**    | nan  | -0       | -0       | -0       | -0       | -0       | -0  | +0  | +0       | +0       | +0       | +0       | +0       | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+minD** | -inf | -4.8e-07 | -1.4e-45 | -0       | -0       | -0       | -0  | +0  | +0       | +0       | +0       | 1.4e-45  | 4.8e-07  | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+maxD** | -inf | -4       | -1.2e-38 | -0       | -0       | -0       | -0  | +0  | +0       | +0       | +0       | 1.2e-38  | 4        | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+minN** | -inf | -4       | -1.2e-38 | -0       | -0       | -0       | -0  | +0  | +0       | +0       | +0       | 1.2e-38  | 4        | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+1**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | -0  | +0  | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38  | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+maxN** | -inf | -inf     | -3.4e+38 | -4       | -4       | -4.8e-07 | -0  | +0  | 4.8e-07  | 4        | 4        | 3.4e+38  | +inf     | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **+INF**  | -inf | -inf     | -inf     | -inf     | -inf     | -inf     | nan | nan | +inf     | +inf     | +inf     | +inf     | +inf     | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+
| **QNAN**  | nan  | nan      | nan      | nan      | nan      | nan      | nan | nan | nan      | nan      | nan      | nan      | nan      | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+-----+-----+----------+----------+----------+----------+----------+------+------+

.. _type-fp64mp2-2:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-2:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12537828 85.50%  9.99e-29   -2.40e-20  93
output special       2095104  14.29%  --         --         --
output denormal      1757     0.01%   2.05e-13   2.49e-16   42
input denormal       2850     0.02%   1.10e-16   4.54e-18   53
output near denormal 871      6e-03%  2.22e-16   1.97e-16   52
cancellation         24953    0.17%   1.11e-16   4.57e-18   53
TOTAL                14663363 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 162.4        | -0.21x  | -0.80x   | 106.7        | -0.21x  | -0.76x   | 2173.4 | -0.30x  | 16.01x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.36         | -0.21x  | -0.80x   | 0.35         | -0.21x  | -0.76x   | 7.47   | -0.30x  | 16.01x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 382.1        | -0.28x  | -0.55x   | 389.1        | -0.27x  | -0.51x   | 43.1   | -0.49x  | 4.61x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      5
other     2
**total** **7**
========= =====

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12537828 99.76%  9.99e-29   -2.40e-20  93
output denormal      1757     0.01%   2.05e-13   2.49e-16   42
input denormal       2850     0.02%   1.10e-16   4.54e-18   53
output near denormal 871      7e-03%  2.22e-16   1.97e-16   52
cancellation         24953    0.20%   1.11e-16   4.57e-18   53
TOTAL                12568259 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 85.8         | -0.11x  | -0.43x   | 56.4         | -0.11x  | -0.40x   | 1536.2 | -0.21x  | 11.33x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.19         | -0.11x  | -0.43x   | 0.19         | -0.11x  | -0.40x   | 5.28   | -0.21x  | 11.33x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 634.9        | -0.17x  | -0.33x   | 659.2        | -0.16x  | -0.30x   | 64.3   | -0.33x  | 3.08x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      9
other     0
**total** **9**
========= =====

*Accuracy: ``high``*

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12537828 99.76%  9.99e-29   -2.40e-20  93
output denormal      1757     0.01%   2.05e-13   2.49e-16   42
input denormal       2850     0.02%   1.10e-16   4.54e-18   53
output near denormal 871      7e-03%  2.22e-16   1.97e-16   52
cancellation         24953    0.20%   1.11e-16   4.57e-18   53
TOTAL                12568259 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 85.8         | -0.11x  | -0.42x   | 56.4         | -0.11x  | -0.40x   | 1541.8 | -0.21x  | 11.34x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.19         | -0.11x  | -0.42x   | 0.19         | -0.11x  | -0.40x   | 5.30   | -0.21x  | 11.34x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 635.0        | -0.17x  | -0.33x   | 659.2        | -0.16x  | -0.30x   | 64.2   | -0.33x  | 3.09x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      9
other     0
**total** **9**
========= =====

**Special Values Table:**

+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **a\\b**  | -INF | -maxN     | -1        | -minN     | -maxD     | -minD     | -0  | +0  | +minD     | +maxD     | +minN     | +1        | +maxN     | +INF | QNAN |
+===========+======+===========+===========+===========+===========+===========+=====+=====+===========+===========+===========+===========+===========+======+======+
| **-INF**  | +inf | +inf      | +inf      | +inf      | +inf      | +inf      | nan | nan | -inf      | -inf      | -inf      | -inf      | -inf      | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-maxN** | +inf | +inf      | 1.8e+308  | 4         | 4         | 8.9e-16   | +0  | -0  | -8.9e-16  | -4        | -4        | -1.8e+308 | -inf      | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-1**    | +inf | 1.8e+308  | 1         | 2.2e-308  | 2.2e-308  | 4.9e-324  | +0  | -0  | -4.9e-324 | -2.2e-308 | -2.2e-308 | -1        | -1.8e+308 | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-minN** | +inf | 4         | 2.2e-308  | +0        | +0        | +0        | +0  | -0  | -0        | -0        | -0        | -2.2e-308 | -4        | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-maxD** | +inf | 4         | 2.2e-308  | +0        | +0        | +0        | +0  | -0  | -0        | -0        | -0        | -2.2e-308 | -4        | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-minD** | +inf | 8.9e-16   | 4.9e-324  | +0        | +0        | +0        | +0  | -0  | -0        | -0        | -0        | -4.9e-324 | -8.9e-16  | -inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **-0**    | nan  | +0        | +0        | +0        | +0        | +0        | +0  | -0  | -0        | -0        | -0        | -0        | -0        | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+0**    | nan  | -0        | -0        | -0        | -0        | -0        | -0  | +0  | +0        | +0        | +0        | +0        | +0        | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+minD** | -inf | -8.9e-16  | -4.9e-324 | -0        | -0        | -0        | -0  | +0  | +0        | +0        | +0        | 4.9e-324  | 8.9e-16   | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+maxD** | -inf | -4        | -2.2e-308 | -0        | -0        | -0        | -0  | +0  | +0        | +0        | +0        | 2.2e-308  | 4         | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+minN** | -inf | -4        | -2.2e-308 | -0        | -0        | -0        | -0  | +0  | +0        | +0        | +0        | 2.2e-308  | 4         | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+1**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | -0  | +0  | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308  | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+maxN** | -inf | -inf      | -1.8e+308 | -4        | -4        | -8.9e-16  | -0  | +0  | 8.9e-16   | 4         | 4         | 1.8e+308  | +inf      | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **+INF**  | -inf | -inf      | -inf      | -inf      | -inf      | -inf      | nan | nan | +inf      | +inf      | +inf      | +inf      | +inf      | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+
| **QNAN**  | nan  | nan       | nan       | nan       | nan       | nan       | nan | nan | nan       | nan       | nan       | nan       | nan       | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----+-----+-----------+-----------+-----------+-----------+-----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-division-div:

Division (div)
~~~~~~~~~~~~~~

.. _type-fp32mp2-3:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-3:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3006532757 80.74%  1.00e-13   2.85e-15   43
output special       536936544  14.42%  --         --         --
input special        5          1e-07%  --         --         --
output denormal      7612929    0.20%   1.00e+00   4.13e-01   0
input denormal       155637297  4.18%   1.79e-07   5.24e-09   22
output near denormal 208307     6e-03%  1.00e+00   6.29e-01   0
input near inf       16515092   0.44%   1.00e+00   1.00e+00   0
cancellation         290372     8e-03%  1.85e-08   1.14e-12   25
TOTAL                3723733303 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 6473.6       | 1.93x   | 61.33x  | 4411.4       | 1.95x   | 63.60x  | 4169.8 | 2.05x   | 3.44x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 14.17        | 1.93x   | 61.33x  | 14.67        | 1.95x   | 63.60x  | 14.34  | 2.05x   | 3.44x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 20.8         | 2.90x   | 27.51x  | 21.7         | 2.80x   | 26.22x  | 21.4   | 2.86x   | 6.14x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      6
fp64      0
other     1
**total** **7**
========= =====

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3006485741 94.22%  1.00e-13   2.08e-15   43
output special       8355823    0.26%   --         --         --
input special        1          3e-08%  --         --         --
output denormal      3233022    0.10%   1.00e+00   9.73e-01   0
input denormal       155782543  4.88%   1.79e-07   5.19e-09   22
output near denormal 207877     7e-03%  1.00e+00   6.30e-01   0
input near inf       16515092   0.52%   1.00e+00   1.00e+00   0
cancellation         192572     6e-03%  1.26e-08   1.11e-12   26
TOTAL                3190772671 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2944.5       | -0.88x  | 27.90x  | 1988.4       | -0.88x  | 28.67x  | 1901.4 | -0.93x  | 1.56x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 6.45         | -0.88x  | 27.90x  | 6.61         | -0.88x  | 28.67x  | 6.54   | -0.93x  | 1.56x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 49.8         | 1.22x   | 11.47x  | 50.1         | 1.22x   | 11.36x  | 49.9   | 1.23x   | 2.64x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3187273516 93.42%  1.00e-13   1.70e-15   43
output special  222266516  6.51%   --         --         --
input special   1          3e-08%  --         --         --
output denormal 2097440    0.06%   1.00e+00   2.17e-06   0
input denormal  49841      1e-03%  1.25e-08   1.19e-12   26
input near inf  4317       1e-04%  2.03e-09   1.69e-12   28
cancellation    211974     6e-03%  1.26e-08   1.10e-12   26
TOTAL           3411903605 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2020.5       | -0.60x  | 19.14x  | 1264.5       | -0.56x  | 18.23x  | 1215.3 | -0.60x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 4.42         | -0.60x  | 19.14x  | 4.20         | -0.56x  | 18.23x  | 4.18   | -0.60x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 86.2         | -0.70x  | 6.63x   | 84.9         | -0.72x  | 6.70x   | 84.8   | -0.73x  | 1.55x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      23
fp64      0
other     23
**total** **46**
========= ======

**Special Values Table:**

+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **a\\b**  | -INF | -maxN    | -1       | -minN    | -maxD    | -minD | -0   | +0   | +minD | +maxD    | +minN    | +1       | +maxN    | +INF | QNAN |
+===========+======+==========+==========+==========+==========+=======+======+======+=======+==========+==========+==========+==========+======+======+
| **-INF**  | nan  | +inf     | +inf     | +inf     | +inf     | +inf  | +inf | -inf | -inf  | -inf     | -inf     | -inf     | -inf     | nan  | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-maxN** | +0   | 1        | 3.4e+38  | +inf     | +inf     | +inf  | +inf | -inf | -inf  | -inf     | -inf     | -3.4e+38 | -1       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-1**    | +0   | 2.9e-39  | 1        | 8.5e+37  | 8.5e+37  | +inf  | +inf | -inf | -inf  | -8.5e+37 | -8.5e+37 | -1       | -2.9e-39 | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-minN** | +0   | +0       | 1.2e-38  | 1        | 1        | +inf  | +inf | -inf | -inf  | -1       | -1       | -1.2e-38 | -0       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-maxD** | +0   | +0       | 1.2e-38  | 1        | 1        | +inf  | +inf | -inf | -inf  | -1       | -1       | -1.2e-38 | -0       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-minD** | +0   | +0       | 1.4e-45  | 1.2e-07  | 1.2e-07  | +inf  | +inf | -inf | -inf  | -1.2e-07 | -1.2e-07 | -1.4e-45 | -0       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **-0**    | +0   | +0       | +0       | +0       | +0       | nan   | nan  | nan  | nan   | -0       | -0       | -0       | -0       | -0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+0**    | -0   | -0       | -0       | -0       | -0       | nan   | nan  | nan  | nan   | +0       | +0       | +0       | +0       | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+minD** | -0   | -0       | -1.4e-45 | -1.2e-07 | -1.2e-07 | -inf  | -inf | +inf | +inf  | 1.2e-07  | 1.2e-07  | 1.4e-45  | +0       | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+maxD** | -0   | -0       | -1.2e-38 | -1       | -1       | -inf  | -inf | +inf | +inf  | 1        | 1        | 1.2e-38  | +0       | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+minN** | -0   | -0       | -1.2e-38 | -1       | -1       | -inf  | -inf | +inf | +inf  | 1        | 1        | 1.2e-38  | +0       | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+1**    | -0   | -2.9e-39 | -1       | -8.5e+37 | -8.5e+37 | -inf  | -inf | +inf | +inf  | 8.5e+37  | 8.5e+37  | 1        | 2.9e-39  | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+maxN** | -0   | -1       | -3.4e+38 | -inf     | -inf     | -inf  | -inf | +inf | +inf  | +inf     | +inf     | 3.4e+38  | 1        | +0   | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **+INF**  | nan  | -inf     | -inf     | -inf     | -inf     | -inf  | -inf | +inf | +inf  | +inf     | +inf     | +inf     | +inf     | nan  | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+
| **QNAN**  | nan  | nan      | nan      | nan      | nan      | nan   | nan  | nan  | nan   | nan      | nan      | nan      | nan      | nan  | nan  |
+-----------+------+----------+----------+----------+----------+-------+------+------+-------+----------+----------+----------+----------+------+------+

.. _type-fp64mp2-3:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-3:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12480461 85.12%  1.00e-28   -3.87e-20  93
output special       2097129  14.30%  --         --         --
output denormal      2577     0.02%   1.86e-15   1.02e-18   48
input denormal       81990    0.56%   2.32e-16   5.83e-18   51
output near denormal 2        1e-05%  1.21e-16   1.18e-16   52
TOTAL                14662159 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 157.0        | 1.49x   | 1.74x    | 103.2        | 1.49x   | 1.69x    | 2258.0 | 1.87x   | 37.54x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.34         | 1.49x   | 1.74x    | 0.34         | 1.49x   | 1.69x    | 7.76   | 1.87x   | 37.54x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 375.6        | 1.52x   | 1.37x    | 393.8        | 1.44x   | 1.25x    | 44.1   | 2.98x   | 11.02x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      1
fp64      11
other     27
**total** **39**
========= ======

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ======== ======= ========== ========== ====
Class                Count    Percent Max RelErr Avg RelErr Bits
==================== ======== ======= ========== ========== ====
normal (OK)          12480681 99.32%  1.00e-28   -3.89e-20  93
output special       4083     0.03%   --         --         --
output denormal      8        6e-05%  1.86e-15   3.27e-16   48
input denormal       81770    0.65%   2.32e-16   5.87e-18   51
output near denormal 2        2e-05%  1.21e-16   1.17e-16   52
TOTAL                12566544 100.00%
==================== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 67.9         | -0.64x  | -0.76x   | 44.6         | -0.64x  | -0.73x   | 1057.9 | -0.87x  | 17.59x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.15         | -0.64x  | -0.76x   | 0.15         | -0.64x  | -0.73x   | 3.64   | -0.87x  | 17.59x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 832.4        | -0.69x  | -0.62x   | 852.4        | -0.67x  | -0.58x   | 99.4   | 1.32x   | 4.90x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      1
fp64      18
other     25
**total** **44**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     12566536 98.79%  1.19e-29   -6.30e-33  96
output special  153482   1.21%   --         --         --
output denormal 229      2e-03%  2.26e-13   6.53e-16   42
TOTAL           12720247 100.00%
=============== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 39.9         | -0.38x  | -0.44x   | 26.2         | -0.38x  | -0.43x   | 584.4 | -0.48x  | 9.72x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.09         | -0.38x  | -0.44x   | 0.09         | -0.38x  | -0.43x   | 2.01  | -0.48x  | 9.72x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 1379.0       | -0.41x  | -0.37x   | 1386.6       | -0.41x  | -0.35x   | 173.3 | -0.76x  | 2.81x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      1
fp64      28
other     53
**total** **82**
========= ======

**Special Values Table:**

+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **a\\b**  | -INF | -maxN     | -1        | -minN     | -maxD     | -minD | -0   | +0   | +minD | +maxD     | +minN     | +1        | +maxN     | +INF | QNAN |
+===========+======+===========+===========+===========+===========+=======+======+======+=======+===========+===========+===========+===========+======+======+
| **-INF**  | nan  | +inf      | +inf      | +inf      | +inf      | +inf  | +inf | -inf | -inf  | -inf      | -inf      | -inf      | -inf      | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-maxN** | +0   | 1         | 1.8e+308  | +inf      | +inf      | +inf  | +inf | -inf | -inf  | -inf      | -inf      | -1.8e+308 | -1        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-1**    | +0   | 5.6e-309  | 1         | 4.5e+307  | 4.5e+307  | +inf  | +inf | -inf | -inf  | -4.5e+307 | -4.5e+307 | -1        | -5.6e-309 | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-minN** | +0   | +0        | 2.2e-308  | 1         | 1         | +inf  | +inf | -inf | -inf  | -1        | -1        | -2.2e-308 | -0        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-maxD** | +0   | +0        | 2.2e-308  | 1         | 1         | +inf  | +inf | -inf | -inf  | -1        | -1        | -2.2e-308 | -0        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-minD** | +0   | +0        | 4.9e-324  | 2.2e-16   | 2.2e-16   | +inf  | +inf | -inf | -inf  | -2.2e-16  | -2.2e-16  | -4.9e-324 | -0        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **-0**    | +0   | +0        | +0        | +0        | +0        | nan   | nan  | nan  | nan   | -0        | -0        | -0        | -0        | -0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+0**    | -0   | -0        | -0        | -0        | -0        | nan   | nan  | nan  | nan   | +0        | +0        | +0        | +0        | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+minD** | -0   | -0        | -4.9e-324 | -2.2e-16  | -2.2e-16  | -inf  | -inf | +inf | +inf  | 2.2e-16   | 2.2e-16   | 4.9e-324  | +0        | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+maxD** | -0   | -0        | -2.2e-308 | -1        | -1        | -inf  | -inf | +inf | +inf  | 1         | 1         | 2.2e-308  | +0        | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+minN** | -0   | -0        | -2.2e-308 | -1        | -1        | -inf  | -inf | +inf | +inf  | 1         | 1         | 2.2e-308  | +0        | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+1**    | -0   | -5.6e-309 | -1        | -4.5e+307 | -4.5e+307 | -inf  | -inf | +inf | +inf  | 4.5e+307  | 4.5e+307  | 1         | 5.6e-309  | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+maxN** | -0   | -1        | -1.8e+308 | -inf      | -inf      | -inf  | -inf | +inf | +inf  | +inf      | +inf      | 1.8e+308  | 1         | +0   | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **+INF**  | nan  | -inf      | -inf      | -inf      | -inf      | -inf  | -inf | +inf | +inf  | +inf      | +inf      | +inf      | +inf      | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+
| **QNAN**  | nan  | nan       | nan       | nan       | nan       | nan   | nan  | nan  | nan   | nan       | nan       | nan       | nan       | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-------+------+------+-------+-----------+-----------+-----------+-----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-accumulate-acc:

Accumulate (acc)
~~~~~~~~~~~~~~~~

.. _type-fp32mp2-4:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-4:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

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

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 5428.9       | -0.25x  | 7.15x   | 3638.6       | -0.24x  | 7.30x   | 3463.9 | -0.25x  | -0.47x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 11.88        | -0.25x  | 7.15x   | 12.10        | -0.24x  | 7.30x   | 11.91  | -0.25x  | -0.47x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 16.6         | -0.76x  | 6.41x   | 17.2         | -0.75x  | 6.19x   | 17.0   | -0.75x  | 1.25x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      7
fp64      0
other     1
**total** **8**
========= =====

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156129 100.00% 7.10e-15   5.63e-17   47
TOTAL       4261156129 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 4475.1       | -0.20x  | 5.89x   | 2998.1       | -0.20x  | 6.01x   | 2853.0 | -0.20x  | -0.38x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 9.80         | -0.20x  | 5.89x   | 9.97         | -0.20x  | 6.01x   | 9.81   | -0.20x  | -0.38x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 38.5         | -0.33x  | 2.76x   | 39.0         | -0.34x  | 2.72x   | 39.0   | -0.33x  | -0.54x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      10
fp64      0
other     0
**total** **10**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261156129 100.00% 7.10e-15   5.63e-17   47
TOTAL       4261156129 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 3471.2       | -0.16x  | 4.57x   | 2316.2       | -0.15x  | 4.64x   | 2213.0 | -0.16x  | -0.30x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 7.60         | -0.16x  | 4.57x   | 7.70         | -0.15x  | 4.64x   | 7.61   | -0.16x  | -0.30x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 50.6         | -0.25x  | 2.10x   | 51.2         | -0.26x  | 2.08x   | 51.0   | -0.26x  | -0.42x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======

**Special Values Table:**

+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **a\\b**  | -INF | -maxN    | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN   | +INF | QNAN |
+===========+======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=========+======+======+
| **-INF**  | -inf | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf    | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-maxN** | -inf | -inf     | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | +0      | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-1**    | -inf | -3.4e+38 | -2       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | +0       | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-minN** | -inf | -3.4e+38 | -1       | -2.4e-38 | -2.4e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.4e-45 | +0       | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-maxD** | -inf | -3.4e+38 | -1       | -2.4e-38 | -2.4e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | +0       | 1.4e-45  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-minD** | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -2.8e-45 | -1.4e-45 | -1.4e-45 | +0       | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **-0**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | -0       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+0**    | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | -1.4e-45 | +0       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+minD** | -inf | -3.4e+38 | -1       | -1.2e-38 | -1.2e-38 | +0       | 1.4e-45  | 1.4e-45  | 2.8e-45  | 1.2e-38  | 1.2e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+maxD** | -inf | -3.4e+38 | -1       | -1.4e-45 | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | 2.4e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+minN** | -inf | -3.4e+38 | -1       | +0       | 1.4e-45  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | 2.4e-38  | 1        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+1**    | -inf | -3.4e+38 | +0       | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | 3.4e+38 | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+maxN** | -inf | +0       | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf    | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **+INF**  | nan  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf    | +inf | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+
| **QNAN**  | nan  | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan     | nan  | nan  |
+-----------+------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+---------+------+------+

.. _type-fp64mp2-4:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-4:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

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

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 108.7        | -0.14x  | -0.50x   | 71.5         | -0.14x  | -0.38x   | 1924.5 | -0.26x  | 11.14x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.24         | -0.14x  | -0.50x   | 0.24         | -0.14x  | -0.38x   | 6.62   | -0.26x  | 11.14x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 499.3        | -0.21x  | -0.35x   | 509.3        | -0.21x  | -0.33x   | 34.5   | -0.62x  | 4.76x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      7
other     2
**total** **9**
========= =====

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760814 100.00% 1.32e-32   -3.35e-38  105
TOTAL       16760814 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 76.1         | -0.10x  | -0.35x   | 50.0         | -0.10x  | -0.27x   | 1448.7 | -0.20x  | 8.23x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.17         | -0.10x  | -0.35x   | 0.17         | -0.10x  | -0.27x   | 4.98   | -0.20x  | 8.23x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 695.0        | -0.15x  | -0.25x   | 723.0        | -0.15x  | -0.23x   | 75.4   | -0.28x  | 2.21x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      10
other     0
**total** **10**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760814 100.00% 1.32e-32   -3.35e-38  105
TOTAL       16760814 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 58.5         | -0.08x  | -0.27x   | 38.5         | -0.08x  | -0.21x   | 1170.6 | -0.16x  | 6.78x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.13         | -0.08x  | -0.27x   | 0.13         | -0.08x  | -0.21x   | 4.03   | -0.16x  | 6.78x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 918.8        | -0.12x  | -0.19x   | 915.5        | -0.12x  | -0.18x   | 99.2   | -0.22x  | 1.66x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      13
other     0
**total** **13**
========= ======

**Special Values Table:**

+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **a\\b**  | -INF | -maxN     | -1        | -minN     | -maxD     | -minD     | -0        | +0        | +minD     | +maxD     | +minN     | +1        | +maxN    | +INF | QNAN |
+===========+======+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+==========+======+======+
| **-INF**  | -inf | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf     | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-maxN** | -inf | -inf      | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | +0       | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-1**    | -inf | -1.8e+308 | -2        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | +0        | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-minN** | -inf | -1.8e+308 | -1        | -4.5e-308 | -4.5e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.9e-324 | +0        | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-maxD** | -inf | -1.8e+308 | -1        | -4.5e-308 | -4.5e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | +0        | 4.9e-324  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-minD** | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -9.9e-324 | -4.9e-324 | -4.9e-324 | +0        | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **-0**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | -0        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+0**    | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | -4.9e-324 | +0        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+minD** | -inf | -1.8e+308 | -1        | -2.2e-308 | -2.2e-308 | +0        | 4.9e-324  | 4.9e-324  | 9.9e-324  | 2.2e-308  | 2.2e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+maxD** | -inf | -1.8e+308 | -1        | -4.9e-324 | +0        | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+minN** | -inf | -1.8e+308 | -1        | +0        | 4.9e-324  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-308  | 1         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+1**    | -inf | -1.8e+308 | +0        | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 2         | 1.8e+308 | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+maxN** | -inf | +0        | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | +inf     | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **+INF**  | nan  | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf     | +inf | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+
| **QNAN**  | nan  | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan      | nan  | nan  |
+-----------+------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+----------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-fused-multiply-add-fma:

Fused Multiply-Add (fma)
~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-5:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-5:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705991498 87.35%  1.00e-13   8.17e-16   43
output special       531120823  12.52%  --         --         --
input special        41         1e-06%  --         --         --
output denormal      64932      2e-03%  1.29e-03   7.37e-07   9
input denormal       5220413    0.12%   5.96e-08   1.37e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4306       1e-04%  3.82e-09   1.89e-12   27
cancellation         174428     4e-03%  1.09e-08   1.27e-12   26
unclassified         74732      2e-03%  2.58e-09   9.39e-13   28
TOTAL                4242691642 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2600.9       | -0.13x  | 3.76x   | 1887.0       | -0.13x  | 3.78x   | 1805.5 | -0.13x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 5.69         | -0.13x  | 3.76x   | 6.27         | -0.13x  | 3.78x   | 6.21   | -0.13x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 34.5         | -0.38x  | 3.42x   | 32.1         | -0.41x  | 3.45x   | 32.0   | -0.41x  | -0.68x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      16
fp64      0
other     1
**total** **17**
========= ======

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705991498 99.85%  1.00e-13   8.17e-16   43
output special       18531      5e-04%  --         --         --
output denormal      42617      1e-03%  1.29e-03   1.12e-06   9
input denormal       5220413    0.14%   5.96e-08   1.37e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4306       1e-04%  3.82e-09   1.89e-12   27
cancellation         174428     5e-03%  1.09e-08   1.27e-12   26
unclassified         74732      2e-03%  2.58e-09   9.39e-13   28
TOTAL                3711566994 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2275.2       | -0.11x  | 3.36x   | 1628.2       | -0.11x  | 3.27x   | 1559.1 | -0.11x  | -0.21x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 4.98         | -0.11x  | 3.36x   | 5.41         | -0.11x  | 3.27x   | 5.36   | -0.11x  | -0.21x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 68.5         | -0.20x  | 1.72x   | 64.3         | -0.20x  | 1.72x   | 64.4   | -0.20x  | -0.34x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      19
fp64      0
other     0
**total** **19**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3706072849 99.85%  1.00e-13   5.02e-16   43
output special       18531      5e-04%  --         --         --
output denormal      42617      1e-03%  1.29e-03   1.12e-06   9
input denormal       5210818    0.14%   5.96e-08   1.37e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       3100       8e-05%  1.22e-09   1.36e-12   29
cancellation         125702     3e-03%  1.22e-08   1.32e-12   26
unclassified         52908      1e-03%  3.70e-09   1.00e-12   28
TOTAL                3711566994 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 1232.6       | -0.06x  | 1.80x   | 842.6        | -0.06x  | 1.69x   | 813.5 | -0.06x  | -0.11x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 2.70         | -0.06x  | 1.80x   | 2.80         | -0.06x  | 1.69x   | 2.80  | -0.06x  | -0.11x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 96.1         | -0.14x  | 1.22x   | 90.4         | -0.15x  | 1.23x   | 90.5  | -0.15x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      37
fp64      0
other     0
**total** **37**
========= ======

.. _type-fp64mp2-5:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-5:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 45.4         | -0.06x  | 1.02x    | 31.6         | -0.06x  | -0.70x   | 999.6 | -0.14x  | 22.52x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.10         | -0.06x  | 1.02x    | 0.11         | -0.06x  | -0.70x   | 3.44  | -0.14x  | 22.52x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 1075.5       | -0.10x  | 1.13x    | 1088.1       | -0.10x  | -0.96x   | 60.0  | -0.36x  | 17.49x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      16
other     2
**total** **18**
========= ======

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 38.1         | -0.05x  | -0.86x   | 26.6         | -0.05x  | -0.59x   | 839.7 | -0.12x  | 18.92x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.08         | -0.05x  | -0.86x   | 0.09         | -0.05x  | -0.59x   | 2.89  | -0.12x  | 18.92x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 1304.8       | -0.09x  | -0.93x   | 1302.9       | -0.08x  | -0.80x   | 122.9 | -0.18x  | 8.52x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      19
other     0
**total** **19**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 19.5         | -0.03x  | -0.43x   | 13.6         | -0.03x  | -0.30x   | 460.9 | -0.06x  | 10.39x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.04         | -0.03x  | -0.43x   | 0.05         | -0.03x  | -0.30x   | 1.58  | -0.06x  | 10.39x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 2454.7       | -0.05x  | -0.49x   | 2453.6       | -0.05x  | -0.42x   | 170.7 | -0.13x  | 6.15x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      37
other     0
**total** **37**
========= ======

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-multiply-add-mad:

Multiply-Add (mad)
~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-6:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-6:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705975209 87.35%  1.00e-13   7.88e-16   43
output special       531120823  12.52%  --         --         --
input special        41         1e-06%  --         --         --
output denormal      68114      2e-03%  1.29e-03   7.03e-07   9
input denormal       5222880    0.12%   5.96e-08   1.36e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4527       1e-04%  1.38e-09   1.31e-12   29
cancellation         184095     4e-03%  1.22e-08   1.30e-12   26
unclassified         78666      2e-03%  3.70e-09   9.60e-13   28
TOTAL                4242694824 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 3365.9       | -0.15x  | 4.44x   | 2225.5       | -0.15x  | 4.46x   | 2120.1 | -0.15x  | -0.28x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 7.37         | -0.15x  | 4.44x   | 7.40         | -0.15x  | 4.46x   | 7.29   | -0.15x  | -0.28x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 30.5         | -0.40x  | 3.63x   | 31.0         | -0.43x  | 3.57x   | 30.9   | -0.43x  | -0.70x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      0
other     0
**total** **13**
========= ======

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705975209 99.85%  1.00e-13   7.88e-16   43
output special       18531      5e-04%  --         --         --
output denormal      42617      1e-03%  1.29e-03   1.12e-06   9
input denormal       5222880    0.14%   5.96e-08   1.36e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4527       1e-04%  1.38e-09   1.31e-12   29
cancellation         184095     5e-03%  1.22e-08   1.30e-12   26
unclassified         78666      2e-03%  3.70e-09   9.60e-13   28
TOTAL                3711566994 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2589.3       | -0.13x  | 3.74x   | 1861.1       | -0.12x  | 3.73x   | 1780.9 | -0.13x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 5.67         | -0.13x  | 3.74x   | 6.19         | -0.12x  | 3.73x   | 6.12   | -0.13x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 47.8         | -0.28x  | 2.46x   | 45.8         | -0.29x  | 2.41x   | 45.5   | -0.29x  | -0.48x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      16
fp64      0
other     0
**total** **16**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3705962819 99.85%  1.00e-13   9.67e-16   43
output special       18531      5e-04%  --         --         --
output denormal      42617      1e-03%  1.29e-03   1.12e-06   9
input denormal       5223153    0.14%   5.96e-08   1.36e-09   24
output near denormal 40469      1e-03%  1.19e-07   8.32e-08   23
input near inf       4776       1e-04%  1.38e-09   1.31e-12   29
cancellation         192586     5e-03%  7.89e-09   1.28e-12   26
unclassified         82043      2e-03%  2.83e-09   9.66e-13   28
TOTAL                3711566994 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 1797.9       | -0.08x  | 2.37x   | 1192.2       | -0.08x  | 2.39x   | 1146.5 | -0.08x  | -0.15x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 3.94         | -0.08x  | 2.37x   | 3.96         | -0.08x  | 2.39x   | 3.94   | -0.08x  | -0.15x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 78.8         | -0.16x  | 1.41x   | 79.3         | -0.17x  | 1.40x   | 78.9   | -0.16x  | -0.28x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      29
fp64      0
other     0
**total** **29**
========= ======

.. _type-fp64mp2-6:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-6:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``low``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 59.2         | -0.08x  | 1.32x    | 39.0         | -0.08x  | -0.86x   | 1201.6 | -0.17x  | 27.07x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.13         | -0.08x  | 1.32x    | 0.13         | -0.08x  | -0.86x   | 4.13   | -0.17x  | 27.07x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 887.9        | -0.12x  | 1.36x    | 890.9        | -0.12x  | 1.17x    | 60.0   | -0.36x  | 17.50x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      13
other     0
**total** **13**
========= ======

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 48.1         | -0.06x  | 1.07x    | 31.6         | -0.06x  | -0.70x   | 982.4 | -0.13x  | 22.14x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.11         | -0.06x  | 1.07x    | 0.11         | -0.06x  | -0.70x   | 3.38  | -0.13x  | 22.14x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 1112.0       | -0.10x  | 1.09x    | 1082.6       | -0.10x  | -0.96x   | 85.7  | -0.25x  | 12.25x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      16
other     0
**total** **16**
========= ======

*Accuracy: ``high``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16646144 100.00% 0.00e+00   0.00e+00   106
TOTAL       16646144 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 26.3         | -0.03x  | -0.59x   | 17.3         | -0.03x  | -0.38x   | 577.6 | -0.08x  | 13.02x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.06         | -0.03x  | -0.59x   | 0.06         | -0.03x  | -0.38x   | 1.99  | -0.08x  | 13.02x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 1936.4       | -0.06x  | -0.62x   | 1927.9       | -0.06x  | -0.54x   | 155.1 | -0.14x  | 6.77x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      29
other     0
**total** **29**
========= ======

.. _libcudacxx-extended-api-fp-fpmp-spec-mathematical-functions:

Mathematical Functions
----------------------

.. _libcudacxx-extended-api-fp-fpmp-spec-square-root-sqrt:

Square Root (sqrt)
~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-7:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-7:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

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

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2339.2       | -0.58x  | 23.13x  | 1559.8       | -0.58x  | 23.47x  | 1493.6 | -0.58x  | 1.29x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 5.12         | -0.58x  | 23.13x  | 5.19         | -0.58x  | 23.47x  | 5.14   | -0.58x  | 1.29x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 71.4         | -0.94x  | 8.20x   | 71.5         | -0.93x  | 8.15x   | 70.4   | -0.94x  | 1.46x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-7:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-7:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============== ======= ======= ========== ========== ====
Class          Count   Percent Max RelErr Avg RelErr Bits
============== ======= ======= ========== ========== ====
normal (OK)    8305868 99.06%  1.00e-28   -2.09e-20  93
input denormal 78644   0.94%   1.58e-16   2.27e-18   52
TOTAL          8384512 100.00%
============== ======= ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 35.5         | -0.35x  | -0.26x   | 23.3         | -0.35x  | -0.30x   | 579.8 | -0.51x  | 7.89x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.08         | -0.35x  | -0.26x   | 0.08         | -0.35x  | -0.30x   | 1.99  | -0.51x  | 7.89x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 1523.7       | -0.38x  | -0.51x   | 1520.3       | -0.38x  | -0.43x   | 172.3 | -0.60x  | 3.79x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-reciprocal-square-root-rsqrt:

Reciprocal Square Root (rsqrt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-8:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-8:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

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

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2311.3       | -0.36x  | 14.70x  | 1567.4       | -0.41x  | 15.17x  | 1500.9 | -0.40x  | -0.94x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 5.06         | -0.36x  | 14.70x  | 5.21         | -0.41x  | 15.17x  | 5.16   | -0.40x  | -0.94x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 72.2         | -0.62x  | 4.70x   | 73.2         | -0.60x  | 4.62x   | 71.5   | -0.61x  | 1.08x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-8:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-8:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 8384512 100.00% 2.45e-32   -4.81e-33  105
TOTAL       8384512 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 37.5         | -0.24x  | -0.61x   | 24.7         | -0.24x  | -0.72x   | 606.5 | -0.38x  | 18.47x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.08         | -0.24x  | -0.61x   | 0.08         | -0.24x  | -0.72x   | 2.09  | -0.38x  | 18.47x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 1436.6       | -0.24x  | 1.17x    | 1434.5       | -0.24x  | 1.12x    | 168.8 | -0.45x  | 9.54x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-exponential-exp:

Exponential (exp)
~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-9:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-9:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

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

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 330.3        | -0.06x  | 6.51x   | 218.7        | -0.06x  | 6.56x   | 211.1 | -0.06x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.72         | -0.06x  | 6.51x   | 0.73         | -0.06x  | 6.56x   | 0.73  | -0.06x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 330.5        | -0.16x  | 2.98x   | 323.3        | -0.16x  | 3.06x   | 323.1 | -0.17x  | -0.49x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-natural-logarithm-log:

Natural Logarithm (log)
~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-10:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-10:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2139095037 49.80%  5.03e-14   8.02e-16   44
output special 2139095043 49.80%  --         --         --
input special  16777216   0.39%   --         --         --
TOTAL          4294967296 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 327.6        | -0.16x  | 12.59x  | 209.6        | -0.16x  | 12.26x  | 202.3 | -0.16x  | -0.59x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.72         | -0.16x  | 12.59x  | 0.70         | -0.16x  | 12.26x  | 0.70  | -0.16x  | -0.59x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 339.2        | -0.27x  | 5.83x   | 331.1        | -0.27x  | 6.00x   | 331.9 | -0.27x  | -0.82x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      135
fp64      0
other     20
**total** **155**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-power-pow:

Power (pow)
~~~~~~~~~~~

.. _type-fp32mp2-11:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-11:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     1066806685 54.96%  1.00e-13   9.15e-16   43
output special  867368869  44.68%  --         --         --
input special   2          1e-07%  --         --         --
output denormal 1029046    0.05%   1.00e+00   1.00e+00   0
input denormal  642056     0.03%   3.40e-09   2.14e-13   28
output near inf 57508      3e-03%  2.27e-12   2.59e-13   38
input near inf  46077      2e-03%  4.60e-11   1.96e-13   34
cancellation    2722373    0.14%   5.68e-09   2.19e-13   27
unclassified    2455173    0.13%   2.66e-12   1.98e-13   38
TOTAL           1941127789 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 144.3        | -0.21x  | 15.93x  | 79.3         | -0.18x  | 13.32x  | 76.9  | -0.18x  | -0.56x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.32         | -0.21x  | 15.93x  | 0.26         | -0.18x  | 13.32x  | 0.26  | -0.18x  | -0.56x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 822.4        | -0.29x  | 7.02x   | 800.7        | -0.29x  | 7.18x   | 801.2 | -0.29x  | -0.97x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      305
fp64      0
other     69
**total** **374**
========= =======

**Special Values Table:**

+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **a\\b**  | -INF | -maxN | -1       | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1       | +maxN | +INF | QNAN |
+===========+======+=======+==========+=======+=======+=======+====+====+=======+=======+=======+==========+=======+======+======+
| **-INF**  | +0   | +0    | -0       | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -inf     | +inf  | +inf | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-maxN** | +0   | nan   | -0       | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -3.4e+38 | nan   | +inf | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-1**    | 1    | 1     | -1       | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -1       | 1     | 1    | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-minN** | +inf | nan   | -8.5e+37 | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-maxD** | +inf | nan   | -8.5e+37 | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-minD** | +inf | nan   | -inf     | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | -0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **-0**    | +inf | +inf  | +inf     | +inf  | +inf  | +inf  | 1  | 1  | +0    | +0    | +0    | +0       | +0    | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+0**    | +inf | +inf  | +inf     | +inf  | +inf  | +inf  | 1  | 1  | +0    | +0    | +0    | +0       | +0    | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+minD** | +inf | nan   | +inf     | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | +0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+maxD** | +inf | nan   | 8.5e+37  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | +0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+minN** | +inf | nan   | 8.5e+37  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | +0       | nan   | +0   | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+1**    | 1    | 1     | 1        | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1        | 1     | 1    | 1    |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+maxN** | +0   | nan   | +0       | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 3.4e+38  | nan   | +inf | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **+INF**  | +0   | +0    | +0       | +0    | +0    | +0    | 1  | 1  | +inf  | +inf  | +inf  | +inf     | +inf  | +inf | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+
| **QNAN**  | nan  | nan   | nan      | nan   | nan   | nan   | 1  | 1  | nan   | nan   | nan   | nan      | nan   | nan  | nan  |
+-----------+------+-------+----------+-------+-------+-------+----+----+-------+-------+-------+----------+-------+------+------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-cube-root-cbrt:

Cube Root (cbrt)
~~~~~~~~~~~~~~~~

.. _type-fp32mp2-12:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-12:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4278190075 100.00% 3.17e-14   1.37e-15   44
input special 3          7e-08%  --         --         --
TOTAL         4278190078 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 493.3        | -0.17x  | 8.03x   | 329.4        | -0.16x  | 8.15x   | 317.2 | -0.17x  | -0.52x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 1.08         | -0.17x  | 8.03x   | 1.10         | -0.16x  | 8.15x   | 1.09  | -0.17x  | -0.52x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 290.2        | -0.30x  | 3.67x   | 290.2        | -0.29x  | 3.66x   | 290.7 | -0.29x  | -0.88x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-reciprocal-cube-root-rcbrt:

Reciprocal Cube Root (rcbrt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-13:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-13:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4278190075 100.00% 1.06e-14   1.19e-15   46
output special 5          1e-07%  --         --         --
TOTAL          4278190080 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 414.0        | -0.24x  | 9.23x   | 266.4        | -0.22x  | 9.03x   | 256.6 | -0.23x  | -0.53x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.91         | -0.24x  | 9.23x   | 0.89         | -0.22x  | 9.03x   | 0.88  | -0.23x  | -0.53x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 339.3        | -0.51x  | 4.13x   | 333.2        | -0.49x  | 4.14x   | 333.7 | -0.50x  | 1.04x   |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      95
fp64      0
other     35
**total** **130**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-sine-sin:

Sine (sin)
~~~~~~~~~~

.. _type-fp32mp2-14:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-14:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

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

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 202.0        | -0.10x  | 4.64x   | 119.5        | -0.10x  | 4.17x   | 116.2 | -0.10x  | -0.22x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.44         | -0.10x  | 4.64x   | 0.40         | -0.10x  | 4.17x   | 0.40  | -0.10x  | -0.22x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 517.1        | -0.21x  | 2.39x   | 505.4        | -0.22x  | 2.45x   | 505.9 | -0.23x  | -0.52x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      226
fp64      0
other     230
**total** **456**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-cosine-cos:

Cosine (cos)
~~~~~~~~~~~~

.. _type-fp32mp2-15:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-15:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

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

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 201.0        | -0.11x  | 4.68x   | 120.0        | -0.11x  | 4.19x   | 115.8 | -0.10x  | -0.23x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.44         | -0.11x  | 4.68x   | 0.40         | -0.11x  | 4.19x   | 0.40  | -0.10x  | -0.23x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 516.0        | -0.21x  | 2.44x   | 505.0        | -0.23x  | 2.50x   | 506.3 | -0.23x  | -0.56x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      226
fp64      0
other     233
**total** **459**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-hyperbolic-tangent-tanh:

Hyperbolic Tangent (tanh)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-16:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-16:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261412864 100.00% 2.47e-14   4.61e-17   45
TOTAL       4261412864 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 251.9        | -0.08x  | 7.49x   | 156.3        | -0.07x  | 7.13x   | 151.4 | -0.08x  | -0.33x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.55         | -0.08x  | 7.49x   | 0.52         | -0.07x  | 7.13x   | 0.52  | -0.08x  | -0.33x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 332.4        | -0.21x  | 2.74x   | 331.5        | -0.20x  | 2.74x   | 330.7 | -0.21x  | -0.47x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      310
fp64      0
other     29
**total** **339**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-error-function-erf:

Error Function (erf)
~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-17:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-17:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          4165995376 97.72%  1.00e-13   5.74e-16   43
output denormal      53642      1e-03%  3.45e-05   2.38e-07   14
input denormal       97165519   2.28%   5.96e-08   1.44e-09   24
output near denormal 160571     4e-03%  1.19e-07   7.95e-08   23
TOTAL                4263375108 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 125.6        | -0.06x  | 7.33x   | 70.0         | -0.05x  | 6.21x   | 67.7  | -0.05x  | -0.20x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.28         | -0.06x  | 7.33x   | 0.23         | -0.05x  | 6.21x   | 0.23  | -0.05x  | -0.20x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 667.9        | -0.11x  | 4.48x   | 659.4        | -0.11x  | 4.53x   | 658.1 | -0.11x  | -0.59x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-complementary-error-function-erfc:

Complementary Error Function (erfc)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-18:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-18:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          1465663971 45.36%  1.00e-13   1.28e-14   43
output denormal      35872      1e-03%  1.00e+00   2.12e-02   0
input denormal       436207615  13.50%  1.03e-13   1.03e-13   43
output near denormal 13089      4e-04%  1.19e-07   8.60e-08   23
unclassified         1328981987 41.13%  5.96e-08   9.69e-13   24
TOTAL                3230902534 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 85.6         | -0.08x  | 6.82x   | 46.2         | -0.06x  | 5.60x   | 44.8  | -0.06x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.19         | -0.08x  | 6.82x   | 0.15         | -0.06x  | 5.60x   | 0.15  | -0.06x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 754.7        | -0.18x  | 5.71x   | 724.7        | -0.19x  | 5.91x   | 730.0 | -0.19x  | -0.56x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      525
fp64      0
other     21
**total** **546**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-boys-function-f0-boys_f0:

Boys Function F0 (boys_f0)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-19:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-19:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============ ========= ======= ========== ========== ====
Class        Count     Percent Max RelErr Avg RelErr Bits
============ ========= ======= ========== ========== ====
normal (OK)  325851359 100.00% 9.88e-14   1.84e-14   43
unclassified 9         3e-06%  1.09e-13   1.04e-13   43
TOTAL        325851368 100.00%
============ ========= ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 163.7        | 15.50x  | 15.07x  | 93.4         | 13.47x  | 13.09x  | 90.6  | -0.64x  | -0.64x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.36         | 15.50x  | 15.07x  | 0.31         | 13.47x  | 13.09x  | 0.31  | -0.64x  | -0.64x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 353.6        | 14.13x  | 13.77x  | 352.5        | 14.15x  | 13.79x  | 352.1 | 2.25x   | 2.18x   |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-inverse-normal-cdf-normcdfinv:

Inverse Normal CDF (normcdfinv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-20:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-20:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============ ========= ======= ========== ========== ====
Class        Count     Percent Max RelErr Avg RelErr Bits
============ ========= ======= ========== ========== ====
normal (OK)  162681277 97.20%  1.00e-13   7.38e-15   43
unclassified 4683464   2.80%   4.37e-13   1.47e-13   41
TOTAL        167364741 100.00%
============ ========= ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 107.3        | -0.05x  | 7.80x   | 61.2         | -0.04x  | 6.77x   | 58.6  | -0.04x  | -0.31x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.23         | -0.05x  | 7.80x   | 0.20         | -0.04x  | 6.77x   | 0.20  | -0.04x  | -0.31x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 881.3        | -0.13x  | 4.19x   | 869.4        | -0.13x  | 4.20x   | 869.9 | -0.13x  | -0.58x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      735
fp64      0
other     36
**total** **771**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-floor-floor:

Floor (floor)
~~~~~~~~~~~~~

.. _type-fp32mp2-21:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-21:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 3212836860 100.00% 0.00e+00   0.00e+00   48
TOTAL       3212836860 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2876.2       | -0.41x  | 3.48x   | 1971.6       | -0.43x  | 3.33x   | 1903.4 | -0.45x  | -0.47x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 6.30         | -0.41x  | 3.48x   | 6.56         | -0.43x  | 3.33x   | 6.54   | -0.45x  | -0.47x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 103.0        | -0.23x  | 1.03x   | 101.7        | -0.27x  | 1.04x   | 100.8  | -0.25x  | -0.28x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-ceiling-ceil:

Ceiling (ceil)
~~~~~~~~~~~~~~

.. _type-fp32mp2-22:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-22:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 3212836861 100.00% 0.00e+00   0.00e+00   48
TOTAL       3212836861 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2881.8       | -0.41x  | 3.48x   | 1973.7       | -0.43x  | 3.33x   | 1903.8 | -0.45x  | -0.47x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 6.31         | -0.41x  | 3.48x   | 6.56         | -0.43x  | 3.33x   | 6.55   | -0.45x  | -0.47x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 102.8        | -0.23x  | 1.03x   | 102.1        | -0.26x  | 1.04x   | 101.1  | -0.26x  | -0.28x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-round-to-nearest-round:

Round to Nearest (round)
~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-23:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-23:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2164260863 100.00% 3.55e-15   9.54e-18   48
TOTAL       2164260863 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 1419.5       | -0.21x  | 3.59x   | 954.8        | -0.21x  | 3.67x   | 917.1 | -0.21x  | -0.25x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 3.11         | -0.21x  | 3.59x   | 3.17         | -0.21x  | 3.67x   | 3.15  | -0.21x  | -0.25x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 205.5        | -0.18x  | -0.82x  | 203.9        | -0.18x  | -0.83x  | 203.1 | -0.18x  | -0.22x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-truncate-trunc:

Truncate (trunc)
~~~~~~~~~~~~~~~~

.. _type-fp32mp2-24:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-24:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2147483646 100.00% 0.00e+00   0.00e+00   48
TOTAL       2147483646 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2845.8       | -0.41x  | 3.44x   | 2036.6       | -0.44x  | 3.44x   | 1930.5 | -0.45x  | -0.48x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 6.23         | -0.41x  | 3.44x   | 6.77         | -0.44x  | 3.44x   | 6.64   | -0.45x  | -0.48x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 104.5        | -0.23x  | 1.02x   | 101.9        | -0.27x  | 1.04x   | 101.0  | -0.26x  | -0.28x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-comparison-operations:

Comparison Operations
---------------------

.. _libcudacxx-extended-api-fp-fpmp-spec-equal-eq:

Equal (eq)
~~~~~~~~~~

.. _type-fp32mp2-25:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-25:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 9266.0       | -0.63x  | 12.21x  | 4092.5       | -0.66x  | 8.20x   | 3926.1 | -0.66x  | -0.95x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 20.28        | -0.63x  | 12.21x  | 13.61        | -0.66x  | 8.20x   | 13.50  | -0.66x  | -0.95x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 20.9         | -0.85x  | 5.25x   | 18.4         | -0.84x  | 5.92x   | 18.5   | -0.81x  | 1.39x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-9:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-9:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 379.9        | -0.50x  | -0.39x   | 249.7        | -0.50x  | -0.50x   | 2353.8 | -0.57x  | 4.80x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.83         | -0.50x  | -0.39x   | 0.83         | -0.50x  | -0.50x   | 8.09   | -0.57x  | 4.80x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 207.7        | -0.53x  | -0.53x   | 207.5        | -0.53x  | -0.53x   | 29.9   | -0.85x  | 3.65x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-not-equal-ne:

Not Equal (ne)
~~~~~~~~~~~~~~

.. _type-fp32mp2-26:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-26:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 9234.7       | -0.62x  | 12.16x  | 4102.3       | -0.66x  | 8.22x   | 3925.2 | -0.66x  | -0.95x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 20.21        | -0.62x  | 12.16x  | 13.64        | -0.66x  | 8.22x   | 13.50  | -0.66x  | -0.95x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 21.0         | -0.84x  | 5.21x   | 18.3         | -0.83x  | 5.93x   | 18.1   | -0.82x  | 1.40x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-10:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-10:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 379.9        | -0.50x  | -0.39x   | 249.8        | -0.50x  | -0.49x   | 2355.8 | -0.57x  | 4.79x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.83         | -0.50x  | -0.39x   | 0.83         | -0.50x  | -0.49x   | 8.10   | -0.57x  | 4.79x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 207.6        | -0.53x  | -0.50x   | 206.7        | -0.53x  | -0.50x   | 30.0   | -0.84x  | 3.40x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 1    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-less-than-lt:

Less Than (lt)
~~~~~~~~~~~~~~

.. _type-fp32mp2-27:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-27:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 7253.8       | -0.49x  | 9.55x   | 3044.0       | -0.49x  | 6.10x   | 2933.1 | -0.49x  | -0.71x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 15.88        | -0.49x  | 9.55x   | 10.12        | -0.49x  | 6.10x   | 10.09  | -0.49x  | -0.71x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 35.6         | -0.49x  | 3.05x   | 32.1         | -0.47x  | 3.40x   | 32.0   | -0.47x  | -0.80x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-11:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-11:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 254.1        | -0.33x  | -0.27x   | 167.0        | -0.33x  | -0.35x   | 2065.3 | -0.50x  | 4.43x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.56         | -0.33x  | -0.27x   | 0.56         | -0.33x  | -0.35x   | 7.10   | -0.50x  | 4.43x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 303.4        | -0.36x  | -0.33x   | 279.6        | -0.39x  | -0.38x   | 38.6   | -0.66x  | 2.71x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-less-than-or-equal-le:

Less Than or Equal (le)
~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-28:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-28:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 7243.7       | -0.49x  | 9.54x   | 3043.5       | -0.49x  | 6.10x   | 2932.6 | -0.49x  | -0.71x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 15.86        | -0.49x  | 9.54x   | 10.12        | -0.49x  | 6.10x   | 10.08  | -0.49x  | -0.71x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 35.3         | -0.50x  | 3.09x   | 32.2         | -0.46x  | 3.39x   | 32.1   | -0.47x  | -0.79x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-12:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-12:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 254.0        | -0.33x  | -0.27x   | 167.0        | -0.33x  | -0.35x   | 2060.4 | -0.50x  | 4.42x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.56         | -0.33x  | -0.27x   | 0.56         | -0.33x  | -0.35x   | 7.08   | -0.50x  | 4.42x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 303.3        | -0.36x  | -0.39x   | 279.7        | -0.39x  | -0.41x   | 38.7   | -0.66x  | 2.91x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 0    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 0    | 0     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 0    | 0     | 0  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 0    | 0     | 0  | 0     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 0    | 0     | 0  | 0     | 0     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 0    | 0     | 0  | 0     | 0     | 0     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-greater-than-gt:

Greater Than (gt)
~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-29:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-29:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 7215.6       | -0.49x  | 9.51x   | 3047.1       | -0.49x  | 6.11x   | 2924.9 | -0.49x  | -0.71x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 15.79        | -0.49x  | 9.51x   | 10.13        | -0.49x  | 6.11x   | 10.06  | -0.49x  | -0.71x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 35.7         | -0.49x  | 3.07x   | 31.9         | -0.47x  | 3.42x   | 31.9   | -0.46x  | -0.81x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-13:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-13:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 253.5        | -0.33x  | -0.27x   | 166.6        | -0.33x  | -0.35x   | 2059.5 | -0.50x  | 4.42x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.56         | -0.33x  | -0.27x   | 0.55         | -0.33x  | -0.35x   | 7.08   | -0.50x  | 4.42x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 306.1        | -0.36x  | -0.36x   | 282.5        | -0.39x  | -0.38x   | 38.6   | -0.66x  | 2.76x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-greater-than-or-equal-ge:

Greater Than or Equal (ge)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-30:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-30:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4261478400 100.00% 0.00e+00   0.00e+00   1
TOTAL       4261478400 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 7225.3       | -0.49x  | 9.52x   | 3048.5       | -0.49x  | 6.11x   | 2922.9 | -0.49x  | -0.71x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 15.82        | -0.49x  | 9.52x   | 10.14        | -0.49x  | 6.11x   | 10.05  | -0.49x  | -0.71x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 35.3         | -0.49x  | 3.09x   | 32.0         | -0.47x  | 3.40x   | 32.0   | -0.47x  | -0.80x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _type-fp64mp2-14:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-14:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16760836 100.00% 0.00e+00   0.00e+00   1
TOTAL       16760836 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 253.5        | -0.33x  | -0.27x   | 166.6        | -0.33x  | -0.35x   | 2060.5 | -0.50x  | 4.42x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.56         | -0.33x  | -0.27x   | 0.55         | -0.33x  | -0.35x   | 7.09   | -0.50x  | 4.42x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 306.1        | -0.36x  | -0.33x   | 282.5        | -0.38x  | -0.36x   | 38.7   | -0.65x  | 2.60x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **a\\b**  | -INF | -maxN | -1 | -minN | -maxD | -minD | -0 | +0 | +minD | +maxD | +minN | +1 | +maxN | +INF | QNAN |
+===========+======+=======+====+=======+=======+=======+====+====+=======+=======+=======+====+=======+======+======+
| **-INF**  | 1    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxN** | 1    | 1     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-1**    | 1    | 1     | 1  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minN** | 1    | 1     | 1  | 1     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-maxD** | 1    | 1     | 1  | 1     | 1     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-minD** | 1    | 1     | 1  | 1     | 1     | 1     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **-0**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+0**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxD** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+minN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+1**    | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+maxN** | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **+INF**  | 1    | 1     | 1  | 1     | 1     | 1     | 1  | 1  | 1     | 1     | 1     | 1  | 1     | 1    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+
| **QNAN**  | 0    | 0     | 0  | 0     | 0     | 0     | 0  | 0  | 0     | 0     | 0     | 0  | 0     | 0    | 0    |
+-----------+------+-------+----+-------+-------+-------+----+----+-------+-------+-------+----+-------+------+------+

.. _libcudacxx-extended-api-fp-fpmp-spec-type-conversions:

Type Conversions
----------------

.. _libcudacxx-extended-api-fp-fpmp-spec-to-int32-mp2int:

To Int32 (mp2int)
~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-31:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-31:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2533359616 100.00% 0.00e+00   0.00e+00   32
TOTAL       2533359616 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2322.3       | -0.33x  | -1.00x  | 1544.3       | -0.33x  | -1.00x  | 1477.9 | -0.35x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 5.08         | -0.33x  | -1.00x  | 5.14         | -0.33x  | -1.00x  | 5.08   | -0.35x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 71.2         | -0.35x  | -1.00x  | 71.2         | -0.38x  | 1.01x   | 69.1   | -0.38x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-15:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-15:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 8577024 100.00% 0.00e+00   0.00e+00   32
TOTAL       8577024 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 158.9        | -0.19x  | -1.00x   | 104.4        | -0.18x  | -1.00x   | 1412.7 | -0.36x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.35         | -0.19x  | -1.00x   | 0.35         | -0.18x  | -1.00x   | 4.86   | -0.36x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 378.7        | -0.28x  | 1.00x    | 378.0        | -0.28x  | 1.00x    | 75.0   | -0.43x  | 1.00x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-to-uint32-mp2uint:

To UInt32 (mp2uint)
~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-32:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-32:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 1266679810 100.00% 0.00e+00   0.00e+00   32
TOTAL       1266679810 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2319.8       | -0.33x  | -1.00x  | 1544.7       | -0.33x  | -1.00x  | 1478.3 | -0.35x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 5.08         | -0.33x  | -1.00x  | 5.14         | -0.33x  | -1.00x  | 5.08   | -0.35x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 71.3         | -0.34x  | -1.00x  | 71.1         | -0.38x  | 1.00x   | 69.2   | -0.37x  | 1.00x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-16:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-16:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 4288512 100.00% 0.00e+00   0.00e+00   32
TOTAL       4288512 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 158.9        | -0.19x  | -1.00x   | 104.4        | -0.18x  | -1.00x   | 1413.2 | -0.36x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.35         | -0.19x  | -1.00x   | 0.35         | -0.18x  | -1.00x   | 4.86   | -0.36x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 379.2        | -0.28x  | -1.00x   | 378.4        | -0.28x  | -1.00x   | 75.1   | -0.43x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-to-int64-mp2ll:

To Int64 (mp2ll)
~~~~~~~~~~~~~~~~

.. _type-fp32mp2-33:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-33:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2533359616 100.00% 0.00e+00   0.00e+00   64
TOTAL       2533359616 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 267.7        | -0.32x  | -1.00x  | 176.2        | -0.30x  | -1.00x  | 1495.5 | -0.35x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 0.59         | -0.32x  | -1.00x  | 0.59         | -0.30x  | -1.00x  | 5.14   | -0.35x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 263.8        | -0.40x  | -1.00x  | 262.9        | -0.41x  | -1.00x  | 60.0   | -0.43x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-17:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-17:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 8577024 100.00% 0.00e+00   0.00e+00   64
TOTAL       8577024 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 157.5        | -0.19x  | -1.00x   | 103.5        | -0.17x  | -1.00x   | 1411.5 | -0.35x  | -0.99x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.34         | -0.19x  | -1.00x   | 0.34         | -0.17x  | -1.00x   | 4.85   | -0.35x  | -0.99x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 381.7        | -0.28x  | -1.00x   | 391.3        | -0.27x  | -1.00x   | 72.3   | -0.39x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-to-uint64-mp2ull:

To UInt64 (mp2ull)
~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-34:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-34:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 1266679810 100.00% 0.00e+00   0.00e+00   64
TOTAL       1266679810 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 267.8        | -0.32x  | -1.00x  | 176.2        | -0.30x  | -1.00x  | 1495.5 | -0.35x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 0.59         | -0.32x  | -1.00x  | 0.59         | -0.30x  | -1.00x  | 5.14   | -0.35x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 263.9        | -0.41x  | -1.00x  | 262.8        | -0.41x  | -1.00x  | 60.0   | -0.43x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-18:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-18:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======= ======= ========== ========== ====
Class       Count   Percent Max RelErr Avg RelErr Bits
=========== ======= ======= ========== ========== ====
normal (OK) 4288512 100.00% 0.00e+00   0.00e+00   64
TOTAL       4288512 100.00%
=========== ======= ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 157.5        | -0.19x  | -1.00x   | 103.6        | -0.17x  | -1.00x   | 1411.6 | -0.35x  | -0.99x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.34         | -0.19x  | -1.00x   | 0.34         | -0.17x  | -1.00x   | 4.85   | -0.35x  | -0.99x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 381.7        | -0.28x  | -1.00x   | 391.2        | -0.27x  | -1.00x   | 72.3   | -0.39x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-from-int32-int2mp:

From Int32 (int2mp)
~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-35:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-35:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967295 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967295 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 13419.1      | -0.97x  | -0.96x  | 9095.8       | -0.99x  | -0.99x  | 8769.0 | -0.99x  | -0.99x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 29.37        | -0.97x  | -0.96x  | 30.25        | -0.99x  | -0.99x  | 30.15  | -0.99x  | -0.99x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 10.4         | -0.98x  | -0.99x  | 11.4         | -1.00x  | -1.00x  | 11.0   | -0.99x  | 1.00x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-19:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-19:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 827.0        | 3.03x   | -1.00x   | 593.1        | 3.31x   | -1.00x   | 4224.1 | 2.92x   | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 1.81         | 3.03x   | -1.00x   | 1.97         | 3.31x   | -1.00x   | 14.52  | 2.92x   | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 105.9        | 2.27x   | 1.01x    | 105.2        | 2.27x   | 1.01x    | 26.2   | 2.44x   | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      1
other     1
**total** **2**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-from-uint32-uint2mp:

From UInt32 (uint2mp)
~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-36:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-36:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967295 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967295 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 13784.3      | -0.99x  | -0.98x  | 9093.3       | -0.99x  | -0.99x  | 8759.8 | -0.99x  | -0.99x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 30.17        | -0.99x  | -0.98x  | 30.24        | -0.99x  | -0.99x  | 30.12  | -0.99x  | -0.99x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 10.7         | -0.98x  | -0.95x  | 10.9         | 1.01x   | 1.04x   | 11.0   | 1.01x   | 1.01x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-20:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-20:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 827.2        | 3.03x   | -1.00x   | 593.2        | 3.31x   | -1.00x   | 4225.7 | 2.92x   | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 1.81         | 3.03x   | -1.00x   | 1.97         | 3.31x   | -1.00x   | 14.53  | 2.92x   | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 106.2        | 2.26x   | 1.01x    | 105.7        | 2.27x   | 1.01x    | 25.8   | 2.49x   | 1.01x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= =====
Class     Count
========= =====
fp32      0
fp64      1
other     1
**total** **2**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-from-int64-ll2mp:

From Int64 (ll2mp)
~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-37:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-37:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967296 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967296 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 280.7        | -0.34x  | -1.00x  | 184.6        | -0.31x  | -1.00x  | 1368.2 | -0.39x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 0.61         | -0.34x  | -1.00x  | 0.61         | -0.31x  | -1.00x  | 4.70   | -0.39x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 306.0        | -0.34x  | -1.00x  | 294.8        | -0.35x  | -1.00x  | 76.4   | -0.49x  | 1.00x   |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-21:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-21:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 826.2        | 6.07x   | -1.00x   | 592.5        | 6.61x   | -1.00x   | 3925.9 | 4.61x   | -0.98x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 1.81         | 6.07x   | -1.00x   | 1.97         | 6.61x   | -1.00x   | 13.50  | 4.61x   | -0.98x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 108.0        | 4.33x   | -1.00x   | 107.6        | 4.34x   | -1.00x   | 28.0   | 3.81x   | 1.01x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-from-uint64-ull2mp:

From UInt64 (ull2mp)
~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-38:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-38:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4294967296 100.00% 0.00e+00   0.00e+00   48
TOTAL       4294967296 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 280.6        | -0.34x  | -1.00x  | 184.6        | -0.31x  | 1.00x   | 1368.8 | -0.39x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 0.61         | -0.34x  | -1.00x  | 0.61         | -0.31x  | 1.00x   | 4.71   | -0.39x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 306.1        | -0.34x  | -1.00x  | 294.4        | -0.35x  | -1.00x  | 76.1   | -0.49x  | -1.00x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

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

.. _type-fp64mp2-22:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-22:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16777216 100.00% 0.00e+00   0.00e+00   106
TOTAL       16777216 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 826.3        | 6.07x   | -1.00x   | 592.3        | 6.61x   | -1.00x   | 3971.8 | 4.66x   | -0.99x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 1.81         | 6.07x   | -1.00x   | 1.97         | 6.61x   | -1.00x   | 13.66  | 4.66x   | -0.99x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 108.2        | 4.33x   | -0.99x   | 108.0        | 4.33x   | -1.00x   | 27.9   | 3.82x   | 1.02x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

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

.. _libcudacxx-extended-api-fp-fpmp-spec-to-native-float-mp2fp:

To Native Float (mp2fp)
~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-39:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-39:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4278190075 100.00% 0.00e+00   0.00e+00   53
input special 3          7e-08%  --         --         --
TOTAL         4278190078 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 1116.7       | 1.35x   | 4.10x   | 733.9        | 1.24x   | 4.09x   | 714.5 | -0.17x  | -0.32x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 2.44         | 1.35x   | 4.10x   | 2.44         | 1.24x   | 4.09x   | 2.46  | -0.17x  | -0.32x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 223.1        | -0.47x  | 1.07x   | 219.9        | -0.48x  | 1.08x   | 151.8 | -0.17x  | -0.28x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      7
fp64      3
other     31
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

.. _type-fp64mp2-23:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-23:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============= ======== ======= ========== ========== ====
Class         Count    Percent Max RelErr Avg RelErr Bits
============= ======== ======= ========== ========== ====
normal (OK)   16769024 99.95%  0.00e+00   0.00e+00   113
input special 8192     0.05%   --         --         --
TOTAL         16777216 100.00%
============= ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 19.9         | -0.50x  | 1.00x    | 12.7         | -0.48x  | 1.00x    | 77.2  | -0.16x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.04         | -0.50x  | 1.00x    | 0.04         | -0.48x  | 1.00x    | 0.27  | -0.16x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 3605.7       | -0.43x  | 1.00x    | 3636.6       | -0.44x  | -1.00x   | 943.3 | -0.26x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ========
Class     Count
========= ========
fp32      0
fp64      86
other     1001
**total** **1087**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-from-native-float-fp2mp:

From Native Float (fp2mp)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _type-fp32mp2-40:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-40:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    532651806  22.09%  1.00e-13   1.60e-17   43
output special 1879048192 77.91%  --         --         --
unclassified   24802      1e-03%  2.97e-09   1.13e-12   28
TOTAL          2411724800 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 2014.0       | 2.43x   | 9.93x   | 1290.8       | 2.18x   | 9.68x   | 1244.1 | -0.32x  | -0.90x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 4.41         | 2.43x   | 9.93x   | 4.29         | 2.18x   | 9.68x   | 4.28   | -0.32x  | -0.90x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 89.8         | 1.18x   | 3.67x   | 86.7         | 1.22x   | 3.79x   | 85.0   | -0.38x  | -0.88x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      6
fp64      1
other     18
**total** **25**
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

.. _type-fp64mp2-24:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-24:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

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

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 14.9         | -0.31x  | 1.00x    | 9.5          | -0.31x  | 1.00x    | 66.7  | -0.24x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.03         | -0.31x  | 1.00x    | 0.03         | -0.31x  | 1.00x    | 0.23  | -0.24x  | -1.00x   |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 4256.6       | -0.30x  | -1.00x   | 4221.5       | -0.29x  | -1.00x   | 895.1 | -0.25x  | 1.00x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ========
Class     Count
========= ========
fp32      0
fp64      103
other     1069
**total** **1172**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-other-functions:

Other Functions
---------------

.. _libcudacxx-extended-api-fp-fpmp-spec-acos:

ACOS
~~~~

.. _type-fp32mp2-41:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-41:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2130706434 100.00% 3.41e-14   5.00e-16   44
TOTAL       2130706434 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 215.8        | -0.10x  | 7.49x   | 124.8        | -0.09x  | 6.58x   | 120.4 | -0.09x  | -0.31x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.47         | -0.10x  | 7.49x   | 0.41         | -0.09x  | 6.58x   | 0.41  | -0.09x  | -0.31x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 389.1        | -0.29x  | 5.08x   | 375.4        | -0.29x  | 5.25x   | 375.0 | -0.29x  | -0.61x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      454
fp64      0
other     10
**total** **464**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-acosh:

ACOSH
~~~~~

.. _type-fp32mp2-42:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-42:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   1073741822 100.00% 6.57e-14   1.14e-15   43
input special 1          9e-08%  --         --         --
TOTAL         1073741823 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 164.1        | -0.13x  | 8.75x   | 94.6         | -0.11x  | 7.67x   | 91.4  | -0.11x  | -0.37x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.36         | -0.13x  | 8.75x   | 0.31         | -0.11x  | 7.67x   | 0.31  | -0.11x  | -0.37x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 637.4        | -0.23x  | 3.69x   | 606.5        | -0.24x  | 3.85x   | 606.1 | -0.24x  | -0.67x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      517
fp64      0
other     85
**total** **602**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-asin:

ASIN
~~~~

.. _type-fp32mp2-43:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-43:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 2113929218 100.00% 4.75e-14   9.84e-17   44
TOTAL       2113929218 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 199.5        | -0.08x  | 6.92x   | 113.7        | -0.07x  | 6.00x   | 109.7 | -0.07x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.44         | -0.08x  | 6.92x   | 0.38         | -0.07x  | 6.00x   | 0.38  | -0.07x  | -0.24x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 445.0        | -0.22x  | 4.40x   | 425.8        | -0.23x  | 4.55x   | 426.1 | -0.23x  | -0.47x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      432
fp64      0
other     9
**total** **441**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-asinh:

ASINH
~~~~~

.. _type-fp32mp2-44:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-44:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4261412864 100.00% 8.97e-14   7.61e-16   43
input special 3          7e-08%  --         --         --
TOTAL         4261412867 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 157.7        | -0.12x  | 8.83x   | 91.3         | -0.11x  | 7.78x   | 89.4  | -0.11x  | -0.38x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.35         | -0.12x  | 8.83x   | 0.30         | -0.11x  | 7.78x   | 0.31  | -0.11x  | -0.38x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 908.4        | -0.19x  | 3.35x   | 857.8        | -0.19x  | 3.49x   | 856.5 | -0.19x  | -0.55x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      521
fp64      0
other     88
**total** **609**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-atan:

ATAN
~~~~

.. _type-fp32mp2-45:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-45:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4261412864 100.00% 2.82e-14   3.29e-16   45
input special 3          7e-08%  --         --         --
TOTAL         4261412867 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 187.0        | -0.07x  | 6.79x   | 101.7        | -0.06x  | 5.62x   | 98.0  | -0.06x  | -0.20x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.41         | -0.07x  | 6.79x   | 0.34         | -0.06x  | 5.62x   | 0.34  | -0.06x  | -0.20x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 456.0        | -0.19x  | 4.02x   | 435.5        | -0.20x  | 4.18x   | 433.7 | -0.19x  | -0.62x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      264
fp64      0
other     6
**total** **270**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-atan2:

ATAN2
~~~~~

.. _type-fp32mp2-46:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-46:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          3883669366 97.32%  1.00e-13   8.70e-16   43
output special       65536      2e-03%  --         --         --
output denormal      1616789    0.04%   1.00e+00   9.73e-01   0
input denormal       88376055   2.21%   1.79e-07   4.59e-09   22
output near denormal 103777     3e-03%  1.00e+00   6.31e-01   0
input near inf       16487858   0.41%   1.00e+00   5.06e-01   0
cancellation         96863      2e-03%  4.02e-09   1.02e-12   27
TOTAL                3990416244 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 164.3        | -0.13x  | 8.23x   | 90.1         | -0.11x  | 6.87x   | 87.4  | -0.11x  | -0.31x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.36         | -0.13x  | 8.23x   | 0.30         | -0.11x  | 6.87x   | 0.30  | -0.11x  | -0.31x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 509.4        | -0.26x  | 4.88x   | 487.6        | -0.26x  | 5.10x   | 485.7 | -0.26x  | -0.72x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      299
fp64      0
other     46
**total** **345**
========= =======

**Special Values Table:**

+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **a\\b**  | -INF | -maxN | -1   | -minN | -maxD | -minD | -0   | +0   | +minD | +maxD    | +minN    | +1       | +maxN    | +INF  | QNAN |
+===========+======+=======+======+=======+=======+=======+======+======+=======+==========+==========+==========+==========+=======+======+
| **-INF**  | -2.4 | -1.6  | -1.6 | -1.6  | -1.6  | -1.6  | -1.6 | -1.6 | -1.6  | -1.6     | -1.6     | -1.6     | -1.6     | -0.79 | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-maxN** | -3.1 | -2.4  | -1.6 | -1.6  | -1.6  | -1.6  | -1.6 | -1.6 | -1.6  | -1.6     | -1.6     | -1.6     | -0.79    | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-1**    | -3.1 | -3.1  | -2.4 | -1.6  | -1.6  | -1.6  | -1.6 | -1.6 | -1.6  | -1.6     | -1.6     | -0.79    | -2.9e-39 | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-minN** | -3.1 | -3.1  | -3.1 | -2.4  | -2.4  | -1.6  | -1.6 | -1.6 | -1.6  | -0.79    | -0.79    | -1.2e-38 | -0       | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-maxD** | -3.1 | -3.1  | -3.1 | -2.4  | -2.4  | -1.6  | -1.6 | -1.6 | -1.6  | -0.79    | -0.79    | -1.2e-38 | -0       | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-minD** | -3.1 | -3.1  | -3.1 | -3.1  | -3.1  | nan   | nan  | nan  | nan   | -1.2e-07 | -1.2e-07 | -1.4e-45 | -0       | -0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **-0**    | 3.1  | 3.1   | 3.1  | 3.1   | 3.1   | nan   | +0   | +0   | nan   | +0       | +0       | +0       | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+0**    | 3.1  | 3.1   | 3.1  | 3.1   | 3.1   | nan   | +0   | +0   | nan   | +0       | +0       | +0       | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+minD** | 3.1  | 3.1   | 3.1  | 3.1   | 3.1   | nan   | nan  | nan  | nan   | 1.2e-07  | 1.2e-07  | 1.4e-45  | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+maxD** | 3.1  | 3.1   | 3.1  | 2.4   | 2.4   | 1.6   | 1.6  | 1.6  | 1.6   | 0.79     | 0.79     | 1.2e-38  | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+minN** | 3.1  | 3.1   | 3.1  | 2.4   | 2.4   | 1.6   | 1.6  | 1.6  | 1.6   | 0.79     | 0.79     | 1.2e-38  | +0       | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+1**    | 3.1  | 3.1   | 2.4  | 1.6   | 1.6   | 1.6   | 1.6  | 1.6  | 1.6   | 1.6      | 1.6      | 0.79     | 2.9e-39  | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+maxN** | 3.1  | 2.4   | 1.6  | 1.6   | 1.6   | 1.6   | 1.6  | 1.6  | 1.6   | 1.6      | 1.6      | 1.6      | 0.79     | +0    | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **+INF**  | 2.4  | 1.6   | 1.6  | 1.6   | 1.6   | 1.6   | 1.6  | 1.6  | 1.6   | 1.6      | 1.6      | 1.6      | 1.6      | 0.79  | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+
| **QNAN**  | nan  | nan   | nan  | nan   | nan   | nan   | nan  | nan  | nan   | nan      | nan      | nan      | nan      | nan   | nan  |
+-----------+------+-------+------+-------+-------+-------+------+------+-------+----------+----------+----------+----------+-------+------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-atanh:

ATANH
~~~~~

.. _type-fp32mp2-47:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-47:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2113929216 100.00% 3.00e-14   1.00e-16   44
output special 2          9e-08%  --         --         --
TOTAL          2113929218 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 206.9        | -0.14x  | 10.16x  | 122.0        | -0.12x  | 9.13x   | 116.9 | -0.12x  | -0.43x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.45         | -0.14x  | 10.16x  | 0.41         | -0.12x  | 9.13x   | 0.40  | -0.12x  | -0.43x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 358.8        | -0.37x  | 6.50x   | 358.5        | -0.37x  | 6.48x   | 358.0 | -0.37x  | 1.23x   |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      463
fp64      0
other     68
**total** **531**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-cosh:

COSH
~~~~

.. _type-fp32mp2-48:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-48:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2233695454 99.81%  1.00e-13   7.95e-16   43
output special  181863     8e-03%  --         --         --
output near inf 22838      1e-03%  2.98e-13   2.27e-13   41
unclassified    4132412    0.18%   6.53e-13   1.57e-13   40
TOTAL           2238032567 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 278.6        | -0.13x  | 7.55x   | 181.7        | -0.11x  | 7.49x   | 175.7 | -0.13x  | -0.35x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.61         | -0.13x  | 7.55x   | 0.60         | -0.11x  | 7.49x   | 0.60  | -0.13x  | -0.35x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 408.7        | -0.26x  | 3.42x   | 400.7        | -0.26x  | 3.48x   | 399.1 | -0.26x  | -0.61x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      160
fp64      0
other     18
**total** **178**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-exp10:

EXP10
~~~~~

.. _type-fp32mp2-49:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-49:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2217857141 68.28%  9.99e-14   8.53e-17   43
output special  1030086470 31.71%  --         --         --
input special   1          3e-08%  --         --         --
output denormal 39806      1e-03%  1.54e-02   2.17e-06   6
unclassified    2079       6e-05%  6.18e-10   1.01e-12   30
TOTAL           3247985497 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 318.9        | -0.06x  | 7.12x   | 183.3        | -0.05x  | 6.23x   | 178.0 | -0.05x  | -0.22x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.70         | -0.06x  | 7.12x   | 0.61         | -0.05x  | 6.23x   | 0.61  | -0.05x  | -0.22x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 491.6        | -0.11x  | 2.27x   | 490.2        | -0.10x  | 2.28x   | 488.9 | -0.11x  | -0.36x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      198
fp64      0
other     22
**total** **220**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-exp2:

EXP2
~~~~

.. _type-fp32mp2-50:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-50:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2245060170 68.79%  1.00e-13   6.76e-16   43
output special  1015021568 31.10%  --         --         --
input special   1          3e-08%  --         --         --
output denormal 573105     0.02%   1.67e-01   6.81e-06   2
output near inf 48145      1e-03%  3.49e-13   1.37e-13   41
unclassified    2776486    0.09%   1.52e-08   2.11e-13   25
TOTAL           3263479475 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 344.9        | -0.05x  | 7.25x   | 217.8        | -0.06x  | 6.97x   | 210.4 | -0.06x  | -0.25x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.75         | -0.05x  | 7.25x   | 0.72         | -0.06x  | 6.97x   | 0.72  | -0.06x  | -0.25x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 316.7        | -0.14x  | 3.31x   | 316.3        | -0.13x  | 3.31x   | 314.9 | -0.14x  | -0.52x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      126
fp64      0
other     22
**total** **148**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-expm1:

EXPM1
~~~~~

.. _type-fp32mp2-51:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-51:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     3239165628 76.01%  1.00e-13   3.33e-16   43
output special  1020169796 23.94%  --         --         --
input special   1          2e-08%  --         --         --
output near inf 66521      2e-03%  3.53e-13   1.73e-13   41
unclassified    2010918    0.05%   6.53e-13   1.56e-13   40
TOTAL           4261412864 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 262.9        | -0.11x  | 6.47x   | 160.1        | -0.11x  | 5.99x   | 154.6 | -0.11x  | -0.32x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.58         | -0.11x  | 6.47x   | 0.53         | -0.11x  | 5.99x   | 0.53  | -0.11x  | -0.32x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 489.3        | -0.15x  | 2.70x   | 478.4        | -0.15x  | 2.76x   | 478.0 | -0.15x  | -0.46x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      285
fp64      0
other     28
**total** **313**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-fmod:

FMOD
~~~~

.. _type-fp32mp2-52:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-52:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4205136149 100.00% 3.55e-15   8.72e-17   48
TOTAL       4205136149 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 1615.8       | -0.27x  | 4.25x   | 846.7        | -0.20x  | 3.39x   | 815.6 | -0.20x  | -0.49x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 3.54         | -0.27x  | 4.25x   | 2.82         | -0.20x  | 3.39x   | 2.80  | -0.20x  | -0.49x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 109.4        | -0.43x  | 1.08x   | 106.3        | -0.44x  | 1.11x   | 106.1 | -0.44x  | -0.78x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      41
fp64      0
other     224
**total** **265**
========= =======

**Special Values Table:**

+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **a\\b**  | -INF     | -maxN    | -1       | -minN    | -maxD    | -minD | -0  | +0  | +minD | +maxD    | +minN    | +1       | +maxN    | +INF     | QNAN |
+===========+==========+==========+==========+==========+==========+=======+=====+=====+=======+==========+==========+==========+==========+==========+======+
| **-INF**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-maxN** | -3.4e+38 | -0       | -0       | -0       | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -0       | -0       | -0       | -3.4e+38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-1**    | -1       | -1       | -0       | -0       | -2.9e-42 | -0    | nan | nan | -0    | -2.9e-42 | -0       | -0       | -1       | -1       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-minN** | -1.2e-38 | -1.2e-38 | -1.2e-38 | -0       | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -0       | -1.2e-38 | -1.2e-38 | -1.2e-38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-maxD** | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -0       | -0    | nan | nan | -0    | -0       | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-minD** | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-0**    | -0       | -0       | -0       | -0       | -0       | -0    | nan | nan | -0    | -0       | -0       | -0       | -0       | -0       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+0**    | +0       | +0       | +0       | +0       | +0       | +0    | nan | nan | +0    | +0       | +0       | +0       | +0       | +0       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+minD** | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+maxD** | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | +0       | +0    | nan | nan | +0    | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+minN** | 1.2e-38  | 1.2e-38  | 1.2e-38  | +0       | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+1**    | 1        | 1        | +0       | +0       | 2.9e-42  | +0    | nan | nan | +0    | 2.9e-42  | +0       | +0       | 1        | 1        | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+maxN** | 3.4e+38  | +0       | +0       | +0       | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | +0       | +0       | +0       | 3.4e+38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+INF**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **QNAN**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-frexp:

FREXP
~~~~~

.. _type-fp32mp2-53:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-53:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   4278190075 100.00% 0.00e+00   0.00e+00   48
input special 3          7e-08%  --         --         --
TOTAL         4278190078 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 449.8        | -0.07x  | 1.88x   | 296.5        | -0.10x  | 1.75x   | 706.5 | -0.25x  | -0.33x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.98         | -0.07x  | 1.88x   | 0.99         | -0.10x  | 1.75x   | 2.43  | -0.25x  | -0.33x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 190.3        | -0.23x  | 1.65x   | 190.3        | -0.21x  | 1.64x   | 147.0 | -0.28x  | -0.45x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      13
fp64      2
other     23
**total** **38**
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

.. _type-fp64mp2-25:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-25:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ======== ======= ========== ========== ====
Class       Count    Percent Max RelErr Avg RelErr Bits
=========== ======== ======= ========== ========== ====
normal (OK) 16769024 100.00% 0.00e+00   0.00e+00   106
TOTAL       16769024 100.00%
=========== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200  | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+=======+=========+==========+
| GFLOPS    | 70.2         | -0.30x  | -0.44x   | 46.1         | -0.27x  | -0.16x   | 693.1 | -0.32x  | 2.50x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| ev/clk/SM | 0.15         | -0.30x  | -0.44x   | 0.15         | -0.27x  | -0.16x   | 2.38  | -0.32x  | 2.50x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+
| clk/ev    | 960.6        | -0.33x  | -0.45x   | 951.5        | -0.33x  | -0.53x   | 172.9 | -0.38x  | 3.85x    |
+-----------+--------------+---------+----------+--------------+---------+----------+-------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      11
other     26
**total** **37**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-ldexp:

LDEXP
~~~~~

.. _type-fp32mp2-54:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-54:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          2222529003 68.64%  1.00e-13   2.72e-19   43
output special       1015103359 31.35%  --         --         --
input special        16225      5e-04%  --         --         --
output denormal      119527     4e-03%  1.00e+00   6.30e-04   0
input denormal       122385     4e-03%  5.96e-08   6.69e-09   24
output near denormal 2982       9e-05%  1.19e-07   8.24e-08   23
cancellation         9038       3e-04%  5.74e-08   2.23e-10   24
TOTAL                3237902519 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 394.9        | -0.04x  | 2.08x   | 259.8        | -0.04x  | 2.08x   | 1384.9 | -0.22x  | -0.40x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 0.86         | -0.04x  | 2.08x   | 0.86         | -0.04x  | 2.08x   | 4.76   | -0.22x  | -0.40x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 152.5        | -0.16x  | 1.94x   | 149.9        | -0.17x  | 1.97x   | 57.5   | -0.44x  | -0.80x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      22
fp64      9
other     51
**total** **82**
========= ======

**Special Values Table:**

+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **a\\b**  | -INF | -maxN | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN | +INF | QNAN     |
+===========+======+=======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=======+======+==========+
| **-INF**  | -inf | -inf  | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf  | -inf | -inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxN** | -0   | -0    | -1.7e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -inf     | -inf  | -inf | -3.4e+38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-1**    | -0   | -0    | -0.5     | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -2       | -inf  | -inf | -1       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minN** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxD** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minD** | -0   | -0    | -0       | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -2.8e-45 | -inf  | -inf | -1.4e-45 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-0**    | -0   | -0    | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0    | -0   | -0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+0**    | +0   | +0    | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0    | +0   | +0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minD** | +0   | +0    | +0       | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 2.8e-45  | +inf  | +inf | 1.4e-45  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxD** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minN** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+1**    | +0   | +0    | 0.5      | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | +inf  | +inf | 1        |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxN** | +0   | +0    | 1.7e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf     | +inf  | +inf | 3.4e+38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+INF**  | +inf | +inf  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf  | +inf | +inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **QNAN**  | nan  | nan   | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan   | nan  | nan      |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-log10:

LOG10
~~~~~

.. _type-fp32mp2-55:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-55:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2139095037 49.80%  5.15e-14   1.47e-15   44
output special 2139095043 49.80%  --         --         --
input special  16777216   0.39%   --         --         --
TOTAL          4294967296 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 300.5        | -0.15x  | 12.35x  | 191.2        | -0.15x  | 11.95x  | 185.4 | -0.15x  | -0.55x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.66         | -0.15x  | 12.35x  | 0.64         | -0.15x  | 11.95x  | 0.64  | -0.15x  | -0.55x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 395.7        | -0.24x  | 5.33x   | 384.3        | -0.24x  | 5.51x   | 385.1 | -0.24x  | -0.75x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      145
fp64      0
other     21
**total** **166**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-log1p:

LOG1P
~~~~~

.. _type-fp32mp2-56:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-56:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============= ========== ======= ========== ========== ====
Class         Count      Percent Max RelErr Avg RelErr Bits
============= ========== ======= ========== ========== ====
normal (OK)   3187671040 100.00% 8.71e-14   4.48e-16   43
input special 1          3e-08%  --         --         --
TOTAL         3187671041 100.00%
============= ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 234.9        | -0.11x  | 9.35x   | 139.6        | -0.10x  | 8.45x   | 135.6 | -0.10x  | -0.41x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.51         | -0.11x  | 9.35x   | 0.46         | -0.10x  | 8.45x   | 0.47  | -0.10x  | -0.41x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 545.5        | -0.16x  | 3.87x   | 538.3        | -0.16x  | 3.91x   | 538.8 | -0.16x  | -0.62x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      288
fp64      0
other     52
**total** **340**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-log2:

LOG2
~~~~

.. _type-fp32mp2-57:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-57:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    2139095037 49.80%  5.01e-14   1.37e-15   44
output special 2139095043 49.80%  --         --         --
input special  16777216   0.39%   --         --         --
TOTAL          4294967296 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 300.4        | -0.16x  | 12.34x  | 191.3        | -0.15x  | 11.96x  | 184.7 | -0.15x  | -0.55x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.66         | -0.16x  | 12.34x  | 0.64         | -0.15x  | 11.96x  | 0.64  | -0.15x  | -0.55x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 395.8        | -0.25x  | 5.33x   | 384.0        | -0.25x  | 5.51x   | 384.8 | -0.25x  | -0.75x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      145
fp64      0
other     21
**total** **166**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-remainder:

REMAINDER
~~~~~~~~~

.. _type-fp32mp2-58:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-58:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=========== ========== ======= ========== ========== ====
Class       Count      Percent Max RelErr Avg RelErr Bits
=========== ========== ======= ========== ========== ====
normal (OK) 4188533852 100.00% 2.93e-15   7.00e-25   48
TOTAL       4188533852 100.00%
=========== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 1162.2       | -0.32x  | 9.16x   | 628.0        | -0.27x  | 7.54x   | 604.8 | -0.27x  | -0.65x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 2.54         | -0.32x  | 9.16x   | 2.09         | -0.27x  | 7.54x   | 2.08  | -0.27x  | -0.65x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 221.3        | -0.36x  | 2.11x   | 217.1        | -0.35x  | 2.12x   | 216.0 | -0.35x  | -0.53x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      65
fp64      0
other     262
**total** **327**
========= =======

**Special Values Table:**

+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **a\\b**  | -INF     | -maxN    | -1       | -minN    | -maxD    | -minD | -0  | +0  | +minD | +maxD    | +minN    | +1       | +maxN    | +INF     | QNAN |
+===========+==========+==========+==========+==========+==========+=======+=====+=====+=======+==========+==========+==========+==========+==========+======+
| **-INF**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-maxN** | -3.4e+38 | -0       | -0       | -0       | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -0       | -0       | -0       | -3.4e+38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-1**    | -1       | -1       | -0       | -0       | -2.9e-42 | -0    | nan | nan | -0    | -2.9e-42 | -0       | -0       | -1       | -1       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-minN** | -1.2e-38 | -1.2e-38 | -1.2e-38 | -0       | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -0       | -1.2e-38 | -1.2e-38 | -1.2e-38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-maxD** | -1.2e-38 | -1.2e-38 | -1.2e-38 | 1.4e-45  | -0       | -0    | nan | nan | -0    | -0       | 1.4e-45  | -1.2e-38 | -1.2e-38 | -1.2e-38 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-minD** | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -0    | nan | nan | -0    | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **-0**    | -0       | -0       | -0       | -0       | -0       | -0    | nan | nan | -0    | -0       | -0       | -0       | -0       | -0       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+0**    | +0       | +0       | +0       | +0       | +0       | +0    | nan | nan | +0    | +0       | +0       | +0       | +0       | +0       | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+minD** | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+maxD** | 1.2e-38  | 1.2e-38  | 1.2e-38  | -1.4e-45 | +0       | +0    | nan | nan | +0    | +0       | -1.4e-45 | 1.2e-38  | 1.2e-38  | 1.2e-38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+minN** | 1.2e-38  | 1.2e-38  | 1.2e-38  | +0       | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | +0       | 1.2e-38  | 1.2e-38  | 1.2e-38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+1**    | 1        | 1        | +0       | +0       | 2.9e-42  | +0    | nan | nan | +0    | 2.9e-42  | +0       | +0       | 1        | 1        | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+maxN** | 3.4e+38  | +0       | +0       | +0       | 1.4e-45  | +0    | nan | nan | +0    | 1.4e-45  | +0       | +0       | +0       | 3.4e+38  | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **+INF**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+
| **QNAN**  | nan      | nan      | nan      | nan      | nan      | nan   | nan | nan | nan   | nan      | nan      | nan      | nan      | nan      | nan  |
+-----------+----------+----------+----------+----------+----------+-------+-----+-----+-------+----------+----------+----------+----------+----------+------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-scalbln:

SCALBLN
~~~~~~~

.. _type-fp32mp2-59:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-59:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          2222529003 68.64%  1.00e-13   2.72e-19   43
output special       1015103359 31.35%  --         --         --
input special        16225      5e-04%  --         --         --
output denormal      119527     4e-03%  1.00e+00   6.30e-04   0
input denormal       122385     4e-03%  5.96e-08   6.69e-09   24
output near denormal 2982       9e-05%  1.19e-07   8.24e-08   23
cancellation         9038       3e-04%  5.74e-08   2.23e-10   24
TOTAL                3237902519 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 394.8        | -0.04x  | 2.08x   | 259.8        | -0.04x  | 2.08x   | 1374.7 | -0.22x  | -0.40x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 0.86         | -0.04x  | 2.08x   | 0.86         | -0.04x  | 2.08x   | 4.73   | -0.22x  | -0.40x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 152.7        | -0.16x  | 1.93x   | 149.6        | -0.17x  | 1.97x   | 57.5   | -0.44x  | -0.80x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      22
fp64      9
other     51
**total** **82**
========= ======

**Special Values Table:**

+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **a\\b**  | -INF | -maxN | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN | +INF | QNAN     |
+===========+======+=======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=======+======+==========+
| **-INF**  | -inf | -inf  | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf  | -inf | -inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxN** | -0   | -0    | -1.7e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -inf     | -inf  | -inf | -3.4e+38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-1**    | -0   | -0    | -0.5     | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -2       | -inf  | -inf | -1       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minN** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxD** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minD** | -0   | -0    | -0       | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -2.8e-45 | -inf  | -inf | -1.4e-45 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-0**    | -0   | -0    | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0    | -0   | -0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+0**    | +0   | +0    | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0    | +0   | +0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minD** | +0   | +0    | +0       | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 2.8e-45  | +inf  | +inf | 1.4e-45  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxD** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minN** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+1**    | +0   | +0    | 0.5      | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | +inf  | +inf | 1        |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxN** | +0   | +0    | 1.7e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf     | +inf  | +inf | 3.4e+38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+INF**  | +inf | +inf  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf  | +inf | +inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **QNAN**  | nan  | nan   | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan   | nan  | nan      |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+

.. _type-fp64mp2-26:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp64mp2-26:

Type: fp64mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=============== ======== ======= ========== ========== ====
Class           Count    Percent Max RelErr Avg RelErr Bits
=============== ======== ======= ========== ========== ====
normal (OK)     15534530 96.21%  0.00e+00   0.00e+00   106
output special  610078   3.78%   --         --         --
output denormal 1035     6e-03%  2.27e-13   -3.64e-15  42
TOTAL           16145643 100.00%
=============== ======== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| Metric    | RTX PRO 6000 | vs fp64 | vs fp128 | B300 SXM6 AC | vs fp64 | vs fp128 | B200   | vs fp64 | vs fp128 |
+===========+==============+=========+==========+==============+=========+==========+========+=========+==========+
| GFLOPS    | 84.6         | -0.44x  | 3.88x    | 55.6         | -0.44x  | 3.88x    | 1506.1 | -0.45x  | 10.77x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| ev/clk/SM | 0.19         | -0.44x  | 3.88x    | 0.18         | -0.44x  | 3.88x    | 5.18   | -0.45x  | 10.77x   |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+
| clk/ev    | 636.4        | -0.46x  | 3.47x    | 636.8        | -0.46x  | 3.45x    | 51.5   | -0.89x  | 6.32x    |
+-----------+--------------+---------+----------+--------------+---------+----------+--------+---------+----------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      0
fp64      14
other     22
**total** **36**
========= ======

**Special Values Table:**

+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **a\\b**  | -INF      | -maxN     | -1        | -minN     | -maxD     | -minD     | -0        | +0        | +minD     | +maxD     | +minN     | +1        | +maxN     | +INF      | QNAN      |
+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+===========+
| **-INF**  | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      | -inf      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-maxN** | -8.8e+217 | -8.8e+217 | -9.0e+307 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -1.8e+308 | -inf      | -inf      | -inf      | -1.8e+308 |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-1**    | -4.9e-91  | -4.9e-91  | -0.5      | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -1        | -2        | -2.0e+90  | -2.0e+90  | -1        |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-minN** | -0        | -0        | -1.1e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.5e-308 | -4.5e-218 | -4.5e-218 | -2.2e-308 |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-maxD** | -0        | -0        | -1.1e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -2.2e-308 | -4.5e-308 | -4.5e-218 | -4.5e-218 | -2.2e-308 |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-minD** | -0        | -0        | -0        | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -4.9e-324 | -9.9e-324 | -1.0e-233 | -1.0e-233 | -4.9e-324 |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **-0**    | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        | -0        |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+0**    | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        | +0        |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+minD** | +0        | +0        | +0        | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 4.9e-324  | 9.9e-324  | 1.0e-233  | 1.0e-233  | 4.9e-324  |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+maxD** | +0        | +0        | 1.1e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-218  | 4.5e-218  | 2.2e-308  |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+minN** | +0        | +0        | 1.1e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 2.2e-308  | 4.5e-308  | 4.5e-218  | 4.5e-218  | 2.2e-308  |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+1**    | 4.9e-91   | 4.9e-91   | 0.5       | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 1         | 2         | 2.0e+90   | 2.0e+90   | 1         |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+maxN** | 8.8e+217  | 8.8e+217  | 9.0e+307  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | 1.8e+308  | +inf      | +inf      | +inf      | 1.8e+308  |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **+INF**  | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      | +inf      |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| **QNAN**  | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       | nan       |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-scalbn:

SCALBN
~~~~~~

.. _type-fp32mp2-60:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-60:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

==================== ========== ======= ========== ========== ====
Class                Count      Percent Max RelErr Avg RelErr Bits
==================== ========== ======= ========== ========== ====
normal (OK)          2222529003 68.64%  1.00e-13   2.72e-19   43
output special       1015103359 31.35%  --         --         --
input special        16225      5e-04%  --         --         --
output denormal      119527     4e-03%  1.00e+00   6.30e-04   0
input denormal       122385     4e-03%  5.96e-08   6.69e-09   24
output near denormal 2982       9e-05%  1.19e-07   8.24e-08   23
cancellation         9038       3e-04%  5.74e-08   2.23e-10   24
TOTAL                3237902519 100.00%
==================== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200   | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+========+=========+=========+
| GFLOPS    | 394.9        | -0.04x  | 2.08x   | 259.8        | -0.04x  | 2.08x   | 1385.4 | -0.22x  | -0.41x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| ev/clk/SM | 0.86         | -0.04x  | 2.08x   | 0.86         | -0.04x  | 2.08x   | 4.76   | -0.22x  | -0.41x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+
| clk/ev    | 152.3        | -0.16x  | 1.94x   | 150.1        | -0.17x  | 1.97x   | 57.5   | -0.43x  | -0.80x  |
+-----------+--------------+---------+---------+--------------+---------+---------+--------+---------+---------+

**SASS Instructions:**

========= ======
Class     Count
========= ======
fp32      22
fp64      9
other     51
**total** **82**
========= ======

**Special Values Table:**

+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **a\\b**  | -INF | -maxN | -1       | -minN    | -maxD    | -minD    | -0       | +0       | +minD    | +maxD    | +minN    | +1       | +maxN | +INF | QNAN     |
+===========+======+=======+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+=======+======+==========+
| **-INF**  | -inf | -inf  | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf     | -inf  | -inf | -inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxN** | -0   | -0    | -1.7e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -3.4e+38 | -inf     | -inf  | -inf | -3.4e+38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-1**    | -0   | -0    | -0.5     | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -1       | -2       | -inf  | -inf | -1       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minN** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-maxD** | -0   | -0    | -5.9e-39 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -1.2e-38 | -2.4e-38 | -inf  | -inf | -1.2e-38 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-minD** | -0   | -0    | -0       | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -1.4e-45 | -2.8e-45 | -inf  | -inf | -1.4e-45 |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **-0**    | -0   | -0    | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0       | -0    | -0   | -0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+0**    | +0   | +0    | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0       | +0    | +0   | +0       |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minD** | +0   | +0    | +0       | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 1.4e-45  | 2.8e-45  | +inf  | +inf | 1.4e-45  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxD** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+minN** | +0   | +0    | 5.9e-39  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 1.2e-38  | 2.4e-38  | +inf  | +inf | 1.2e-38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+1**    | +0   | +0    | 0.5      | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 2        | +inf  | +inf | 1        |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+maxN** | +0   | +0    | 1.7e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | 3.4e+38  | +inf     | +inf  | +inf | 3.4e+38  |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **+INF**  | +inf | +inf  | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf     | +inf  | +inf | +inf     |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+
| **QNAN**  | nan  | nan   | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan      | nan   | nan  | nan      |
+-----------+------+-------+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+-------+------+----------+

..

   *Note: ``fp64mp2`` is a thin wrapper over the system ``fp64`` (or ``fp128`` reference) math for this function and is omitted from the spec.*

--------------

.. _libcudacxx-extended-api-fp-fpmp-spec-sinh:

SINH
~~~~

.. _type-fp32mp2-61:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-61:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

=============== ========== ======= ========== ========== ====
Class           Count      Percent Max RelErr Avg RelErr Bits
=============== ========== ======= ========== ========== ====
normal (OK)     2216918324 99.80%  1.00e-13   7.17e-16   43
output special  181863     8e-03%  --         --         --
output near inf 22838      1e-03%  2.98e-13   2.27e-13   41
unclassified    4132326    0.19%   6.53e-13   1.57e-13   40
TOTAL           2221255351 100.00%
=============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 269.8        | -0.14x  | 10.30x  | 167.5        | -0.12x  | 9.73x   | 161.5 | -0.13x  | -0.52x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.59         | -0.14x  | 10.30x  | 0.56         | -0.12x  | 9.73x   | 0.56  | -0.13x  | -0.52x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 434.9        | -0.26x  | 5.07x   | 427.3        | -0.25x  | 5.17x   | 425.9 | -0.26x  | -0.93x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      288
fp64      0
other     23
**total** **311**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-tan:

TAN
~~~

.. _type-fp32mp2-62:

.. _libcudacxx-extended-api-fp-fpmp-spec-type-fp32mp2-62:

Type: fp32mp2
^^^^^^^^^^^^^

*Accuracy: ``def``*

**Measured Accuracy:**

============== ========== ======= ========== ========== ====
Class          Count      Percent Max RelErr Avg RelErr Bits
============== ========== ======= ========== ========== ====
normal (OK)    4199082652 98.54%  1.00e-13   4.81e-15   43
input near inf 961493     0.02%   2.72e-12   1.46e-13   38
unclassified   61368719   1.44%   1.75e-08   1.48e-13   25
TOTAL          4261412864 100.00%
============== ========== ======= ========== ========== ====

**Measured Performance:**

+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| Metric    | RTX PRO 6000 | vs fp32 | vs fp64 | B300 SXM6 AC | vs fp32 | vs fp64 | B200  | vs fp32 | vs fp64 |
+===========+==============+=========+=========+==============+=========+=========+=======+=========+=========+
| GFLOPS    | 188.7        | -0.10x  | 7.80x   | 109.8        | -0.08x  | 6.62x   | 107.0 | -0.08x  | -0.30x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| ev/clk/SM | 0.41         | -0.10x  | 7.80x   | 0.37         | -0.08x  | 6.62x   | 0.37  | -0.08x  | -0.30x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+
| clk/ev    | 582.9        | -0.22x  | 3.70x   | 562.6        | -0.23x  | 3.84x   | 560.9 | -0.24x  | -0.70x  |
+-----------+--------------+---------+---------+--------------+---------+---------+-------+---------+---------+

**SASS Instructions:**

========= =======
Class     Count
========= =======
fp32      242
fp64      0
other     231
**total** **473**
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

.. _libcudacxx-extended-api-fp-fpmp-spec-appendix-legends:

Appendix: Legends
-----------------

.. _libcudacxx-extended-api-fp-fpmp-spec-measured-accuracy-legend:

Measured Accuracy Legend
~~~~~~~~~~~~~~~~~~~~~~~~

.. _libcudacxx-extended-api-fp-fpmp-spec-pattern-dataset:

Pattern Dataset
^^^^^^^^^^^^^^^

Accuracy measurements use the **pattern dataset**, which provides exhaustive bit-pattern coverage across the IEEE-754 floating-point representation space. The dataset is designed to systematically test:

-  **Sign bit**: Both positive and negative values
-  **Exponent field**: Full range from denormals through maximum normal values
-  **Mantissa bits**: Both most significant and least significant bits

For multi-precision types (fp32mp2, fp64mp2), both the high and low components are generated with coordinated exponents to ensure proper normalization (\|lo\| < ulp(hi)/2).

**Sample distribution:**

+----------------------------------------+-----------------------------------------------------------+
| Region                                 | Coverage                                                  |
+========================================+===========================================================+
| Denormal inputs                        | Systematically tested via exponent field patterns         |
+----------------------------------------+-----------------------------------------------------------+
| Near-denormal (smallest normal binade) | Covered by exponent boundary patterns                     |
+----------------------------------------+-----------------------------------------------------------+
| Normal range                           | Dense coverage with sign/exponent/mantissa bit variations |
+----------------------------------------+-----------------------------------------------------------+
| Near-infinity (largest normal binade)  | Covered by exponent boundary patterns                     |
+----------------------------------------+-----------------------------------------------------------+
| Special values (INF, NaN)              | Tested separately in the Special Values Table             |
+----------------------------------------+-----------------------------------------------------------+

For binary functions, bits are divided equally between arguments. For example, with 32-bit rigor and a binary function, each argument receives 16 bits of pattern control, yielding ~4 billion test combinations (2^32 total samples).

*Note: For unary functions with a single 32-bit argument (e.g., ``int2mp``, ``uint2mp``), the pattern dataset provides* **exhaustive coverage** *of all 2^32 possible input values.*

.. _libcudacxx-extended-api-fp-fpmp-spec-classification-categories:

Classification Categories
^^^^^^^^^^^^^^^^^^^^^^^^^

The accuracy classification table shows only deviations from expected behavior: cases that exceed the warning threshold or have special value mismatches. If only ``normal (OK)`` is shown, all test cases passed within acceptable accuracy bounds.

+----------------------+-------------------------------------------------------------------------+
| Category             | Description                                                             |
+======================+=========================================================================+
| normal (OK)          | Result within warning threshold (acceptable accuracy)                   |
+----------------------+-------------------------------------------------------------------------+
| output special       | Output special value mismatch (INF or NaN)                              |
+----------------------+-------------------------------------------------------------------------+
| input special        | Special input (INF or NaN) causes result mismatch                       |
+----------------------+-------------------------------------------------------------------------+
| output denormal      | Output is denormal, accuracy loss                                       |
+----------------------+-------------------------------------------------------------------------+
| input denormal       | Denormal input causes accuracy loss                                     |
+----------------------+-------------------------------------------------------------------------+
| output near denormal | Output is in smallest normal binade and exceeds warning threshold       |
+----------------------+-------------------------------------------------------------------------+
| input near denormal  | Input is in smallest normal binade and result exceeds warning threshold |
+----------------------+-------------------------------------------------------------------------+
| output near inf      | Output is in largest normal binade and exceeds warning threshold        |
+----------------------+-------------------------------------------------------------------------+
| input near inf       | Input is in largest normal binade and result exceeds warning threshold  |
+----------------------+-------------------------------------------------------------------------+
| cancellation         | Cancellation case: result is much smaller than inputs                   |
+----------------------+-------------------------------------------------------------------------+
| unclassified         | Warning without identified cause (between warning and error threshold)  |
+----------------------+-------------------------------------------------------------------------+
| error (FAIL)         | Exceeds error threshold without any mitigating factor                   |
+----------------------+-------------------------------------------------------------------------+

**Table columns:**

+------------+-----------------------------------------------------------------+
| Column     | Description                                                     |
+============+=================================================================+
| Count      | Number of test cases in this category                           |
+------------+-----------------------------------------------------------------+
| Percent    | Percentage of total test cases                                  |
+------------+-----------------------------------------------------------------+
| Max RelErr | Maximum relative error observed in this category                |
+------------+-----------------------------------------------------------------+
| Avg RelErr | Average relative error in this category                         |
+------------+-----------------------------------------------------------------+
| Bits       | Minimum correct mantissa bits (derived from max relative error) |
+------------+-----------------------------------------------------------------+

.. _libcudacxx-extended-api-fp-fpmp-spec-special-values-legend-floating-point:

Special Values Legend (Floating Point)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _libcudacxx-extended-api-fp-fpmp-spec-special-values-legend-integer-conversions:

Special Values Legend (Integer Conversions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _libcudacxx-extended-api-fp-fpmp-spec-performance-metrics-legend:

Performance Metrics Legend
~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------+------------+----------------------------------------------------------------------------------------------------+
| Metric    | Type       | Description                                                                                        |
+===========+============+====================================================================================================+
| GFLOPS    | Throughput | Giga floating-point operations per second                                                          |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| ev/clk/SM | Throughput | Evaluations per clock cycle per SM                                                                 |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| clk/ev    | Latency    | Clock cycles per evaluation                                                                        |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| vs fp32   | Ratio      | Performance ratio compared to native fp32 operations                                               |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| vs fp64   | Ratio      | Performance ratio compared to native fp64 operations (for fp32mp2) or reference fp64 (for fp64mp2) |
+-----------+------------+----------------------------------------------------------------------------------------------------+
| vs fp128  | Ratio      | Performance ratio compared to reference fp128 operations (for fp64mp2)                             |
+-----------+------------+----------------------------------------------------------------------------------------------------+

*Note: Negative ratios indicate slower performance than the baseline.*

.. _libcudacxx-extended-api-fp-fpmp-spec-sass-instructions-legend:

SASS Instructions Legend
~~~~~~~~~~~~~~~~~~~~~~~~

Counts come from the generated SASS (``cuobjdump --dump-sass``). Only the primary ``<op>_device_impl`` function body, up to (but not including) its ``RET`` epilogue, is counted; ``NOP`` scheduling slots, trailing self-loop ``BRA`` padding, and any helper functions emitted in the same dump (libdevice slowpaths, outlined denormal handlers, etc.) are excluded — their cost is included only when they're actually called.

+-------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Class | Mnemonics                                                                                                                                                                                                                                                                                                                                                                         | Notes                                                                                                                                                                                                                                                                                                         |
+=======+===================================================================================================================================================================================================================================================================================================================================================================================+===============================================================================================================================================================================================================================================================================================================+
| fp32  | ``FADD``, ``FMUL``, ``FFMA``, ``FSET``, ``FSETP``, ``FMNMX``, ``FCMP``, ``FCHK``, ``FRND``, ``FSWZADD``, ``MUFU.{RCP,SQRT,RSQ,SIN,COS,EX2,LG2}``, plus conversion family (``F2F``, ``F2I``, ``I2F``, ``I2FP``, ``F2FP``, ``F2IP``, ``F2DP``, ``D2FP``) when the precision suffix is ``F32`` (e.g. ``F2I.F32.TRUNC``, ``I2FP.F32.S32``)                                            | Single-precision IEEE-754 ops. ``MUFU`` is the Multi-Function Unit (transcendentals & reciprocals). ``FSEL`` is *not* in this class — despite the ``F`` prefix it's a predicated 32-bit register MOV with no FP semantics, freely emitted by ptxas in pure bit-manipulation code, so it falls into ``other``. |
+-------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| fp64  | ``DADD``, ``DMUL``, ``DFMA``, ``DSET``, ``DSETP``, ``DMNMX``, ``DCMP``, ``F2D``, ``D2F``, ``I2D``, ``UI2D``, ``D2I``, ``D2UI``, ``MUFU.*64H`` (e.g. ``MUFU.RCP64H``, ``MUFU.RSQ64H`` — high-half helpers of fp64 transcendental sequences), plus any conversion-family op with ``F64`` in the suffix (e.g. ``F2I.F64.TRUNC``, ``I2F.F64.S32``, ``F2F.F64.F32``, ``I2FP.F64.S32``) | Double-precision IEEE-754 ops. F32↔F64 conversions are counted as fp64 since they exercise the fp64 datapath.                                                                                                                                                                                                 |
+-------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| other | everything else (integer ALU, memory, control flow, predicate/uniform, fp16 ``H*2``, ``FSEL``, etc.)                                                                                                                                                                                                                                                                              | Scheduling slots (``NOP``) are excluded entirely.                                                                                                                                                                                                                                                             |
+-------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

*Note: SASS counts are taken from the accuracy run directory (the ``--acc`` argument) and reflect the GPU architecture that directory was built for. Different GPUs may compile to different instruction counts; the spec shows one representative count per (function, type, method).*

.. _libcudacxx-extended-api-fp-fpmp-spec-sass-instructions-summary:

SASS Instructions Summary
~~~~~~~~~~~~~~~~~~~~~~~~~

Per-function instruction counts (NOPs excluded) for every ``(type, method)`` combination that exposes a *dedicated* implementation. Empty cells (``—``) mean either:

-  the combination wasn't built / tested, or
-  the combination is a wrapper over higher-precision system math — specifically the ``fp64mp2`` path of ``exp``/``log``/``pow``/``sin``/``cos``/``tanh``/``cbrt``/``rcbrt``/ ``erf``/``erfc``/``boys_f0``/``normcdfinv``/``floor``/``ceil``/ ``round``/``trunc``, which falls back to system ``fp64`` / reference ``fp128`` math and drags in arbitrary ``libm`` sub-routines whose count isn't comparable to the dedicated implementations (see :ref:`Mathematical Functions <libcudacxx-extended-api-fp-fpmp-spec-mathematical-functions>`).

.. raw:: html

   <table>
   <thead>
   <tr><th rowspan="3">Function</th><th colspan="9">fp32mp2</th><th colspan="9">fp64mp2</th></tr>
   <tr><th colspan="3">low</th><th colspan="3">def</th><th colspan="3">high</th><th colspan="3">low</th><th colspan="3">def</th><th colspan="3">high</th></tr>
   <tr><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th><th>fp32</th><th>fp64</th><th>other</th></tr>
   </thead>
   <tbody>
   <tr><td><code>add</code></td><td align="right">8</td><td align="right">0</td><td align="right">1</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">8</td><td align="right">4</td><td align="right">0</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td></tr>
   <tr><td><code>sub</code></td><td align="right">8</td><td align="right">0</td><td align="right">1</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">8</td><td align="right">4</td><td align="right">0</td><td align="right">11</td><td align="right">0</td><td align="right">0</td><td align="right">20</td><td align="right">0</td></tr>
   <tr><td><code>mul</code></td><td align="right">5</td><td align="right">0</td><td align="right">1</td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">5</td><td align="right">2</td><td align="right">0</td><td align="right">9</td><td align="right">0</td><td align="right">0</td><td align="right">9</td><td align="right">0</td></tr>
   <tr><td><code>div</code></td><td align="right">6</td><td align="right">0</td><td align="right">1</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">23</td><td align="right">0</td><td align="right">23</td><td align="right">1</td><td align="right">11</td><td align="right">27</td><td align="right">1</td><td align="right">18</td><td align="right">25</td><td align="right">1</td><td align="right">28</td><td align="right">53</td></tr>
   <tr><td><code>acc</code></td><td align="right">7</td><td align="right">0</td><td align="right">1</td><td align="right">10</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">7</td><td align="right">2</td><td align="right">0</td><td align="right">10</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td></tr>
   <tr><td><code>fma</code></td><td align="right">16</td><td align="right">0</td><td align="right">1</td><td align="right">19</td><td align="right">0</td><td align="right">0</td><td align="right">37</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">16</td><td align="right">2</td><td align="right">0</td><td align="right">19</td><td align="right">0</td><td align="right">0</td><td align="right">37</td><td align="right">0</td></tr>
   <tr><td><code>mad</code></td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">16</td><td align="right">0</td><td align="right">0</td><td align="right">29</td><td align="right">0</td><td align="right">0</td><td align="right">0</td><td align="right">13</td><td align="right">0</td><td align="right">0</td><td align="right">16</td><td align="right">0</td><td align="right">0</td><td align="right">29</td><td align="right">0</td></tr>
   <tr><td><code>sqrt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">18</td><td align="right">0</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">23</td><td align="right">24</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>rsqrt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">17</td><td align="right">0</td><td align="right">0</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">22</td><td align="right">22</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>exp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">136</td><td align="right">0</td><td align="right">14</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>log</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">135</td><td align="right">0</td><td align="right">20</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>pow</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">305</td><td align="right">0</td><td align="right">69</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>cbrt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">75</td><td align="right">0</td><td align="right">27</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>rcbrt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">95</td><td align="right">0</td><td align="right">35</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>sin</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">226</td><td align="right">0</td><td align="right">230</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>cos</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">226</td><td align="right">0</td><td align="right">233</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>tanh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">310</td><td align="right">0</td><td align="right">29</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>erf</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">560</td><td align="right">0</td><td align="right">22</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>erfc</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">525</td><td align="right">0</td><td align="right">21</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>boys_f0</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">749</td><td align="right">0</td><td align="right">11</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>normcdfinv</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">735</td><td align="right">0</td><td align="right">36</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>floor</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">28</td><td align="right">0</td><td align="right">13</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ceil</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">28</td><td align="right">0</td><td align="right">13</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>round</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">71</td><td align="right">0</td><td align="right">29</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>trunc</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">35</td><td align="right">0</td><td align="right">23</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>eq</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">2</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ne</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">2</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>lt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>le</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>gt</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ge</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">3</td><td align="right">0</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">2</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2int</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">5</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2uint</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">5</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2ll</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2ull</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">5</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>int2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">1</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>uint2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">2</td><td align="right">0</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">1</td><td align="right">1</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ll2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ull2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">3</td><td align="right">3</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>mp2fp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">7</td><td align="right">3</td><td align="right">31</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">86</td><td align="right">1001</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>fp2mp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">6</td><td align="right">1</td><td align="right">18</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">103</td><td align="right">1069</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>acos</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">454</td><td align="right">0</td><td align="right">10</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>acosh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">517</td><td align="right">0</td><td align="right">85</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>asin</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">432</td><td align="right">0</td><td align="right">9</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>asinh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">521</td><td align="right">0</td><td align="right">88</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>atan</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">264</td><td align="right">0</td><td align="right">6</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>atan2</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">299</td><td align="right">0</td><td align="right">46</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>atanh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">463</td><td align="right">0</td><td align="right">68</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>cosh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">160</td><td align="right">0</td><td align="right">18</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>exp10</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">198</td><td align="right">0</td><td align="right">22</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>exp2</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">126</td><td align="right">0</td><td align="right">22</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>expm1</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">285</td><td align="right">0</td><td align="right">28</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>fmod</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">41</td><td align="right">0</td><td align="right">224</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>frexp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">13</td><td align="right">2</td><td align="right">23</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">11</td><td align="right">26</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>ldexp</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">22</td><td align="right">9</td><td align="right">51</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>log10</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">145</td><td align="right">0</td><td align="right">21</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>log1p</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">288</td><td align="right">0</td><td align="right">52</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>log2</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">145</td><td align="right">0</td><td align="right">21</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>remainder</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">65</td><td align="right">0</td><td align="right">262</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>scalbln</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">22</td><td align="right">9</td><td align="right">51</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">0</td><td align="right">14</td><td align="right">22</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>scalbn</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">22</td><td align="right">9</td><td align="right">51</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>sinh</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">288</td><td align="right">0</td><td align="right">23</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   <tr><td><code>tan</code></td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">242</td><td align="right">0</td><td align="right">231</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td><td align="right">—</td></tr>
   </tbody>
   </table>
