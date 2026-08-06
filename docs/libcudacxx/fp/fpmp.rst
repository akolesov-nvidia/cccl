CUDA Multi-Precision Floating-Point Library
===========================================

A high-performance C++ library for extended-precision floating-point arithmetic on NVIDIA GPUs and CPUs. It provides double-float (``fp32mp2``) and double-double (``fp64mp2``) data types that achieve higher precision than standard IEEE-754 types while maintaining good memory efficiency and performance.

Overview
--------

The CUDA Multi-Precision library implements multi-component floating-point arithmetic using pairs of IEEE-754 floating-point values. This approach enables applications to achieve extended precision arithmetic while potentially outperforming native higher-precision operations on hardware with limited resources.

Why Multi-Precision?
~~~~~~~~~~~~~~~~~~~~

Modern GPU architectures often have reduced FP64 (double precision) compute resources compared to FP32 resources. This library addresses this limitation by:

-  **Leveraging abundant FP32 units**: Utilizes the more plentiful single-precision floating-point units available on most GPUs
-  **Comparable precision**: Achieves slightly less than IEEE-754 double precision (``fp32mp2``)
-  **Performance**: Can outperform native FP64 on GPUs with limited double-precision capabilities

Precision beyond FP64 is also challenging on GPUs: there is typically no native IEEE-754 binary128 (FP128) hardware, and fully IEEE-correct software emulation is expensive. This library also addresses that limitation by:

-  **Using existing FP64 units**: Builds extended precision from hardware FP64 operations (double-double, ``fp64mp2``)
-  **Quad-like precision**: Achieves ~104 bits of mantissa (vs 113 bits for IEEE-754 binary128), suitable for many “FP128-like” use cases
-  **Performance**: Can outperform fully IEEE-compliant binary128 software emulation on platforms where FP64 is available

Key Features
~~~~~~~~~~~~

-  **Multi-precision data types**:

   -  ``fp32mp2``: Double-float precision (up to ~46 bit mantissa) using pairs of floats
   -  ``fp64mp2``: Double-double precision (up to ~104 bit mantissa) using pairs of doubles

-  **Comprehensive operations**:

   -  Basic arithmetic: addition, subtraction, multiplication, division
   -  Core math: sqrt, rsqrt, fma/mad
   -  Math extensions (in ``fpmp_math.h``): dedicated fp32mp2 implementations for ``exp``, ``log``, ``pow``, ``sin``, ``cos``, ``tan``, ``sincos``, ``asin``, ``acos``, ``atan``, ``atan2``, ``tanh``, ``erf``, ``erfc``, ``normcdfinv``, ``icdf``, ``cbrt``, ``rcbrt``, ``floor``, ``ceil``, ``round``, ``trunc``, ``fabs``, ``fmin``, ``fmax``, ``min``, ``max``; remaining CUDA math functions are currently provided as **placeholders** delegating to standard higher precision math
   -  Comparison operators: ==, !=, <, <=, >, >=
   -  Type conversions: double, float, integer types
   -  Atomic operations: atomicAdd/atomicSub (CUDA device only)

-  **Cross-platform compatibility**:

   -  GPU (CUDA device code)
   -  CPU (host code)
   -  Both header-only and library based implementations for easy integration

-  **Performance optimized**:

   -  Error-free transformation algorithms (Dekker, Karp-Markstein)
   -  FMA-accelerated operations where available
   -  Fully inlined or LTO linked for minimal overhead

Namespace
~~~~~~~~~

All FP SDK types live in the ``cuda::experimental`` namespace (it will be promoted
to ``cuda::`` later). The examples below assume ``using namespace cuda::experimental;``.
The accuracy-explicit free functions (``add``, ``sub``, ``mul``, ``div``, ``fma``,
``mad``) additionally live in ``cuda::experimental::fpmp`` — bring them in with
``using namespace cuda::experimental::fpmp;`` where shown.

Table of Contents
-----------------

1.  `Quick Start <#quick-start>`__
2.  `Implementation Status <#implementation-status>`__
3.  `Data Types <#data-types>`__
4.  `Operations <#operations>`__
5.  `Usage Examples <#usage-examples>`__
6.  `Performance Characteristics <#performance-characteristics>`__
7.  `Build Instructions <#build-instructions>`__
8.  `File Structure <#file-structure>`__
9.  `Examples <#examples>`__
10. `References <#references>`__

Quick Start
-----------

Basic Usage (Double-Float)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   #include <cuda/fpmp>       // Core: types, operators, sqrt, rsqrt, fma
   #include <cuda/fpmp_math>  // Math: exp, log, trig, pow, ... (also pulls in <cuda/fpmp>)

   using namespace cuda::experimental;

   // Double-float arithmetic. double -> fp32mp2 is a narrowing conversion, so it
   // is explicit by default (see CCCL_FPMP_EXPLICIT_CASTS). The cast stays constexpr.
   fp32mp2 a = fp32mp2(1.23456789123456789);
   fp32mp2 b = fp32mp2(9.87654321987654321);

   auto sum = a + b;           // High-precision addition (core)
   auto product = a * b;        // High-precision multiplication (core)
   auto result = fma(a, b, sum); // Fused multiply-add: a*b + sum (core)
   auto root = sqrt(a);         // Square root (core)

   // Mathematical functions (require <cuda/fpmp_math>)
   auto exponential = exp(a);   // Exponential function
   auto sine = sin(a);          // Sine function
   auto logarithm = log(a);     // Natural logarithm

   // Convert back to standard types
   double d = static_cast<double>(result);
   float f = static_cast<float>(result);

Double-Double Precision (Quad-like)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   #include <cuda/fpmp>

   // Type aliases for double-double
   using fp64mp2 = fpmp2<double, fpmp2_accuracy::def>;

   // Double-double arithmetic (~106-bit mantissa)
   fp64mp2 a = 1.234567890123456789;
   fp64mp2 b = 9.876543210987654321;

   auto sum = a + b;           // Quad-precision-like addition
   auto product = a * b;       // Quad-precision-like multiplication
   auto root = sqrt(a);        // High-precision square root

CUDA Kernel Example
~~~~~~~~~~~~~~~~~~~

.. code:: c++

   __global__ void kernel(fp32mp2* input, fp32mp2* output, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           fp32mp2 x = input[idx];
           // Perform high-precision computation
           output[idx] = sqrt(x * x + fp32mp2(1.0));
       }
   }

Implementation Status
---------------------


fp32mp2 (Double-Float) & fp64mp2 (Double-Double)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ✅ All arithmetic operations (add, subtract, multiply, divide) & core math functions (sqrt, rsqrt, fma, mad)
-  ✅ Multiple accuracy levels (low, mid [default], high) including error-free transformations (Dekker, Karp-Markstein algorithms)
-  ✅ Comparison operators
-  ✅ Type conversions
-  ✅ ``cuda::std::numeric_limits<>`` specialization (``fpmp_limits.h``)
-  ✅ Atomic operations (CUDA device only)
-  ✅ Dedicated fp32mp2 math in ``fpmp_math.h``: ``exp``, ``log``, ``pow``, ``sin``, ``cos``, ``tan``, ``sincos``, ``tanh``, ``erf``, ``erfc``, ``normcdfinv``, ``icdf``, ``cbrt``, ``rcbrt``, ``floor``, ``ceil``, ``round``, ``trunc``, ``fabs``, ``fmin``, ``fmax``, ``min``, ``max``
-  ⬜ Remaining CUDA math functions are present in ``fpmp_math.h`` but are currently **placeholders** (delegating to standard higher precision math)
-  ✅ Examples

Data Types
----------

fp32mp2 (Double-Float)
~~~~~~~~~~~~~~~~~~~~~~~~

Represents a number as the unevaluated sum of two single-precision floats stored directly in the class: ``value = hi + lo``

-  **Precision**: up to ~46 bits of mantissa (between IEEE-754 single and double precision)
-  **Range**: Same as single precision exponent range (approximately ±10^38)
-  **Memory**: 8 bytes (2 × float), 8-byte aligned
-  **Storage**: Direct members ``mp2_hi`` and ``mp2_lo``
-  **Best for**: Applications requiring double-precision accuracy with better performance on FP32-optimized hardware

fp64mp2 (Double-Double)
~~~~~~~~~~~~~~~~~~~~~~~~~

Represents a number as the unevaluated sum of two double-precision floats: ``value = hi + lo``

-  **Precision**: up to ~104 bits of mantissa (between IEEE-754 double and quad precision)
-  **Range**: Same as double precision exponent range (approximately ±10^308)
-  **Memory**: 16 bytes (2 × double), 8-byte aligned
-  **Storage**: Direct members ``mp2_hi`` and ``mp2_lo``
-  **Best for**: Scientific computing requiring quad-precision-like accuracy without hardware quad support

Accuracy Levels
~~~~~~~~~~~~~~~

The library provides three accuracy levels via template parameter:

-  ``fp32mp2`` / ``fp64mp2`` (default, == ``mid``): Dekker-based normalization, good balance of speed and accuracy
-  ``fp32mp2_low`` / ``fp64mp2_low``: Fast mode without renormalization, highest performance
-  ``fp32mp2_high`` / ``fp64mp2_high``: More accurate implementations using different algorithms, lower performance

.. code:: c++

   // Double-float variants (double -> fp32mp2 is explicit by default)
   fp32mp2 a = fp32mp2(1.0);            // Default accuracy (mid)
   fp32mp2_low b = fp32mp2_low(2.0);    // Low accuracy
   fp32mp2_high c = fp32mp2_high(3.0);  // High accuracy

   // Double-double variants
   fp64mp2 x = 1.0;          // Default accuracy (mid)
   fp64mp2_low y = 2.0;      // Low accuracy
   fp64mp2_high z = 3.0;     // High accuracy

Operations
----------

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

All operations maintain error-free transformations to ensure accuracy:

.. code:: c++

   fp32mp2 a, b;

   // Basic arithmetic
   auto sum   = a + b;          // Addition
   auto diff  = a - b;          // Subtraction
   auto prod  = a * b;          // Multiplication
   auto quot  = a / b;          // Division
   auto neg   = -a;             // Negation
   auto root  = sqrt(a);        // Square root
   auto rroot = rsqrt(a);       // Reciprocal square root (1/sqrt(a))

   auto fma_result = fma(a, b, c);  // a * b + c (fused multiply-add)
   auto mad_result = mad(a, b, c);  // a * b + c (multiply-add)

   // Optimized single-component accumulate (faster than full mp2+mp2 addition)
   a += 1.5f;                   // Accumulate single float into mp2 (saves ~6 ops vs a + fp32mp2(1.5f))
   a -= 0.5f;                   // Subtract single float from mp2

Accuracy-Explicit Arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override the arithmetic accuracy for a single operation without changing the type.
This is useful in performance-critical code where a low-accuracy type is used for most
operations, but specific steps (e.g., argument reductions) require higher accuracy:

.. code:: c++

   using namespace cuda::experimental;
   using namespace cuda::experimental::fpmp;  // accuracy-explicit add/sub/mul/div/fma/mad
   using ffloat = fp32mp2_low;  // low-accuracy type for bulk computation

   ffloat a = ..., b = ...;

   // Override accuracy for a single operation — result stays fp32mp2_low
   ffloat diff = sub<fpmp2_accuracy::high>(a, b);  // high-accuracy subtraction
   ffloat sum  = add<fpmp2_accuracy::def>(a, b);   // default (Dekker) addition
   ffloat prod = mul<fpmp2_accuracy::high>(a, b);  // high-accuracy multiplication
   ffloat quot = div<fpmp2_accuracy::def>(a, b);   // default division

   // Also available for fma/mad
   ffloat r = fma<fpmp2_accuracy::high>(a, b, c);  // high-accuracy fused multiply-add
   ffloat s = mad<fpmp2_accuracy::def>(a, b, c);   // default multiply-add

This approach calls the underlying C-style API directly on scalar ``(hi, lo)`` pairs
without instantiating a second ``fpmp2`` class specialization, avoiding the
register pressure issues that arise from mixing types on GPU.

Mathematical Functions
~~~~~~~~~~~~~~~~~~~~~~

The transcendental functions are provided by ``<cuda/fpmp_math>`` (which also
pulls in the core ``<cuda/fpmp>``):

.. code:: c++

   #include <cuda/fpmp_math>

   // Exponential and logarithmic
   auto exp_val   = exp(a);      // Exponential function (e^x)
   auto log_val   = log(a);      // Natural logarithm
   auto log2_val  = log2(a);    // Base-2 logarithm
   auto log10_val = log10(a);  // Base-10 logarithm
   auto log1p_val = log1p(a);  // log(1+x)

   // Power functions
   auto pow_val  = pow(a, b);   // Power function (x^y)
   auto cbrt_val = cbrt(a);    // Cube root

   // Trigonometric functions
   auto sin_val   = sin(a);      // Sine
   auto cos_val   = cos(a);      // Cosine
   auto asin_val  = asin(a);    // Arcsine
   auto acos_val  = acos(a);    // Arccosine
   auto atan_val  = atan(a);    // Arctangent
   auto atan2_val = atan2(a,b);// Two-argument arctangent
   auto max_val   = max(a,b);   // std::max-like (first-arg tie/unordered)
   auto min_val   = min(a,b);   // std::min-like (first-arg tie/unordered)
   auto fmax_val  = fmax(a,b);  // IEEE fmax semantics (NaN-aware)
   auto fmin_val  = fmin(a,b);  // IEEE fmin semantics (NaN-aware)

   // Simultaneous sine and cosine
   fp32mp2 sin_result, cos_result;
   sincos(a, &sin_result, &cos_result);

   // Hyperbolic functions
   auto sinh_val = sinh(a);    // Hyperbolic sine
   auto cosh_val = cosh(a);    // Hyperbolic cosine
   auto tanh_val = tanh(a);    // Hyperbolic tangent

   // Error functions
   auto erf_val  = erf(a);      // Error function
   auto erfc_val = erfc(a);    // Complementary error function

   // Probability / Gaussian RNG
   auto inv_cdf = normcdfinv(a);          // Inverse normal CDF: Φ⁻¹(p)

Comparison Operations
~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   bool equal      = (a == b);
   bool less       = (a < b);
   bool greater    = (a > b);
   bool less_eq    = (a <= b);
   bool greater_eq = (a >= b);
   bool not_equal  = (a != b);

Classification Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Classification provides a portable prefixed API (``fpmp_is*``) and conditional
standard-name overloads. The standard overloads (``isfinite``, ``isinf``, ``isnan``,
``signbit``) are declared only when the corresponding macro is not defined by the
platform headers. This keeps standard call sites on common toolchains while
avoiding macro conflicts on platforms where these identifiers are macros.

.. code:: c++

   // Portable prefixed API (always available)
   bool finite = fpmp_isfinite(a);
   bool infv   = fpmp_isinf(a);
   bool nanv   = fpmp_isnan(a);
   bool sign   = fpmp_signbit(a);

   // Standard names are available only when not provided as macros
   bool finite_std = isfinite(a);
   bool infv_std   = isinf(a);
   bool nanv_std   = isnan(a);
   bool sign_std   = signbit(a);

Conversions
~~~~~~~~~~~

.. code:: c++

   // From standard types
   fp32mp2 from_double(3.14159265358979);
   fp32mp2 from_float(3.14159f);
   fp32mp2 from_int(42);

   // To standard types
   double to_d = static_cast<double>(a);
   float  to_f = static_cast<float>(a);  // Returns high component

   // IEEE-compliant bit-level access
   uint64_t bits = bit_cast<uint64_t>(a);

Volatile Objects
~~~~~~~~~~~~~~~~

An ``fpmp2`` object may be declared ``volatile``, which covers the legacy CUDA pattern
of holding shared-memory scalars in volatile variables. Volatile support is limited to
storage: loads, stores, copies between two volatile objects and reading ``hi()`` /
``lo()``. The round-trip is bit-preserving, and the types stay trivially copyable, as
required by ``cooperative_groups`` and ``__shfl``.

A volatile object cannot be an operand. Arithmetic, comparison and the math functions
take ``const fpmp2&``, and a volatile lvalue does not bind to it, so compute on a
non-volatile copy and store the result back. This mirrors what the compiler does for a
built-in ``volatile double``, where the load is volatile but the arithmetic is not.

.. code:: c++

   __global__ void volatile_kernel(const fp64mp2* in, fp64mp2* out) {
       __shared__ volatile fp64mp2 tile[64];
       tile[threadIdx.x] = in[threadIdx.x];    // store to volatile
       __syncthreads();

       fp64mp2 acc = tile[threadIdx.x];        // load into a non-volatile local
       acc = acc * acc + fp64mp2(1.0);         // compute there, not on the volatile
       tile[threadIdx.x] = acc;                // store the result back

       out[threadIdx.x] = fp64mp2(tile[threadIdx.x]);
   }

Atomic Operations
~~~~~~~~~~~~~~~~~

.. code:: c++

   __global__ void atomic_kernel(fp32mp2* shared_data) {
       fp32mp2 value = fp32mp2(1.0);   // double -> fp32mp2 is explicit by default
       atomicAdd(shared_data, value);  // Atomic addition
   }

Numeric Limits
~~~~~~~~~~~~~~

``cuda::std::numeric_limits<>`` is specialized for both double-word types (and all
accuracy variants), so generic code can query their characteristics exactly as it
does for ``float``/``double``. The specialization is pulled in by the
``<cuda/fpmp>`` umbrella — no extra include is required.

.. code:: c++

   #include <cuda/fpmp>
   namespace cs = cuda::std;

   constexpr int  d   = cs::numeric_limits<fp64mp2>::digits;    // 104
   constexpr auto eps = cs::numeric_limits<fp64mp2>::epsilon(); // 2^-103
   const     auto mx  = cs::numeric_limits<fp64mp2>::max();     // ~1.8e308
   const     auto inf = cs::numeric_limits<fp32mp2>::infinity();

Reported characteristics are derived from the underlying IEEE-754 component type. A
normalized, non-overlapping double-word carries ``2*p - 2`` contiguous mantissa
bits, so ``digits`` is ``46`` for ``fp32mp2`` and ``104`` for ``fp64mp2`` (matching
the mantissa bit counts in the specification). The exponent range follows the
double-double model: the maximum exponent matches the component type, while the
minimum normalized exponent is raised so that both halves stay normal. Because the
format is not IEEE-754, ``is_iec559`` is ``false``, but ``Inf``/``NaN`` and
round-to-nearest behavior are inherited from the component arithmetic.

.. list-table::
   :header-rows: 1

   * - Property
     - ``fp32mp2``
     - ``fp64mp2``
   * - ``digits`` / ``digits10`` / ``max_digits10``
     - 46 / 13 / 15
     - 104 / 31 / 33
   * - ``min_exponent`` / ``max_exponent``
     - -101 / 128
     - -968 / 1024
   * - ``epsilon()``
     - 2^-45
     - 2^-103
   * - ``min()`` / ``max()``
     - 2^-102 / ``FLT_MAX``
     - 2^-969 / ``DBL_MAX``
   * - ``is_iec559``
     - false
     - false

Usage Examples
--------------

Example 1: High-Precision Summation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   __global__ void precise_sum(const float* input, fp32mp2* output, int n) {
       __shared__ fp32mp2 partial_sums[256];
       
       int tid = threadIdx.x;
       int idx = blockIdx.x * blockDim.x + tid;
       
       // Initialize with high-precision conversion
       partial_sums[tid] = (idx < n) ? fp32mp2(input[idx]) : fp32mp2(0.0f);
       __syncthreads();
       
       // Parallel reduction with extended precision
       for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
           if (tid < stride) {
               partial_sums[tid] = partial_sums[tid] + partial_sums[tid + stride];
           }
           __syncthreads();
       }
       
       if (tid == 0) {
           output[blockIdx.x] = partial_sums[0];
       }
   }

Example 2: Double-Double High Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   #include <cuda/fpmp>

   using fp64mp2 = fpmp2<double, fpmp2_accuracy::def>;

   // Compute (1 + 1e-15)^2 - 1 - 2e-15 with quad-like precision
   // This should be exactly 1e-30, but double precision loses it
   fp64mp2 one = 1.0;
   fp64mp2 epsilon = 1e-15;
   fp64mp2 one_plus_eps = one + epsilon;
   fp64mp2 squared = one_plus_eps * one_plus_eps;
   fp64mp2 result = squared - one - epsilon - epsilon;
   // result accurately captures 1e-30 where double would fail

Example 3: Compensated Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   // Kahan summation with multi-precision
   fp32mp2 compensated_sum(const float* data, int n) {
       fp32mp2 sum(0.0);
       fp32mp2 compensation(0.0);
       
       for (int i = 0; i < n; i++) {
           fp32mp2 value(data[i]);
           fp32mp2 y = value - compensation;
           fp32mp2 temp = sum + y;
           compensation = (temp - sum) - y;
           sum = temp;
       }
       
       return sum;
   }

When to Use Multi-Precision
---------------------------

**Use fp32mp2 when:**

-  Your GPU has limited FP64 resources (e.g., consumer GPUs with 1:32 or 1:64 FP64:FP32 ratio)
-  You need double-precision accuracy but want better performance than native FP64
-  You want portable precision across CPU/GPU

**Use fp64mp2 when:**

-  You need quad-precision-like accuracy (~104-bit mantissa)
-  Native quad-precision is unavailable or slow
-  Hardware supports fast FP64 operations
-  Scientific computing requiring very high precision
-  Financial calculations requiring exact decimal representation

**Stick with native double when:**

-  FP64 units are abundant (e.g., HPC-focused GPUs with good FP64 performance)
-  Simplicity and standard compliance are paramount
-  You need hardware-level IEEE-754 guarantees
-  You require the full IEEE-754 specification (denormals, rounding modes, etc.)

Build Instructions
------------------

The core API is header-based and can be used directly by including headers.

Prerequisites
~~~~~~~~~~~~~

-  **CUDA Toolkit**: Version 11.0 or later (for GPU support)
-  **C++ compiler**: C++17 for the core library
-  **libquadmath** (optional): Required for double-double nath functions when ``_CCCL_FPMP_FP128_MATH_FALLBACK=1`` (provides ``expq``, ``sinq``, etc.)

Compilation Modes
~~~~~~~~~~~~~~~~~

The library supports two compilation modes for each component:

================ ======= =======================================================================================================================================
Macro            Default Description
================ ======= =======================================================================================================================================
``CCCL_FPMP_INLINE``  ``1``   Header-only inline mode for FPMP. When ``1`` (default), all implementations are compiled directly into the including translation unit.
``CCCL_FPMP_LIB``     ``0``   When ``1``, link against precompiled library for FPMP. Mutually exclusive with ``CCCL_FPMP_INLINE=1``.
``CCCL_FPEMU_INLINE`` ``1``   Header-only inline mode for FPEMU. When ``1`` (default), all implementations are compiled directly into the including translation unit.
``CCCL_FPEMU_LIB``    ``0``   When ``1``, link against precompiled library for FPEMU. Mutually exclusive with ``CCCL_FPEMU_INLINE=1``.
================ ======= =======================================================================================================================================

In the default inline mode (``CCCL_FPMP_INLINE=1``, ``CCCL_FPEMU_INLINE=1``) all implementations
are compiled directly into the including translation unit. Set ``CCCL_FPMP_LIB=1`` and/or
``CCCL_FPEMU_LIB=1`` when compiling against a prebuilt static library:

.. code:: bash

   # Header-only (default)
   g++ -std=c++17 -I/path/to/cccl/libcudacxx/include my_code.cpp

   # Library mode
   g++ -std=c++17 -DCCCL_FPMP_LIB=1 -DCCCL_FPEMU_LIB=1 -I/path/to/cccl/libcudacxx/include my_code.cpp -lcufp

Within CCCL the FP SDK is shipped header-only; the default inline mode requires no
separate build or link step. Library mode is documented for completeness but no
prebuilt library is currently provided in-tree.

Configuration Macros
~~~~~~~~~~~~~~~~~~~~

The library behavior can be customized via preprocessor macros (optional):

===================================== ======= ==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
Macro                                 Default Description
===================================== ======= ==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================
``CCCL_FPMP_EXPLICIT_CASTS``               ``1``   When ``1`` (default), lossy/narrowing conversions INTO ``fpmp2`` (``double``/``fp64mp2``/``__float128`` and ``int32``/``uint32``/``int64``/``uint64``) require explicit casts, matching CCCL's strict-cast conventions. The widening conversion OUT to ``double`` (``operator double()``) is always implicit and is not affected by this macro. Set ``0`` to restore the fully-implicit model (all conversions implicit). **Set ``0`` when** ``fpmp2`` is a near drop-in for ``double``/``float`` across a large codebase (existing call sites and mixed-type expressions compile unchanged instead of needing an explicit cast at every narrowing boundary) or for rapid prototyping where minimizing edit churn matters. **Warning:** with implicit casts the compiler silently narrows INTO ``fpmp2``, which can drop precision at unintended conversions or introduce accidental round-trips / FP64 use (accuracy/perf) with no diagnostic — keep the default ``1`` unless the migration benefit outweighs that risk. Note: explicit construction from ``double`` literals (e.g. ``fp32mp2(3.14159)``) remains ``constexpr`` (compile-time).
``_CCCL_FPMP_FP128_ENABLE``                 Auto    Automatically computed from the host toolchain's 128-bit floating-point support, and therefore identical in both passes of a CUDA compilation, so host code in a ``.cu`` file has the ``fp128`` interchange whatever the target architecture. Can be explicitly set to ``0`` to disable ``__float128`` support (e.g., for older compilers or compatibility).
``_CCCL_FPMP_FP128_DEVICE_OPS``             Auto    Automatically computed from ``__CUDA_ARCH_LIST__``: whether the ``fp128`` constructor and conversion are callable from device code, which requires every targeted architecture to be ``sm_100`` or later. When ``0`` they are host-only, and device code that reaches for quad precision is diagnosed at the call site. Set it to ``1`` explicitly on a toolchain that provides device ``fp128`` on earlier architectures.
``CCCL_FPMP_FP128_MATH_FALLBACK``           Auto    When ``1``, ``fp64mp2`` math functions compute in quad precision (``__float128``, ~113-bit) instead of ``double``. Requires ``libquadmath`` linkage on x86_64 GCC hosts, slower compilation, and larger code. When ``0``, every pass stays on ``double`` — faster builds, smaller code, but transcendental accuracy limited to ~53 bits. Left unset, the library decides per compilation pass, since this selects function bodies rather than declarations: a host-only build takes the quad path wherever ``fp128`` is available, while in a CUDA compilation only the device pass does, and only where every targeted architecture can run ``fp128``. That keeps a ``.cu`` file from silently acquiring a ``libquadmath`` dependency its host-only counterpart never had, at the price of the two halves differing in accuracy; see :ref:`below <libcudacxx-extended-api-fp-fpmp-quad-both-passes>` to put both on the quad path.
``CCCL_FPMP_LIB``                          ``0``   When ``1``, link against a precompiled FPMP library. Core arithmetic functions are declared as ``extern "C"`` symbols resolved at link time, reducing compile times and code duplication across translation units. Requires building the library separately (not provided in-tree; the CCCL FP SDK is header-only by default).
``CCCL_FPMP_INLINE``                       ``1``   When ``1`` (default), header-only inline mode. All functions are inlined directly into the calling code — no separate library build or link step required. Produces the fastest code (full inlining/optimization) at the cost of longer compile times in large projects. Mutually exclusive with ``CCCL_FPMP_LIB=1``.
``CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP``     ``1``   When ``1`` (default), the ``double`` to ``fpmp2`` conversion uses integer bit manipulation to split the double mantissa into two float components without FP64 arithmetic. This avoids the slow FP64 pipeline on GPUs with limited double-precision throughput (e.g., consumer GPUs with 1:64 ratio) and applies to the FP32-based ``fp32mp2`` (``fp64mp2`` conversions are inherently FP64 either way). When ``0``, uses the standard cast-based approach. **Set ``0`` if** you hit register pressure / reduced occupancy in large kernels (the integer path uses more registers and may spill), or you target a GPU with high FP64 throughput (e.g. datacenter A100/H100, ~1:2) where the FP64 path is already cheap. Profile your specific kernel to verify.
``CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE``     ``1``   When ``1`` (default), the ``fpmp2`` to ``double`` conversion reconstructs the double bit pattern from two float components using integer arithmetic (float-to-double bit promotion + software double-add) without FP64 operations. When ``0``, uses the standard ``(double)hi + (double)lo`` (2x F2D + DADD = 3 FP64 ops). The integer path is a large win on FP64-throttled GPUs (measured several-x faster than the FP64 casts on an L40S). **Set ``0`` if** you hit register pressure / reduced occupancy in large kernels, or you target a high-FP64 GPU where the FP64 path is already cheap. Profile your specific kernel to verify.
``_CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK`` ``0``   Controls fp32mp2 ``sin``/``cos``/``sincos``/``tan`` behavior for large arguments (\`
===================================== ======= ==================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================

.. _libcudacxx-extended-api-fp-fpmp-quad-both-passes:

Quad precision on both host and device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default a CUDA translation unit takes the binary128 path for ``fp64mp2`` math only in
its device pass, and only where every targeted architecture can run ``fp128`` (``sm_100``
and later); the host pass stays on ``double``. The two halves can therefore differ in
accuracy. Programs whose host and device results have to agree to the last bits put both on
the quad path with ``CCCL_FPMP_FP128_MATH_FALLBACK``:

.. code:: bash

   nvcc -arch=sm_100 -DCCCL_FPMP_FP128_MATH_FALLBACK=1 app.cu -lquadmath

``-lquadmath`` is what the host half needs on x86_64 GCC, where the quad entry points
(``expq``, ``sinq``, ...) live in that library. Hosts whose ``long double`` is IEEE
binary128 (AArch64, PPC64LE, s390x) call libm's ``*l`` entry points instead and need no
extra library, and a host-only build already takes the quad path wherever ``fp128`` is
available.

Two things to watch. Asking for the quad path on a target whose device cannot run ``fp128``
makes the device pass fail to compile, since its bodies then need quad arithmetic the
architecture does not have — such targets can only be opted in through the internal switch
documented in ``<cuda/__fp/fpmp_math_impl.h>``, and only with a toolchain that emits
``fp128`` for them. And every translation unit in the program must agree on the value, as
must the library build in library mode, since it selects which implementation the
``fp64mp2`` entry points get.

Header-Only Integration
~~~~~~~~~~~~~~~~~~~~~~~

1. **Direct inclusion**:

.. code:: c++

   #include <cuda/fpmp>       // Core: multi-precision types, operators, sqrt, rsqrt, fma
   #include <cuda/fpmp_math>  // Optional: transcendental math (exp, log, trig, ...)

2. **Compiler flags** (point the include path at the CCCL ``libcudacxx`` headers):

.. code:: bash

   # For CUDA files (.cu)
   nvcc -std=c++17 -I/path/to/cccl/libcudacxx/include your_code.cu

   # For host-only code (C++17)
   g++ -std=c++17 -I/path/to/cccl/libcudacxx/include your_code.cpp

Building the Examples and Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For internal validation the FP SDK ships lightweight, self-contained Makefiles
(no CMake/install step needed, since the SDK is header-only). They compile
directly against the in-tree CCCL headers:

-  Examples: ``examples/fp/Makefile``
-  Unit tests: ``libcudacxx/test/libcudacxx/cuda/fp/units/Makefile``

.. note::

   These local Makefiles are a draft for internal validation. The eventual
   CCCL CMake/CTest integration is a separate follow-up.

Makefile Parameters
^^^^^^^^^^^^^^^^^^^

=========== ================================================== ===========================
Parameter   Description                                        Default
=========== ================================================== ===========================
``TARGET``  Build backend: ``device`` (nvcc) or ``host`` (g++) ``device``
``EXAMPLE`` Example/test(s) to build (basename, no extension)  all ``*.cpp`` in the folder
``ARCH``    CUDA ``sm_xx`` for device builds                   auto-detect (else ``86``)
``OUT``     Output directory                                   ``_out``
``VERBOSE`` ``0`` (silent) or ``1``                            ``1``
=========== ================================================== ===========================

Build Targets
^^^^^^^^^^^^^

================= =================================
Target            Description
================= =================================
``all``/``build`` Build the selected examples/tests
``run``           Build and run (writes ``*.log``)
``rebuild``       Clean, then build
``rerun``         Clean, then build and run
``clean``         Remove the output directory
``help``          Print usage and available targets
================= =================================

Usage examples
^^^^^^^^^^^^^^

.. code:: bash

   # Build + run all examples on the GPU (device, nvcc)
   make -C examples/fp run

   # Build + run all examples on the CPU (host, g++)
   make -C examples/fp TARGET=host run

   # Build + run a single example on the host
   make -C examples/fp TARGET=host EXAMPLE=fp32mp2 run

   # Build + run the unit tests on the host
   make -C libcudacxx/test/libcudacxx/cuda/fp/units TARGET=host run

File Structure
--------------

Within CCCL the library is shipped as a public umbrella header plus a set of
internal implementation headers under ``cuda/__fp/`` (users include only the
umbrella):

::

   libcudacxx/include/cuda/
   ├── fpmp                       # Public umbrella header (core) — include via <cuda/fpmp>
   ├── fpmp_math                  # Public umbrella header (core + math) — include via <cuda/fpmp_math>
   └── __fp/                      # Internal implementation headers (do not include directly)
       ├── fpmp.h               # Types, C++ class fpmp2, operators/conversions, core ops
       ├── fpmp_common.h        # Platform/compiler macros, utilities, error-free transform building blocks
       ├── fpmp_impl.h          # Low-level C-style API (builtins, conversions, comparisons, atomics)
       ├── fpmp_limits.h        # cuda::std::numeric_limits<> specialization for fp32mp2/fp64mp2
       └── fpmp_math.h          # Math extensions (exp, log, pow, trig, ...) + placeholders

   examples/fp/                                # Example programs (fp32mp2.cpp, fp64mp2.cpp, *_math, *_thread, ...)
   libcudacxx/test/libcudacxx/cuda/fp/units/   # Unit tests
   docs/libcudacxx/fp/                         # Documentation (fpmp, fpmp_spec, fpemu, fp64_tool, devtools)

Header Descriptions
~~~~~~~~~~~~~~~~~~~

There are two public entry points. ``<cuda/fpmp>`` provides the core type and
operations; ``<cuda/fpmp_math>`` adds the transcendental math functions (and
also pulls in ``<cuda/fpmp>``). Translation units that do not need math
functions should include only ``<cuda/fpmp>`` to reduce compile time. Both
umbrellas pull in the internal ``cuda/__fp/`` headers below, which are
documented for reference only and should not be included directly.

============================= ===================================================================================================================================================================================================================================================================================================================================================================================================================
Header                        Description
============================= ===================================================================================================================================================================================================================================================================================================================================================================================================================
``<cuda/fpmp>``               Public umbrella header (core). Include this for the type and operations; it brings in ``cuda/__fp/fpmp.h`` and ``cuda/__fp/fpmp_limits.h``.
``<cuda/fpmp_math>``          Public umbrella header (core + math). Brings in ``cuda/__fp/fpmp.h`` and ``cuda/__fp/fpmp_math.h``.
``cuda/__fp/fpmp.h``        Types, C++ class ``fpmp2`` with operators/conversions, and core ops.
``cuda/__fp/fpmp_common.h`` Platform/compiler macros, utilities, and shared building blocks for error-free transforms.
``cuda/__fp/fpmp_impl.h``   Low-level C-style API (builtins for arithmetic, conversions, comparisons; CUDA atomics).
``cuda/__fp/fpmp_limits.h`` ``cuda::std::numeric_limits<>`` specialization for the ``fp32mp2``/``fp64mp2`` types (and their method variants); pulled in by ``<cuda/fpmp>``.
``cuda/__fp/fpmp_math.h``   Math extensions: dedicated fp32mp2 implementations for ``exp``, ``log``, ``pow``, ``sin``, ``cos``, ``tan``, ``sincos``, ``asin``, ``acos``, ``atan``, ``atan2``, ``tanh``, ``erf``, ``erfc``, ``normcdfinv``, ``icdf``, ``cbrt``, ``rcbrt``, ``floor``, ``ceil``, ``round``, ``trunc``, ``fabs``, ``fmin``, ``fmax``, ``min``, ``max``; remaining CUDA math functions are placeholder wrappers (higher precision).
============================= ===================================================================================================================================================================================================================================================================================================================================================================================================================

Examples
--------

The library includes example programs:

Examples (``examples/``)
~~~~~~~~~~~~~~~~~~~~~~~~

-  **fp32mp2.cpp**: Float-float (double-float) precision demo with accuracy comparison
-  **fp64mp2.cpp**: Double-double (quad-like) precision demonstration

Algorithm Details
-----------------

Error-Free Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

The library implements proven algorithms for accurate multi-component arithmetic:

Two-Sum (Knuth/Dekker)
^^^^^^^^^^^^^^^^^^^^^^

Computes exact sum with error term:

.. code:: c++

   // s = a + b (approximate sum)
   // e = error in s such that a + b = s + e exactly
   auto [s, e] = two_sum(a, b);

Two-Product (Dekker)
^^^^^^^^^^^^^^^^^^^^

Computes exact product with error term:

.. code:: c++

   // p = a * b (approximate product)
   // e = error in p such that a * b = p + e exactly
   auto [p, e] = two_product(a, b);

Fast Two-Sum
^^^^^^^^^^^^

Faster version when \|a\| ≥ \|b\|:

.. code:: c++

   auto [s, e] = fast_two_sum(a, b);  // Requires |a| >= |b|

Normalization
~~~~~~~~~~~~~

Multi-precision values maintain the invariant that components are non-overlapping:

-  For fp32mp2: ``|lo| < ulp(hi)`` (the low component is smaller than the unit in the last place of the high component)
-  For fp64mp2: Same invariant with double-precision components

This ensures consistent representation and optimal accuracy. The ``renormalize()`` function can be used to restore this property after fast arithmetic operations.

Dedicated fp32mp2 Math Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All dedicated implementations use **pure float-float arithmetic** (no double-precision operations), making them suitable for GPU architectures where fp64 throughput is limited.

exp(x)
^^^^^^

Argument reduction: ``x = n·ln2 + r``, ``|r| < ln2/2``. Computes ``exp(r)`` using a 14-term Taylor series in fp32mp2, then scales the result by ``2ⁿ`` via IEEE-754 exponent bit manipulation. Achieves ~10⁻¹⁰ – 10⁻¹¹ relative error.

log(x)
^^^^^^

Range reduction extracts ``x = m·2ᵉ`` with ``m ∈ [1, √2)``. Approximates ``log(m) = 2·atanh((m−1)/(m+1))`` via a degree-8 minimax polynomial, then reconstructs ``log(x) = log(m) + e·ln2`` using an fp32mp2 ``ln2`` constant. Handles denormals via pre-scaling. All arithmetic is in fp32mp2.

pow(a, b)
^^^^^^^^^

Canonical identity ``pow(a, b) = exp(b · log(|a|))``, with sign fixup for ``a < 0`` and integer ``b``. All three primitives (log, multiply, exp) are dedicated fp32mp2 — no fp64 operations in the main path. Structurally identical to libdevice's ``__nv_pow`` (same special-case order, same IEEE 754-2008 corner cases), but drops libdevice's hi/lo bookkeeping around the exp call: libdevice has to fma-track ``(t_hi, t_lo)`` because its scalar ``__nv_exp`` only takes a single ``double``; our ``__fpmp2_exp`` consumes the full fp32mp2 pair natively, so the correction is implicit and the main path is just three calls.

Special cases (priority order):

===== ============================ ======================================================================
Order Condition                    Result
===== ============================ ======================================================================
1     ``a == 1`` or ``b == 0``     ``1`` (IEEE 754-2008, overrides NaN)
2     ``a`` or ``b`` is NaN        NaN
3     ``a == 0``, ``b > 0``        ``±0`` (``-0`` iff ``a == -0`` and ``b`` is odd integer)
4     ``a == 0``, ``b < 0``        ``±Inf`` (same sign rule)
5     ``a < 0``, ``b`` non-integer NaN
6     ``|a| == Inf``               ``±Inf`` or ``±0`` per sign of ``b`` and integer parity
7     ``|b| == Inf``, ``|a| == 1`` ``1`` (IEEE: ``pow(-1, ±Inf) = 1``)
8     ``|b| == Inf`` otherwise     ``+Inf`` or ``+0`` per ``(|a|>1) == (b>0)``
9     otherwise                    ``exp(b · log(|a|))``, sign-flip if ``a < 0`` and ``b`` is odd integer
===== ============================ ======================================================================

Integer-``b`` detection uses ``b.lo == 0 && truncf(b.hi) == b.hi``; odd-integer parity is only checked for ``|b.hi| < 2²⁴`` (above that, every float-representable ``b.hi`` is automatically even). No exponent clamping is needed since ``|log(a)| ≤ ~88`` for any finite ``a > 0``, so the dedicated ``exp`` saturation paths cover all overflow/underflow cases automatically.

Precision model
'''''''''''''''

The accuracy of ``pow(a, b)`` depends on ``|b · log(a)|``, scaling roughly as:

::

   result_bits ≈ 46 − log₂(|b · log(a)|)

This is **inherent to the ``exp(b · log(a))`` algorithm** — libdevice's fp64 ``pow`` exhibits the same shape, just at a different baseline. Practical breakdown:

================ ======================================= =============
``|b · log(a)|`` Typical result range                    Expected bits
================ ======================================= =============
≤ 1              ``[1/e, e]`` ≈ ``[0.37, 2.7]``          46
≈ 8              ``[e⁻⁸, e⁸]`` ≈ ``[3·10⁻⁴, 3·10³]``     43
≈ 32             ``[e⁻³², e³²]`` ≈ ``[10⁻¹⁴, 8·10¹³]``   41
≈ 64             ``[e⁻⁶⁴, e⁶⁴]`` ≈ ``[2·10⁻²⁸, 6·10²⁷]`` 40
≈ 88             near fp32 saturation edges              39
================ ======================================= =============

For everyday use (gamma correction, ``pow(x, 2)``, probabilities raised to modest exponents) you get 43–46 effective bits — full fp32mp2 precision. Extreme exponentials approaching the fp32 representable range (``|result| ≈ 10³⁸`` or ``≈ 10⁻³⁸``) drop to ~39–40 bits, which is still substantially better than fp32's ~24 bits.

Inputs that produce results outside the fp32 representable range (overflow → ``+Inf``, underflow → ``±0``) are correctly saturated and flagged as ``WARN`` in the test suite (not ``FAIL``) because they fall into the "output is special / output is denormal" mitigating categories.

erf(x)
^^^^^^

Computes ``erf(x) = −expm1(−|x|·P(|x|))`` where ``P`` is a degree-24 Remez polynomial. The ``expm1`` is evaluated via argument reduction and a dedicated polynomial, all in fp32mp2.

erfc(x)
^^^^^^^

Computes ``erfc(x) = erfcx(|x|)·exp(−x²)``. The scaled complementary error function ``erfcx`` is approximated by a degree-22 Chebyshev polynomial in the transformed variable ``t = 1/(1+|x|)``, combined with an ``exp(−x²)`` evaluation using a degree-8 polynomial. Uses the identity ``erfc(−x) = 2 − erfc(x)`` for negative arguments. All arithmetic is in fp32mp2.


normcdfinv(p) — Inverse Normal CDF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Branchless rational approximation using Mike Giles coefficients. Computes ``w = −log(4p(1−p))`` via the fp32mp2 ``log``, then evaluates one of two Horner polynomials:

=============== ============= ============================================= ==================
Branch          Condition     Polynomial                                    Variable
=============== ============= ============================================= ==================
Central         ``w < 6.125`` degree-22                                     ``tc = w − 3.125``
Tail            ``w ≥ 6.125`` degree-18                                     ``tt = √w − 3.25``
Low-probability ``p < 2⁻²⁴``  degree-10 rational (erfcinvf tail polynomial) ``1/√(−log(2p))``
=============== ============= ============================================= ==================

Boundary clamping: returns ``±FLT_MAX`` for ``p ≤ 0`` or ``p ≥ 1`` (never ``±inf``), which is critical for safe Gaussian variate generation.


icdf(uint32_t x) / icdf(uint64_t x) — Integer Uniform to Gaussian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts an integer uniform RNG output to a Gaussian fp32mp2 sample. Follows the cuRAND mirroring convention:

-  **32-bit**: Mirrors ``x`` around ``2³¹`` to map into ``(0, 0.5]``. Computes ``p = (x + 0.5)/2³²`` as an exact fp32mp2 value by splitting ``x`` into two 16-bit halves, then calls ``normcdfinv(p)``.
-  **64-bit**: Keeps the top 48 bits (matching fp32mp2 mantissa capacity), mirrors around ``2⁴⁷``, computes ``p = (x + 0.5)/2⁴⁸`` by splitting into two 24-bit halves, then calls ``normcdfinv(p)``.

These functions enable high-precision Gaussian sampling directly from integer RNG state without going through double-precision intermediates.

sin(x), cos(x), sincos(x)
^^^^^^^^^^^^^^^^^^^^^^^^^

Argument reduction maps ``x = n·(π/2) + r`` with ``|r| ≤ π/4``, then evaluates Taylor-series kernels for sin(r) and cos(r) in fp32mp2. Three reduction paths based on ``|x|``:

===== ========= ======
Path  Condition Method
===== ========= ======
Tiny  \`        x
Fast  \`        x
Large \`        x
===== ========= ======

Core polynomials (evaluated in fp32mp2 Horner form):

-  **sin(r)**: 8-term Taylor (x through x¹⁵)
-  **cos(r)**: 9-term Taylor (1 through x¹⁶)

``sincos`` computes both kernels with quadrant mapping via ``n mod 4`` (sign/swap adjustment). ``sin`` and ``cos`` call ``sincos`` internally.

When ``_CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK=0`` (default), the large-argument path uses a dedicated pure-fp32mp2 Payne-Hanek reduction (no fp64 ops) that delivers ~46 bits in the reduced argument. Set to ``1`` to instead delegate the large-\|x\| branch to system fp64 ``sin``/``cos`` (smaller code, accuracy capped by fp64).

tan(x)
^^^^^^

Composed from the shared sin/cos pipeline (mirrors libdevice ``__nv_tan``, but without any FP64 ops on the main path):

1. Argument reduction ``x = n·(π/2) + r``, ``|r| ≤ π/4`` — same ``__internal_fpmp2_trig_reduction`` used by ``sin``/``cos``/``sincos``.
2. Evaluate ``sin(r)`` and ``cos(r)`` via the shared Taylor kernels.
3. ``tan`` has period π, so only the LSB of the quadrant ``n`` matters:

============ =============================================
Quadrant LSB Result
============ =============================================
``n`` even   ``tan(x) =  sin(r) / cos(r)``
``n`` odd    ``tan(x) = −cos(r) / sin(r)`` (``= −cot(r)``)
============ =============================================

The full quadrant-mod-4 sign dance from ``sincos`` is unnecessary here because ``tan(x + π) = tan(x)`` absorbs the ``n == 2, 3`` sign flips. Singularities at ``x ≡ π/2 (mod π)`` produce ``±∞`` through the ``n``-odd branch when ``sin(r)`` underflows to zero (IEEE / libdevice convention).

Like ``sin``/``cos``, the large-\|x\| path honors ``_CCCL_FPMP_LARGE_TRIG_FP64_FALLBACK``: when ``0`` (default) it uses the dedicated pure-fp32mp2 Payne-Hanek reduction; when set to ``1`` and ``|x_hi| ≥ 2²⁰``, the implementation delegates to system fp64 ``::tan``.

   **Note on accuracy near singularities.** For any input close to ``π/2 + kπ``, the output precision is fundamentally limited by ``tan'(x) = 1 + tan²(x)``, which amplifies the fp32mp2 quantization of the input into the result. At ``x ≈ ±10⁴`` and ``|tan(x)| ≈ 6·10⁵``, the input ulp (~7·10⁻¹²) is amplified by ``tan' ≈ 3·10¹¹`` to ~2.5 in the output, capping the relative precision around 17 bits — *regardless* of which reduction algorithm is used. This is intrinsic to the type, not a bug in either Cody-Waite or Payne-Hanek; both deliver ≳ 40 bits in the reduced argument ``r``.

tanh(x)
^^^^^^^

Two-branch sigmoid scheme structurally identical to the ``erf`` path (both are odd S-curves saturating at ±1):

========== ========= ======
Path       Condition Method
========== ========= ======
Saturation \`        x
Large \`   x         \`
Small \`   x         \`
========== ========= ======

Sign of ``x`` is preserved naturally by the polynomial branch and reintroduced via negation in the exp branch. All arithmetic is in fp32mp2.

asin(x), acos(x), atan(x), atan2(y, x)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dedicated inverse-trigonometric family, all four functions share two polynomial kernels evaluated in pure fp32mp2 (no fp64 arithmetic). Coefficients are the libdevice fp64 minimax fits whose truncation noise sits at fp64 ulp — well below the fp32mp2 target. Both kernels use full-precision Horner (``M = 0``) because even the smallest coefficient (~2·10⁻⁵ for atan, ~2.8·10⁻⁶ for acos) carries enough float-rounding noise to clip the result above the fp32mp2 ulp on the ``|a| ≤ 1`` boundary; mixed-precision Horner would require a finer reduction or refitted coefficients.

=============== ====================== ======
Function        Reduction              Method
=============== ====================== ======
``atan(x)``     \`                     x
``atan2(y, x)`` Octant analysis on \`( y
``asin(x)``     \`                     x
``asin(x)``     \`                     x
``acos(x)``     \`                     x
``acos(x)``     \`                     x
=============== ====================== ======

**Special-case handling.** ``atan2`` honors the IEEE-754 / C99 §F.10.1.4 cardinal-direction answers exactly (``0, ±π/2, ±π, ±π/4, ±3π/4``), including all ``±0`` and ``±∞`` sign combinations. Inputs outside ``[−1, 1]`` to ``asin``/``acos`` produce ``NaN`` via the square-root of a negative ``y``. ``NaN`` in any input component propagates.

**Denominator pre-scaling in ``atan2``.** The fp32mp2 division operator uses ``rcp(b_hi)`` which underflows to a denormal (or 0 under FTZ) when ``b_hi ≈ FLT_MAX``, and overflows to ∞ when ``b_hi`` is itself a denormal. Both cases collapse ``MAX/MAX`` to 0 (instead of 1) or ``MIN/MIN`` to NaN, on perfectly valid finite atan2 inputs. We sniff the denominator exponent and rescale both numerator and denominator by a fixed exact power of two (2⁻⁶⁴, 2⁺⁶⁴, or 2¹²⁷ for denormals) before the division. The ratio is preserved bit-for-bit.

   **Note on signed-zero semantics for ``atan2``.** The test-framework reference round-trips an fp32mp2 input through ``(double)(hi + lo)`` before evaluating the reference. Per IEEE addition of signed zeros, this sum is ``−0`` only when **both** components are ``−0``; every other ``±0`` combination (``(−0, +0)``, ``(+0, −0)``, ``(+0, +0)``) collapses to ``+0``. Our ``atan2`` mirrors that convention: when an argument's ``hi`` is zero, the "effective" sign of that argument is taken from the AND of ``signbit(hi)`` and ``signbit(lo)``. This means, for example, ``atan2((−0, +0), −1)`` returns ``+π`` (because the effective ``y`` is ``+0``), while ``atan2((−0, −0), −1)`` returns ``−π``. Application code that doesn't care about signed zeros sees the more intuitive result; code that does can construct a "true" multi-precision ``−0`` by setting both components negative.

API Reference
-------------

Construction and Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   // Constructors
   fp32mp2();                    // Default (zero)
   fp32mp2(double d);            // From double
   fp32mp2(float f);             // From float
   fp32mp2(int32_t i);           // From integer
   fp32mp2(float hi, float lo);  // Direct component initialization

   // Conversions
   explicit operator double() const;
   explicit operator float() const;

Arithmetic Operators
~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   fp32mp2 operator+(const fp32mp2& other) const;  // Addition
   fp32mp2 operator-(const fp32mp2& other) const;  // Subtraction
   fp32mp2 operator*(const fp32mp2& other) const;  // Multiplication
   fp32mp2 operator/(const fp32mp2& other) const;  // Division
   fp32mp2 operator-() const;                         // Negation

   // Compound assignment (multi-precision operand)
   fp32mp2& operator+=(const fp32mp2& other);
   fp32mp2& operator-=(const fp32mp2& other);
   fp32mp2& operator*=(const fp32mp2& other);
   fp32mp2& operator/=(const fp32mp2& other);

   // Optimized compound assignment (single-component operand)
   // Uses __fpmp2_acc which is more efficient than full mp2+mp2 addition
   // (saves ~6 operations by avoiding low-part 2Sum)
   fp32mp2& operator+=(float c);   // Accumulate single float
   fp32mp2& operator-=(float c);   // Subtract single float

   // Core math functions
   fp32mp2 sqrt(const fp32mp2& x);         // Square root
   fp32mp2 rsqrt(const fp32mp2& x);        // Reciprocal square root (1/sqrt(x))
   fp32mp2 fma(const fp32mp2& a,          // Fused multiply-add (a*b+c)
                 const fp32mp2& b, 
                 const fp32mp2& c);
   fp32mp2 mad(const fp32mp2& a,          // Multiply-add (a*b+c)
                 const fp32mp2& b, 
                 const fp32mp2& c);

Accuracy-Explicit Free Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override the arithmetic accuracy for a single operation. The template parameter ``m``
is one of ``fpmp2_accuracy::def``, ``fpmp2_accuracy::low``, or ``fpmp2_accuracy::high``.
The return type matches the input type — no type conversion occurs.

.. code:: c++

   // Available for any fpmp2<FpType, met> type
   template<fpmp2_accuracy m> T add(const T& x, const T& y);  // Addition
   template<fpmp2_accuracy m> T sub(const T& x, const T& y);  // Subtraction
   template<fpmp2_accuracy m> T mul(const T& x, const T& y);  // Multiplication
   template<fpmp2_accuracy m> T div(const T& x, const T& y);  // Division
   template<fpmp2_accuracy m> T fma(const T& x, const T& y, const T& z);  // Fused multiply-add
   template<fpmp2_accuracy m> T mad(const T& x, const T& y, const T& z);  // Multiply-add

Transcendental mathematical Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   // Roots and power functions (fpmp_math.h)
   fp32mp2 exp(const fp32mp2& x);          // Exponential (e^x)
   fp32mp2 log(const fp32mp2& x);          // Natural logarithm
   fp32mp2 log2(const fp32mp2& x);         // Base-2 logarithm
   fp32mp2 log10(const fp32mp2& x);        // Base-10 logarithm
   fp32mp2 log1p(const fp32mp2& x);        // log(1+x)
   fp32mp2 pow(const fp32mp2& x,           // Power (x^y)
                 const fp32mp2& y);
   fp32mp2 cbrt(const fp32mp2& x);         // Cube root

   // Trigonometric functions (fpmp_math.h)
   fp32mp2 sin(const fp32mp2& x);          // Sine
   fp32mp2 cos(const fp32mp2& x);          // Cosine
   void sincos(const fp32mp2& x,             // Simultaneous sine/cosine
               fp32mp2* s, fp32mp2* c);
   fp32mp2 asin(const fp32mp2& x);         // Arcsine
   fp32mp2 acos(const fp32mp2& x);         // Arccosine
   fp32mp2 atan(const fp32mp2& x);         // Arctangent
   fp32mp2 atan2(const fp32mp2& y,         // Two-argument arctangent
                   const fp32mp2& x);

   // Hyperbolic functions (fpmp_math.h)
   fp32mp2 sinh(const fp32mp2& x);         // Hyperbolic sine
   fp32mp2 cosh(const fp32mp2& x);         // Hyperbolic cosine
   fp32mp2 tanh(const fp32mp2& x);         // Hyperbolic tangent

   // Error functions (fpmp_math.h)
   fp32mp2 erf(const fp32mp2& x);          // Error function
   fp32mp2 erfc(const fp32mp2& x);         // Complementary error function

   // Probability / Gaussian RNG (fpmp_math.h)
   fp32mp2 normcdfinv(const fp32mp2& p);   // Inverse normal CDF: Φ⁻¹(p)

   // Rounding (fpmp_math.h)
   fp32mp2 floor(const fp32mp2& x);        // Dedicated fp32mp2 optimization
   fp32mp2 ceil(const fp32mp2& x);         // Dedicated fp32mp2 optimization
   fp32mp2 round(const fp32mp2& x);        // Dedicated fp32mp2 optimization
   fp32mp2 trunc(const fp32mp2& x);        // Dedicated fp32mp2 optimization

Comparison Operators
~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   bool operator==(const fp32mp2& other) const;
   bool operator!=(const fp32mp2& other) const;
   bool operator<(const fp32mp2& other) const;
   bool operator<=(const fp32mp2& other) const;
   bool operator>(const fp32mp2& other) const;
   bool operator>=(const fp32mp2& other) const;

Utility Functions
~~~~~~~~~~~~~~~~~

.. code:: c++

   fp32mp2 renormalize(const fp32mp2& x);  // Ensure non-overlapping components
   uint64_t bit_cast<uint64_t>(const fp32mp2& x);  // Bit-cast to IEEE-754 double format

   // Atomic operations (CUDA device only)
   fp32mp2 atomicAdd(fp32mp2* address, const fp32mp2& value);
   fp32mp2 atomicSub(fp32mp2* address, const fp32mp2& value);

References
----------

The library implements algorithms from these seminal papers:

1. | **Dekker, T. (1971)**
   | "A floating-point technique for extending the available precision"
   | *Numerische Mathematik*, 18, 224–242.
   | `DOI: 10.1007/BF01397083 <https://doi.org/10.1007/BF01397083>`__

2. | **Karp, A. H., & Markstein, P. (1997)**
   | "High Precision Division and Square Root"
   | *ACM Transactions on Mathematical Software*, 23(4), 561–589.
   | `DOI: 10.1145/279232.279237 <https://doi.org/10.1145/279232.279237>`__

3. | **Thall, Andrew**
   | "Extended-Precision Floating-Point Numbers for GPU Computation"
   | `Paper PDF <http://andrewthall.org/papers/df64_qf128.pdf>`__

4. | **Nagai et al. (2008)**
   | "Fast Quadruple Precision Arithmetic Library on Parallel Computer SR11000/J2"
   | *ICCS '08*

5. | **Ogita, T., Rump, S. M., & Oishi, S. (2005)**
   | "Accurate Sum and Dot Product"
   | *SIAM Journal on Scientific Computing*, 26(6), 1955–1988.
