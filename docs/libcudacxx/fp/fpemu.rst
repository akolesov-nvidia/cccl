
FPEMU — Scalar Emulation Floating Point Library
===============================================

The FPEMU library provides emulated IEEE-754 double-precision floating-point
scalar arithmetic using integer computation units. It is designed for environments
where native FP64 hardware is limited, absent, or where trading precision for
performance is acceptable.

The library offers three accuracy levels that let the user balance accuracy
against speed:

============================ ======================== ======================================
Accuracy                     Mantissa precision       Special-value support
============================ ======================== ======================================
``fp64emu_accuracy::high``   Correctly rounded        Full IEEE-754 (INF, NaN, subnormals)
``fp64emu_accuracy::mid``    Up to 1-2 LSB error      Limited INF, NaN and subnormal support
``fp64emu_accuracy::low``    Up to half mantissa bits Limited INF, NaN and subnormal support
============================ ======================== ======================================

``fp64emu_accuracy::def`` is the default selector and equals ``high`` (IEEE-correct).

Rounding modes ``rn`` (nearest), ``rz`` (toward zero), ``ru`` (toward +∞) and
``rd`` (toward −∞) are supported for every accuracy level.

All FP SDK types live in the ``cuda::experimental`` namespace (it will be promoted
to ``cuda::`` later). The examples below assume ``using namespace cuda::experimental;``.

Table of Contents
-----------------

1. `C++ API <#c-api>`__
2. `Core Built-in Functions <#core-built-in-functions>`__
3. `Operations <#operations>`__
4. `Usage Examples <#usage-examples>`__

C++ API
-------

The primary interface is the C++ template class ``fp64emu_t<fp64emu_accuracy>``.
The ``fp64emu_accuracy`` template parameter selects the accuracy level at compile time,
so there is no runtime branching overhead.

Convenient type aliases are provided:

.. code:: c++

   using fp64emu      = fp64emu_t<fp64emu_accuracy::def>;   // default (== high)
   using fp64emu_low  = fp64emu_t<fp64emu_accuracy::low>;   // low accuracy
   using fp64emu_mid  = fp64emu_t<fp64emu_accuracy::mid>;   // mid accuracy
   using fp64emu_high = fp64emu_t<fp64emu_accuracy::high>;  // high accuracy

The class supports:

-  Implicit construction from ``float``, ``double``, and integer types
-  Implicit conversion back to ``double``
-  Standard arithmetic operators (``+``, ``-``, ``*``, ``/``) and compound assignments
-  CUDA-style built-in functions (``__dadd_rn``, ``__dmul_rn``, ``__fma_rn``, etc.)
   that automatically deduce the accuracy level from the argument type
-  Comparison operators (``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``)

Basic usage
~~~~~~~~~~~

.. code:: c++

   #include <cuda/fpemu>

   using namespace cuda::experimental;

   // Default accuracy (high, correctly rounded, full IEEE-754)
   fp64emu x = 1.0;
   fp64emu y = 2.0;
   fp64emu z = x + y;          // operator+
   fp64emu w = __fma_rn(x, y, z); // fused multiply-add

   // High accuracy (correctly rounded, full IEEE-754)
   fp64emu_t<fp64emu_accuracy::high> a = 1.0;
   auto b = __dadd_rn(a, a);     // accuracy deduced from argument type

   // Low accuracy (up to half mantissa bits error)
   fp64emu_t<fp64emu_accuracy::low> f = 1.0;
   auto g = f * f;

   // Convert back to double
   double result = (double)z;

Unpacked representation
~~~~~~~~~~~~~~~~~~~~~~~

An unpacked variant ``fp64emu_unpacked_t<fp64emu_accuracy>`` stores the sign,
exponent, and mantissa as separate fields. This avoids repeated pack/unpack
overhead in chains of operations:

.. code:: c++

   using fp64emu_unpacked = fp64emu_unpacked_t<fp64emu_accuracy::def>;

   fp64emu_unpacked a(1.0);
   fp64emu_unpacked b(2.0);
   fp64emu_unpacked c = a + b;  // stays unpacked through the chain
   double result = (double)c;     // pack + convert on final use

Core Built-in Functions
-----------------------

For direct control over the accuracy level, the ``libfpemu`` library provides
C-callable built-in functions operating on the raw ``fpbits64_t`` type
(a ``uint64_t`` holding the IEEE-754 bit pattern).

These built-in declarations are available through ``<cuda/fpemu>``.

Naming convention (packed ``fpbits64_t``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   __nv_fp64emu_<op>_<rm>            — default  (== high: correctly rounded, full IEEE-754 specials support)
   __nv_fp64emu_high_<op>_<rm>       — high     (correctly rounded, full IEEE-754 specials support)
   __nv_fp64emu_mid_<op>_<rm>        — mid      (1-2 LSB error, limited IEEE-754 specials support)
   __nv_fp64emu_low_<op>_<rm>        — low      (up to half mantissa bits, limited IEEE-754 specials support)

Naming convention (unpacked ``fpbits64_unpacked_t``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   __nv_fp64emu_unpacked_<op>_<rm>            — default  (== high: correctly rounded, full IEEE-754 specials support)
   __nv_fp64emu_unpacked_high_<op>_<rm>       — high     (correctly rounded, full IEEE-754 specials support)
   __nv_fp64emu_unpacked_mid_<op>_<rm>        — mid      (1-2 LSB error, limited IEEE-754 specials support)
   __nv_fp64emu_unpacked_low_<op>_<rm>        — low      (up to half mantissa bits, limited IEEE-754 specials support)

where ``<op>`` is the operation and ``<rm>`` is the rounding mode (``rn``, ``rz``,
``ru``, ``rd``).

Example
~~~~~~~

.. code:: c++

   #include <cuda/fpemu>

   fpbits64_t x = __nv_fp64emu_from_double(1.2345);
   fpbits64_t y = __nv_fp64emu_from_double(2.3456);

   // High-accuracy multiply (correctly rounded)
   fpbits64_t r1 = __nv_fp64emu_high_dmul_rn(x, y);

   // Low-accuracy add
   fpbits64_t r2 = __nv_fp64emu_low_dadd_rn(x, y);

   // Default FMA (== high)
   fpbits64_t r3 = __nv_fp64emu_fma_rn(x, y, r2);

   double result = __nv_fp64emu_to_double(r3);

Operations
----------

Arithmetic
~~~~~~~~~~

================== ============ ===============
Operation          C++ operator Built-in prefix
================== ============ ===============
Addition           ``+``        ``dadd``
Subtraction        ``-``        ``dsub``
Multiplication     ``*``        ``dmul``
Division           ``/``        ``ddiv``
Square root        ``sqrt()``   ``dsqrt``
Fused multiply-add ``fma()``    ``fma``
Multiply-add       ``mad()``    ``mad``
Dot product        ``dot()``    ``dot``
Complex multiply   ``cmul()``   ``cmul``
================== ============ ===============

Conversions
~~~~~~~~~~~

Implicit and explicit conversions between ``fp64emu_t`` and standard types:

=========================== ===================
Conversion                  Direction
=========================== ===================
``double`` ↔ ``fp64emu_t``  implicit both ways
``float`` → ``fp64emu_t``   implicit
``fp64emu_t`` → ``float``   explicit cast
``int32_t`` ↔ ``fp64emu_t`` implicit / explicit
``int64_t`` ↔ ``fp64emu_t`` explicit cast
=========================== ===================

Comparisons
~~~~~~~~~~~

All six relational operators are supported for both packed and unpacked types
and follow IEEE-754 comparison semantics.

Usage Examples
--------------

Setting a global accuracy level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   #include <cuda/fpemu>

   // Use the default accuracy everywhere
   using real_t = fp64emu;

   real_t a = 1.0, b = 2.0;
   real_t c = a * b + a;   // all operations use fp64emu_accuracy::def

Mixing accuracy levels in the same code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   #include <cuda/fpemu>

   // High-accuracy computation in a critical section
   fp64emu_t<fp64emu_accuracy::high> precise_a = input;
   auto precise_r = __fma_rn(precise_a, precise_a, precise_a);

   // Low-accuracy computation in a non-critical section
   fp64emu_t<fp64emu_accuracy::low> fast_a = input;
   auto fast_r = fast_a * fast_a;

Using core built-ins from library directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: c++

   #include <cuda/fpemu>

   fpbits64_t x = __nv_fp64emu_from_double(value);
   fpbits64_t y = __nv_fp64emu_from_double(value);

   // IEEE-754 compliant multiply
   fpbits64_t r = __nv_fp64emu_dmul_rn(x, y);

   // Fast variant of the same operation
   fpbits64_t r_fast = __nv_fp64emu_low_dmul_rn(x, y);

Compilation modes
~~~~~~~~~~~~~~~~~

The library supports two compilation modes controlled by user-facing macros:

================ ======= =============================================================================================================================
Macro            Default Description
================ ======= =============================================================================================================================
``FPEMU_INLINE`` ``1``   When ``1`` (default), header-only inline mode. All implementations are compiled directly into the including translation unit.
``FPEMU_LIB``    ``0``   When ``1``, link against precompiled ``libfpemu.a`` library. Mutually exclusive with ``FPEMU_INLINE=1``.
================ ======= =============================================================================================================================

In the default inline mode (``FPEMU_INLINE=1``) all implementations are compiled
directly into the including translation unit. Set ``FPEMU_LIB=1`` when compiling
against a prebuilt static library:

.. code:: bash

   # Header-only (default)
   g++ -std=c++17 -I/path/to/cccl/libcudacxx/include my_code.cpp

   # Library mode
   g++ -std=c++17 -DFPEMU_LIB=1 -I/path/to/cccl/libcudacxx/include my_code.cpp -lfpemu

Within CCCL the FP SDK is shipped header-only; the default inline mode requires no
separate build or link step. Library mode is documented for completeness but no
prebuilt library is currently provided in-tree.

Host and device portability
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All APIs work on both host (CPU) and device (GPU) code. When compiled with
``nvcc``, the appropriate ``__host__`` / ``__device__`` decorators are applied
automatically.

.. code:: c++

   #include <cuda/fpemu>

   __global__ void kernel(double* in, double* out) {
       fp64emu x = in[0];
       fp64emu y = in[1];
       out[0] = (double)(x * y);
   }
