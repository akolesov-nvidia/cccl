FP64 Precision Emulation Tool
=============================

A standalone utility for algorithm sensitivity analysis and mixed-precision research. The ``<cuda/fptool>`` header provides a drop-in replacement for ``double`` that applies precision reduction callbacks to all arithmetic operations.

Quick Start
-----------

The FP SDK lives in the ``cuda::experimental`` namespace (it will be promoted to
``cuda::`` later). The examples below assume ``using namespace cuda::experimental;``.

.. code:: cpp

   #define FP64_TOOL_MANTISSA_BITS 23  // Emulate float precision (52 → 23 bits)
   #include <cuda/fptool>

   using namespace cuda::experimental;

   fp64_tool a = 1.5, b = 2.5;
   fp64_tool result = a + b;  // Precision callbacks applied automatically
   double native = result;      // Convert back to native double

Configuration Macros
--------------------

Define these **before** including ``<cuda/fptool>``:

============================== ======= =================================================================================================
Macro                          Default Description
============================== ======= =================================================================================================
``FP64_TOOL_MANTISSA_BITS``    52      Number of mantissa bits to preserve (1-52)
``FP64_TOOL_EXPONENT_BITS``    11      Number of exponent bits to preserve (1-11)
``FP64_TOOL_IEEE_ROUNDING``    default Use IEEE 754 round-to-nearest-even
``FP64_TOOL_ROUND_TO_NEAREST`` -       Simple round-to-nearest (biased)
``FP64_TOOL_TRUNCATION``       -       Simple truncation (floor toward zero)
``FP64_TOOL_NO_UNDERFLOW``     -       Preserve underflowing values (no flush to zero)
``FP64_TOOL_NO_OVERFLOW``      -       Preserve overflowing values (no clamp to INF)
``FP64_TOOL_DISABLE``          -       Disable precision emulation entirely
``FP64_TOOL_RUNTIME_SIZE``     -       Enable runtime precision control (see `Runtime Precision Control <#runtime-precision-control>`__)
============================== ======= =================================================================================================

Common Precision Configurations
-------------------------------

====== ======== ======== =========================================================
Format Mantissa Exponent Configuration
====== ======== ======== =========================================================
FP64   52       11       Default (no callbacks)
FP32   23       8        ``FP64_TOOL_MANTISSA_BITS=23, FP64_TOOL_EXPONENT_BITS=8``
BF16   7        8        ``FP64_TOOL_MANTISSA_BITS=7, FP64_TOOL_EXPONENT_BITS=8``
FP16   10       5        ``FP64_TOOL_MANTISSA_BITS=10, FP64_TOOL_EXPONENT_BITS=5``
TF32   10       8        ``FP64_TOOL_MANTISSA_BITS=10, FP64_TOOL_EXPONENT_BITS=8``
====== ======== ======== =========================================================

How It Works
------------

Each arithmetic operation follows this pattern:

1. Apply callback to input operands (reduce precision)
2. Perform native FP64 operation
3. Apply callback to result (reduce precision)

This models how lower-precision hardware would handle the computation while maintaining full FP64 representation for intermediate storage.

Underflow/Overflow Control
--------------------------

When exponent bits are reduced (e.g., from 11 to 8 for FP32 emulation), values outside the new dynamic range will normally be clamped:

-  **Overflow**: Values too large for reduced exponent → Infinity (±INF)
-  **Underflow**: Values too small for reduced exponent → Zero (±0)

The ``FP64_TOOL_NO_OVERFLOW`` and ``FP64_TOOL_NO_UNDERFLOW`` macros change this behavior:

-  When defined, the original FP64 value is preserved instead of clamping
-  The value still goes through mantissa reduction
-  Useful for algorithms that need extended dynamic range while still emulating reduced mantissa precision

Example: Emulate FP32 mantissa with FP64 dynamic range:

.. code:: cpp

   #define FP64_TOOL_MANTISSA_BITS 23
   #define FP64_TOOL_NO_OVERFLOW
   #define FP64_TOOL_NO_UNDERFLOW
   #include <cuda/fptool>
   // Now: 23-bit mantissa precision, but full FP64 exponent range

Rounding Modes
--------------

Three rounding modes are available for mantissa reduction:

IEEE 754 Round-to-Nearest-Even (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default rounding mode produces statistically unbiased results:

-  If discarded bits > 0.5: round up
-  If discarded bits < 0.5: round down (truncate)
-  If discarded bits == 0.5: round to nearest even

Simple Round-to-Nearest
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp

   #define FP64_TOOL_ROUND_TO_NEAREST

Always rounds 0.5 up (biased).

Truncation
~~~~~~~~~~

.. code:: cpp

   #define FP64_TOOL_TRUNCATION

Simple truncation toward zero (floor for positive, ceiling for negative).

Runtime Precision Control
-------------------------

By default, mantissa and exponent sizes are fixed at compile time. Defining ``FP64_TOOL_RUNTIME_SIZE`` before including the header enables runtime adjustment of both parameters without recompilation.

Enabling Runtime Mode
~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp

   #define FP64_TOOL_RUNTIME_SIZE
   #define FP64_TOOL_MANTISSA_BITS 52  // Initial mantissa (used until changed)
   #define FP64_TOOL_EXPONENT_BITS 11  // Initial exponent (used until changed)
   #include <cuda/fptool>

The ``FP64_TOOL_MANTISSA_BITS`` and ``FP64_TOOL_EXPONENT_BITS`` macros set the *initial* values; they can be changed at any point during execution using the setter functions below.

Setter Functions
~~~~~~~~~~~~~~~~

=========================================== ====== ==================================
Function                                    Target Description
=========================================== ====== ==================================
``fp64_tool_set_host_mantissa_size(int)``   CPU    Set mantissa bits on host (1-52)
``fp64_tool_set_host_exponent_size(int)``   CPU    Set exponent bits on host (1-11)
``fp64_tool_set_device_mantissa_size(int)`` GPU    Set mantissa bits on device (1-52)
``fp64_tool_set_device_exponent_size(int)`` GPU    Set exponent bits on device (1-11)
=========================================== ====== ==================================

Host and device sizes are stored independently, so CPU and GPU code can run with different precision settings simultaneously.

CPU Example
~~~~~~~~~~~

.. code:: cpp

   #define FP64_TOOL_RUNTIME_SIZE
   #define FP64_TOOL_MANTISSA_BITS 52
   #define FP64_TOOL_EXPONENT_BITS 11
   #include <cuda/fptool>

   // Start with full precision
   fp64_tool a = 1.0, b = 1.0 / (1ULL << 52);
   double full = (double)(a - b);   // preserves 2^-52

   // Switch to reduced precision at runtime
   fp64_tool_set_host_mantissa_size(50);
   fp64_tool c = 1.0, d = 1.0 / (1ULL << 52);
   double reduced = (double)(c - d); // small value lost → 1.0

CUDA Example
~~~~~~~~~~~~

.. code:: cpp

   #define FP64_TOOL_RUNTIME_SIZE
   #define FP64_TOOL_MANTISSA_BITS 52
   #define FP64_TOOL_EXPONENT_BITS 11
   #include <cuda/fptool>

   __global__ void kernel(double a, double b, double* result) {
       fp64_tool x = a, y = b;
       *result = (double)(x + y);
   }

   // Host code:
   fp64_tool_set_device_mantissa_size(23);  // float-like precision on GPU
   kernel<<<1, 1>>>(a, b, d_result);
   cudaDeviceSynchronize();

   fp64_tool_set_device_exponent_size(8);   // also reduce exponent range
   kernel<<<1, 1>>>(a, b, d_result);
   cudaDeviceSynchronize();

The device setters use ``cudaMemcpyToSymbol`` when called from host code, so a ``cudaDeviceSynchronize()`` before the next kernel launch ensures the new value is visible on the GPU.

Notes
~~~~~

-  The ``host`` and ``device`` setters are independent — changing one does not affect the other.
-  On the device side, only thread 0 of block 0 writes the global variable to avoid race conditions.
-  All other features (rounding modes, underflow/overflow control, etc.) continue to work exactly the same in runtime mode.
-  There is a performance cost compared to compile-time mode because the bit counts are read from memory instead of being compiled as constants.

Use Cases
---------

-  **Algorithm sensitivity analysis**: Test how algorithms behave with reduced precision
-  **Mixed-precision research**: Emulate lower-precision formats before hardware implementation
-  **CUDA/CPU compatibility**: Works identically on both host and device code
-  **Precision debugging**: Identify precision-sensitive code paths

CUDA Support
------------

.. code:: cpp

   __global__ void kernel(fp64_tool* data, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           // Precision callbacks applied in device code too
           data[idx] = data[idx] * data[idx] + fp64_tool(1.0);
       }
   }

Supported Operations
--------------------

-  **Arithmetic**: ``+``, ``-``, ``*``, ``/``, unary ``-``
-  **Compound assignment**: ``+=``, ``-=``, ``*=``, ``/=``
-  **Increment/Decrement**: ``++``, ``--`` (pre and post)
-  **Comparison**: ``==``, ``!=``, ``<``, ``>``, ``<=``, ``>=``
-  **Math functions**: ``sqrt``, ``fma``
-  **Conversions**: to/from ``double``, ``float``, ``int32_t``, ``uint32_t``, ``int64_t``, ``uint64_t``

Type Alias
----------

.. code:: cpp

   using fp64_tool = fp64_tool_t;

Use ``fp64_tool`` as a drop-in replacement for ``double`` in your code.
