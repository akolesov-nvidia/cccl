Development Tools Reference
===========================

This document describes the development tools available in the ``cuda_multi_fp`` package for testing, benchmarking, and release management.

--------------

Table of Contents
-----------------

-  `1. Test Suites (tsmp / tsemu) <#1-test-suites-tsmp--tsemu>`__
-  `2. Unit Tests <#2-unit-tests>`__
-  `3. Benchmarks <#3-benchmarks>`__
-  `4. Release Generator (make tar) <#4-release-generator-make-tar>`__
-  `5. Specification Generator (make spec) <#5-specification-generator-make-spec>`__

--------------


1. Test Suites (tsmp / tsemu)
-----------------------------

Two test suites are available under the ``ts/`` directory:

============== ============= ====================================
Target         Directory     Description
============== ============= ====================================
``make tsmp``  ``ts/fpmp/``  FPMP accuracy and performance tests
``make tsemu`` ``ts/fpemu/`` FPEMU accuracy and performance tests
============== ============= ====================================

FPMP Test Suite Location (``ts/fpmp/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   ts/fpmp/
   ├── Makefile          # Build system
   ├── ts.cpp            # Main test driver
   ├── ts_accuracy.hpp   # Accuracy measurement logic
   ├── ts_dataset.hpp    # Test data generation
   ├── ts_print.hpp      # Output formatting
   ├── ts_types.hpp      # Type definitions
   └── ts_functions.hpp  # Function wrappers

FPEMU Test Suite Location (``ts/fpemu/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   ts/fpemu/
   ├── Makefile          # Build system
   ├── ts.cpp            # Main test driver
   ├── ts_analyze.hpp    # Result analysis
   ├── ts_datasets.hpp   # Test data generation
   ├── ts_fixed.hpp      # Fixed-point test cases
   ├── ts_functions.hpp  # Function wrappers
   ├── ts_print.hpp      # Output formatting
   ├── ts_run.hpp        # Test execution logic
   ├── ts_types.hpp      # Type definitions
   └── ts_utils.hpp      # CUDA utilities

Parameters
~~~~~~~~~~

============================== =========================================================================================================================================================================================================================================== ========== ============================================================================
Parameter                      Values                                                                                                                                                                                                                                      Default    Description
============================== =========================================================================================================================================================================================================================================== ========== ============================================================================
``SET``                        ``def``, ``all``                                                                                                                                                                                                                            ``def``    Test set: default functions or all functions
``FUNC``                       ``fma``, ``mad``, ``add``, ``sub``, ``mul``, ``div``, ``sqrt``, ``rsqrt``, ``exp``, ``mp2int``, ``int2mp``, ``mp2uint``, ``uint2mp``, ``mp2ll``, ``ll2mp``, ``mp2ull``, ``ull2mp``, ``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``, ``all`` all        Functions to test
``TYPE``                       ``fp32mp2``, ``fp64mp2``, ``all``                                                                                                                                                                                                           all        Data types to test
``ACCURACY``                   ``def``, ``low``, ``high``, ``all``                                                                                                                                                                                                         all        Accuracy variants
``TARGET``                     ``device``, ``host``                                                                                                                                                                                                                        ``device`` Execution target
``ARCH``                       60, 70, 75, 80, 86, 89, 90, ...                                                                                                                                                                                                             86         CUDA compute capability
``LINKAGE``                    ``inline``, ``static``, ``lto``                                                                                                                                                                                                             ``inline`` Library linkage mode
``RIGOR``                      16–32                                                                                                                                                                                                                                       auto       Accuracy rigor level (log₂ of samples)
``RUN_ACCURACY``               ``y``, ``n``                                                                                                                                                                                                                                ``y``      Enable accuracy tests
``CLASSIFY``                   ``y``, ``n``                                                                                                                                                                                                                                ``y``      Enable accuracy classification
``TIMING``                     ``y``, ``n``                                                                                                                                                                                                                                ``y``      Enable timing/performance tests
``PRINT``                      ``ok``, ``fail``, ``warn``, ``all``, ``none``                                                                                                                                                                                               ``fail``   What results to print
``DATASET``                    ``work``, ``normal``, ``special``, ``pattern``                                                                                                                                                                                              all        Datasets to include in reports
``A1``, ``A2``, ``A3``, ``A4`` value                                                                                                                                                                                                                                       -          Fixed input arguments for debugging
``DEBUG``                      ``y``, ``n``                                                                                                                                                                                                                                ``n``      Enable debug mode
``CONSOLE``                    ``stdout``, ``stderr``, ``file``, ``null``                                                                                                                                                                                                  ``stdout`` Output stream
``OUT``                        path                                                                                                                                                                                                                                        ``_out``   Output directory
``VERBOSE`` / ``V``            0–3                                                                                                                                                                                                                                         0          Verbosity level
``FP2MP``                      ``1``                                                                                                                                                                                                                                       -          Enable optimized double→fpmp2 conversion (``FPMP_OPTIMIZED_DOUBLE_TO_FPMP``)
``MP2FP``                      ``1``                                                                                                                                                                                                                                       -          Enable optimized fpmp2→double conversion (``FPMP_OPTIMIZED_FPMP_TO_DOUBLE``)
``EXTRA_FLAGS``                string                                                                                                                                                                                                                                      -          Extra compiler flags
============================== =========================================================================================================================================================================================================================================== ========== ============================================================================

Targets
~~~~~~~

=========== ======================================
Target      Description
=========== ======================================
``run``     Build and run selected tests (default)
``rerun``   Clean, build, and run
``rebuild`` Clean and build without running
``build``   Build only
``clean``   Remove output directory
``help``    Show usage information
=========== ======================================

Usage Examples
~~~~~~~~~~~~~~

.. code:: bash

   # Run all default FPMP tests on GPU
   make tsmp

   # Run all default FPEMU tests on GPU
   make tsemu

   # Run specific FPMP function test
   make -C ts/fpmp FUNC=add TYPE=fp32mp2 ACCURACY=def rerun

   # Run specific FPEMU function test
   make -C ts/fpemu FUNC=fma ROUNDING=rn ACCURACY=high rerun

   # Run FPMP accuracy tests only (no timing)
   make -C ts/fpmp TIMING=n rerun

   # Run tests on host with high accuracy rigor
   make -C ts/fpmp TARGET=host RIGOR=28 rerun

   # Run fast subset for quick CI check
   make -C ts/fpmp SET=def TIMING=n CLASSIFY=n DATASET=work rerun

   # Debug specific input values
   make -C ts/fpmp FUNC=div A1=1.0:1e-8 A2=0.5:1e-9 DEBUG=y CONSOLE=null rerun

   # Generate detailed logs for all functions
   make tsmp OUT=_out_l40 VERBOSE=1

Output Files
~~~~~~~~~~~~

-  ``*.exe`` — Compiled test executables
-  ``*.log`` — Test results (stderr capture, includes accuracy tables)
-  ``YYYY-MM-DD.csv`` — Performance data in CSV format

--------------


2. Unit Tests
-------------

Unit tests (``units/``) verify correctness of individual API functions and components.

Location
~~~~~~~~

::

   units/
   ├── Makefile        # Build system
   ├── api.cpp         # API tests
   ├── atomic.cpp      # Atomic operation tests
   ├── convert.cpp     # Conversion tests
   └── ...             # Other unit test files


Parameters
~~~~~~~~~~

=================== =============================== ========== ============================
Parameter           Values                          Default    Description
=================== =============================== ========== ============================
``TARGET``          ``device``, ``host``            ``device`` Execution target
``UNIT``            test name(s)                    all        Specific unit test(s) to run
``LINKAGE``         ``inline``, ``static``, ``lto`` ``inline`` Library linkage mode
``ARCH``            60–90+                          86         CUDA compute capability
``OUT``             path                            ``_out``   Output directory
``VERBOSE`` / ``V`` 0–3                             0          Verbosity level
``EXTRA_FLAGS``     string                          -          Extra compiler flags
=================== =============================== ========== ============================


Targets
~~~~~~~

===================== ===============================
Target                Description
===================== ===============================
``build`` / ``units`` Build unit test executables
``run``               Build and run all tests
``rerun``             Clean, build, and run
``rebuild``           Clean and build without running
``logs``              Run tests and generate logs
``clean``             Remove output directory
``help``              Show usage information
===================== ===============================


Usage Examples
~~~~~~~~~~~~~~

.. code:: bash

   # Run all unit tests on GPU
   make units

   # Run specific unit test
   make -C units UNIT=api rerun

   # Run unit tests on host
   make -C units TARGET=host rerun

   # Run with LTO linkage
   make -C units LINKAGE=lto rerun

   # Run with verbose output
   make -C units VERBOSE=2 rerun


Output Files
~~~~~~~~~~~~

-  ``*.exe`` — Compiled unit test executables
-  ``*.log`` — Test output logs

--------------


3. Benchmarks
-------------

Benchmarks (``benchmarks/``) measure performance of specific use cases and algorithms.


Location
~~~~~~~~

::

   benchmarks/
   ├── Makefile      # Wrapper Makefile
   ├── bs/           # Black-Scholes analytical pricing
   │   ├── Makefile
   │   ├── bs.cpp
   │   └── README.md
   ├── fft/          # FFT benchmark
   │   └── Makefile
   ├── demo/         # Simple perf & accuracy estimation
   │   └── Makefile
   ├── gauss/        # Gaussian RNG performance
   │   ├── Makefile
   │   ├── gauss.cpp
   │   └── README.md
   ├── d2ff/         # Double-to-fp32mp2 conversion throughput
   │   ├── Makefile
   │   ├── d2ff.cpp
   │   └── README.md
   ├── mc/           # Monte Carlo European option pricing
   │   ├── Makefile
   │   ├── mc.cpp
   │   └── README.md
   └── ...           # Other benchmarks


Parameters
~~~~~~~~~~

=================== =============================== ========== ============================
Parameter           Values                          Default    Description
=================== =============================== ========== ============================
``TARGET``          ``device``, ``host``            ``device`` Execution target
``BENCHMARK``       subdirectory name(s)            all        Specific benchmark(s) to run
``LINKAGE``         ``inline``, ``static``, ``lto`` depends    Library linkage mode
``OUT``             path                            ``_out``   Output directory
``VERBOSE`` / ``V`` 0–3                             0          Verbosity level
=================== =============================== ========== ============================


Targets
~~~~~~~

========= ===============================
Target    Description
========= ===============================
``run``   Build and run benchmarks
``rerun`` Clean, build, and run (default)
``clean`` Remove output directory
``help``  Show available benchmarks
========= ===============================


Usage Examples
~~~~~~~~~~~~~~

.. code:: bash

   # Run all benchmarks
   make benchmarks

   # Run a specific benchmark
   make -C benchmarks BENCHMARK=demo rerun

   # Run benchmarks on host
   make -C benchmarks TARGET=host rerun

   # Run with custom output directory
   make benchmarks OUT=_benchmarks_l40

Double/fp32mp2 Conversion Benchmark (``benchmarks/d2ff/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures throughput of conversions between ``double`` and ``fp32mp2`` in both
directions, comparing standard (FP64-based) and optimized (integer-only) paths:

================ ================================= ==================== ============================== ==============
Direction        Flag                              Standard             Optimized                      FP64 ops saved
================ ================================= ==================== ============================== ==============
double → fp32mp2 ``FPMP_OPTIMIZED_DOUBLE_TO_FPMP`` FP64 cast + subtract Integer bit split              2
fp32mp2 → double ``FPMP_OPTIMIZED_FPMP_TO_DOUBLE`` 2x F2D + DADD        Integer promote + software add 3
================ ================================= ==================== ============================== ==============

All four kernel variants are compiled into the same binary via separate
compilation and compared in a single run. Each kernel performs a DAXPY-like
loop with ``DEPTH`` parallel conversion chains per thread, with a data-dependency
feedback loop to prevent dead code elimination.

Quick Start
^^^^^^^^^^^

.. code:: bash

   # GPU (default: DEPTH=16)
   make -C benchmarks/d2ff rerun

   # CPU
   make -C benchmarks/d2ff T=host rerun

   # Show kernel register usage, spills, and compile times
   make -C benchmarks/d2ff info

   # High register pressure
   make -C benchmarks/d2ff DEPTH=64 rerun


Parameters
^^^^^^^^^^

============== =============================== =========================================================
Parameter      Default                         Description
============== =============================== =========================================================
``TARGET``     ``device``                      ``device`` (GPU) or ``host`` (CPU)
``ARCH``       ``86``                          CUDA SM architecture
``DEPTH``      ``16``                          Parallel conversion chains per thread (register pressure)
``REPS``       ``4096`` (GPU) / ``1024`` (CPU) Loop iterations per chain
``UNROLL``     ``8``                           Loop unroll factor
``NUM_ITER``   ``10``                          Timing iterations for averaging
``THREADS``    ``256``                         CUDA threads per block
``NUM_BLOCKS`` ``2048``                        CUDA grid blocks
``LINKAGE``    ``inline``                      ``inline``, ``static``, or ``lto``
``VERBOSE``    ``1``                           Verbosity level (``0``..\ ``3``)
============== =============================== =========================================================

Example Output
^^^^^^^^^^^^^^

::

   Double/fp32mp2 Conversion Benchmark
   ===================================
   Target: GPU (NVIDIA L40)
   Threads: 524288  REPS: 4096  DEPTH: 16  UNROLL: 8  Timing iterations: 10
   Conversions per iteration: 34359738368

   double -> fp32mp2:
     Conversion                              Time (ms)    Gconv/s  Speedup
     -------------------------------------- ---------- ---------- --------
     Standard (FP64 ops)                       xxx.xxx      xxx.x      ref
     Optimized (INT32 bit ops, no FP64)         xx.xxx      xxx.x    x.xxx
     Sanity: std=xxxx.xxxxxxxxxx  opt=xxxx.xxxxxxxxxx  OK (exact)

   fp32mp2 -> double:
     Conversion                              Time (ms)    Gconv/s  Speedup
     -------------------------------------- ---------- ---------- --------
     Standard (FP64 ops)                       xxx.xxx      xxx.x      ref
     Optimized (INT32 bit ops, no FP64)         xx.xxx      xxx.x    x.xxx
     Sanity: std=xxxx.xxxxxxxxxx  opt=xxxx.xxxxxxxxxx  OK (exact)

Black-Scholes Benchmark (``benchmarks/bs/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prices N random European options using the analytical Black-Scholes formula
and compares throughput and accuracy across six floating-point types:

==================== ============================================ =========
Type                 Description                                  Mantissa
==================== ============================================ =========
``fp32mp2_low``     Double-float, low accuracy                   ~46 bits
``fp32mp2``          Double-float, default accuracy (mid)         ~46 bits
``double``           IEEE-754 double precision                    53 bits
``fp64emu``          Emulated fp64 via fp32 arithmetic (default)  53 bits
``fp64emu_high`` Emulated fp64 via fp32 arithmetic (high)     53 bits
``fp64mp2``          Double-double (pair of doubles)              ~104 bits
==================== ============================================ =========

Each option has randomized parameters (S, K, T, sigma). The benchmark is
fully templated so timing differences reflect arithmetic cost alone.
Results are validated against a double-precision reference.


Quick Start
^^^^^^^^^^^

.. code:: bash

   # GPU (default: 1M options × 128 reps)
   make -C benchmarks/bs rerun

   # CPU
   make -C benchmarks/bs T=host rerun


Parameters
^^^^^^^^^^

=============== ============================ ===================================
Parameter       Default                      Description
=============== ============================ ===================================
``TARGET``      ``device``                   ``device`` (GPU) or ``host`` (CPU)
``ARCH``        ``86``                       CUDA SM architecture
``NUM_OPTIONS`` ``1M`` (GPU) / ``32K`` (CPU) Options per launch
``REPS``        ``128`` (GPU) / ``8`` (CPU)  Inner repetitions per kernel launch
``NUM_ITER``    ``10``                       Timing iterations for averaging
``THREADS``     ``256``                      CUDA threads per block
``NUM_BLOCKS``  ``2048``                     CUDA grid blocks
``LINKAGE``     ``inline``                   ``inline``, ``static``, or ``lto``
``VERBOSE``     ``1``                        Verbosity level (``0``..\ ``3``)
=============== ============================ ===================================


Example Output
^^^^^^^^^^^^^^

::

   Black-Scholes Option Pricing Benchmark
   ======================================
   Target: GPU (NVIDIA L40)

   Options: 1048576  REPS: 128  r=0.0200  q=0.0100  Call
   Threads: 524288  Timing iterations: 10

   Reference mean price (double): xx.xxxxxx

   Accuracy vs double reference:

   Type                        Mean Price   Max|RelErr|   RMS|RelErr|
   ------------------------  ------------  ------------  ------------
   fp32mp2_low               xx.xxxxxx    x.xxxxe-xx    x.xxxxe-xx
   fp32mp2                    xx.xxxxxx    x.xxxxe-xx    x.xxxxe-xx
   double                       xx.xxxxxx    x.xxxxe-xx    x.xxxxe-xx
   fp64emu                    xx.xxxxxx    x.xxxxe-xx    x.xxxxe-xx
   fp64emu_high           xx.xxxxxx    x.xxxxe-xx    x.xxxxe-xx
   fp64mp2                    xx.xxxxxx    x.xxxxe-xx    x.xxxxe-xx

   Performance:

   Type                        Time(ms)         Options/sec
   ------------------------  ----------  ------------------
   fp32mp2_low                xx.xxx   x'xxx'xxx'xxx'xxx
   fp32mp2                     xx.xxx   x'xxx'xxx'xxx'xxx
   double                        xx.xxx   x'xxx'xxx'xxx'xxx
   fp64emu                     xx.xxx   x'xxx'xxx'xxx'xxx
   fp64emu_high            xx.xxx   x'xxx'xxx'xxx'xxx
   fp64mp2                    xxx.xxx   x'xxx'xxx'xxx'xxx

   Speedups vs double:
     ------------------------  ------
     fp32mp2_low             x.xxx
     fp32mp2                  x.xxx
     fp64emu                  x.xxx
     fp64emu_high         x.xxx
     fp64mp2                  x.xxx

   Result: COMPLETED

Monte Carlo Benchmark (``benchmarks/mc/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prices a European option via Monte Carlo simulation and compares accuracy
and throughput across multiple floating-point data types:

==================== ============================================ =========
Type                 Description                                  Mantissa
==================== ============================================ =========
``float``            IEEE-754 single precision                    24 bits
``fp32mp2``          Double-float (pair of floats)                ~46 bits
``double``           IEEE-754 double precision                    53 bits
``fp64emu``          Emulated fp64 via fp32 arithmetic (default)  53 bits
``fp64emu_high`` Emulated fp64 via fp32 arithmetic (accurate) 53 bits
``fp64emu_unpacked`` Emulated fp64 via fp32 arithmetic (unpacked) 53 bits
``fp64mp2``          Double-double (pair of doubles)              ~104 bits
==================== ============================================ =========

``fp64emu_unpacked`` is optional and controlled by ``MC_FPEMU_UNPACKED`` (default: off).

Every type goes through the same templated code path (GBM asset evolution,
payoff evaluation, partial-sum reduction) so that timing differences reflect
arithmetic cost alone. Results are validated against the Black-Scholes
closed-form price.


Quick Start
^^^^^^^^^^^

.. code:: bash

   # GPU (default: 16M paths × 128 reps, antithetic variates)
   make -C benchmarks/mc rerun

   # CPU
   make -C benchmarks/mc T=host rerun

   # With fpmp normcdfinv for inline RNG path
   make -C benchmarks/mc ICDF=2 rerun

Both pre-generated and inline RNG modes run in every execution, so a single
``make rerun`` produces results for both.


Parameters
^^^^^^^^^^

============== ============================= ====================================================================================================
Parameter      Default                       Description
============== ============================= ====================================================================================================
``TARGET``     ``device``                    ``device`` (GPU) or ``host`` (CPU)
``ARCH``       ``86``                        CUDA SM architecture
``NUM_PATHS``  ``16M`` (GPU) / ``32K`` (CPU) Total Monte Carlo paths
``REPS``       ``128`` (GPU) / ``16`` (CPU)  Inner repetitions per kernel launch
``NUM_ITER``   ``10``                        Timing iterations for averaging
``THREADS``    ``256``                       CUDA threads per block
``NUM_BLOCKS`` ``2048``                      CUDA grid blocks
``ICDF``       ``0``                         Gaussian method for inline RNG: ``0``\ =Box-Muller, ``1``\ =CUDA normcdfinv, ``2``\ =fpmp normcdfinv
``LINKAGE``    ``inline``                    ``inline``, ``static``, or ``lto``
``VERBOSE``    ``1``                         Verbosity level (``0``..\ ``3``)
============== ============================= ====================================================================================================

The ``REPS`` parameter is the main knob for measurement stability: higher values
produce longer kernels and more stable timing without additional memory.

Gaussian Variate Generation (``ICDF``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ICDF`` parameter selects how Gaussian variates are generated in the
inline RNG path:

======== =============== ============================================
``ICDF`` Method          Description
======== =============== ============================================
``0``    Box-Muller      cuRAND built-in (``curand_normal``)
``1``    CUDA normcdfinv ``normcdfinv(curand_uniform)``
``2``    FPMP icdf       ``icdf(curand_integer)`` — fp32mp2 path only
======== =============== ============================================

``ICDF=2`` uses the FPMP ``icdf`` function which converts a raw integer uniform
RNG output directly to a Gaussian fp32mp2 sample via ``normcdfinv``, avoiding
double-precision intermediates entirely.


Example Output
^^^^^^^^^^^^^^

::

   Monte Carlo European Option Pricing Benchmark
   ==============================================
   Target: GPU (NVIDIA L40)

   Option: S0=100.00 K=100.00 r=0.0500 q=0.0000 sigma=0.2000 T=1.0000 Call
   Paths: 16777216  Antithetic: yes  REPS: 128
   RNG (pregen): std::mt19937_64 normal distribution
   RNG (inline): cuRAND Philox4x32 Box-Muller
   Threads: 524288  Timing iterations: 10

   Black-Scholes analytical: 10.4506176247

   Each type is benchmarked twice:
     (pregen)        - pre-generated normals, measures arithmetic cost only
     (inline)        - RNG inside kernel, measures arithmetic + RNG cost
     (inline + icdf) - fpmp icdf(uint64) instead of Box-Muller (GPU only)
                       uses 64-bit uniform input; double icdf uses fp32mp2 precision

   Type                          MC Price    Time(ms)          Paths/sec  |Err vs BS|  Check
   ----------------------------  --------------  ----------  ------------------  ------------  -----
   float (pregen)                10.45xxxxxxxx       x.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   float (inline)                10.45xxxxxxxx       x.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   fp32mp2 (pregen)            10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   fp32mp2 (inline)            10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   fp32mp2 (inline + icdf)     10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   double (pregen)               10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   double (inline)               10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   double (inline + icdf)        10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   fp64emu (pregen)            10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   fp64emu (inline)            10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   fp64emu_high (pregen)   10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   fp64emu_high (inline)   10.45xxxxxxxx      xx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   fp64mp2 (pregen)            10.45xxxxxxxx     xxx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK
   fp64mp2 (inline)            10.45xxxxxxxx     xxx.xxx   x'xxx'xxx'xxx'xxx    x.xxe-xx  OK

   Speedups vs double:          pregen   inline
     -----------------------    ------   ------
     float              :      xx.xxx   xx.xxx
     fp32mp2          :       x.xxx    x.xxx
     fp64emu          :       x.xxx    x.xxx
     fp64emu_high :       x.xxx    x.xxx
     fp64mp2          :       0.xxx    0.xxx
     fp32mp2 (+icdf)  :           -    x.xxx
     double    (+icdf)  :           -    x.xxx

   Result: ALL PASSED (tolerance pregen=x.xxe-xx, inline=x.xxe-xx)

Gaussian RNG Benchmark (``benchmarks/gauss/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures throughput and statistical correctness of double-precision Gaussian
random number generation on GPU, comparing three methods:

======================== ======================================================= ==========================================
Method                   Implementation                                          Notes
======================== ======================================================= ==========================================
``curand_normal_double`` ``curand_normal_double(&state)``                        cuRAND built-in Box-Muller
``uniform + normcdfinv`` ``normcdfinv(curand_uniform_double(&state))``           CUDA math intrinsic
``fpmp icdf(uint64)``    Two ``curand()`` → ``uint64`` → ``icdf()`` → ``double`` fpmp fp32mp2 polynomial (~46-bit mantissa)
======================== ======================================================= ==========================================

All methods use the same cuRAND Philox state so timing differences reflect
the Gaussian transform cost alone. Per-thread sum and sum-of-squares are
accumulated to compute mean and stddev for statistical validation without
storing individual samples.


Quick Start
^^^^^^^^^^^

.. code:: bash

   make -C benchmarks/gauss rerun

   # Higher rigor
   make -C benchmarks/gauss REPS=256 rerun


Parameters
^^^^^^^^^^

=============== ========== ===================================
Parameter       Default    Description
=============== ========== ===================================
``ARCH``        ``86``     CUDA SM architecture
``NUM_SAMPLES`` ``16M``    Samples per launch
``REPS``        ``128``    Inner repetitions per kernel launch
``NUM_ITER``    ``10``     Timing iterations for averaging
``THREADS``     ``256``    CUDA threads per block
``NUM_BLOCKS``  ``2048``   CUDA grid blocks
``LINKAGE``     ``inline`` ``inline``, ``static``, or ``lto``
``VERBOSE``     ``1``      Verbosity level (``0``..\ ``3``)
=============== ========== ===================================


Example Output
^^^^^^^^^^^^^^

::

   Double-Precision Gaussian RNG Performance Benchmark
   ===================================================
   Target: GPU (NVIDIA GeForce RTX 3090)

   Samples: 16777216  REPS: 128
   Threads: 524288  Timing iterations: 10

   Effective samples per launch: 2147483648

   Methods:
     curand_normal_double  - cuRAND built-in Box-Muller
     uniform + normcdfinv  - cuRAND uniform + CUDA normcdfinv
     fpmp icdf(uint64)     - two curand() -> uint64 -> fpmp ICDF -> double
                             (fp32mp2 precision, ~46-bit mantissa)

   Running benchmarks...

   Method                    Time(ms)          Samples/sec           Mean        StdDev  Check
   ------------------------  ----------  ------------------  -------------  ------------  -----
   curand_normal_double         xxx.xxx   xx'xxx'xxx'xxx'xxx   x.xxxxxxe-xx  x.xxxxxxxxxx  OK
   uniform + normcdfinv         xxx.xxx   xx'xxx'xxx'xxx'xxx   x.xxxxxxe-xx  x.xxxxxxxxxx  OK
   fpmp icdf(uint64)             xx.xxx   xx'xxx'xxx'xxx'xxx   x.xxxxxxe-xx  x.xxxxxxxxxx  OK

   Speedups vs curand_normal_double:
     ------------------------  ------
     uniform + normcdfinv       x.xxx
     fpmp icdf(uint64)          x.xxx

   Result: ALL PASSED (tolerance: mean<x.xxe-xx, |stddev-1|<x.xxe-xx)

--------------


4. Release Generator (make tar)
-------------------------------

The release generator creates distribution tarballs with license headers injected into source files.

Script Location
~~~~~~~~~~~~~~~

::

   scripts/make_release.py


Parameters
~~~~~~~~~~

=================== ======== ========================================================================================== =========================================
Parameter           Values   Default                                                                                    Description
=================== ======== ========================================================================================== =========================================
``RELEASE_TYPE``    ``nda``  ``nda``                                                                                    Release type (determines license/headers)
``RELEASE_NAME``    filename auto                                                                                       Custom tarball name
``RELEASE_INCLUDE`` paths    ``include examples docs/README_fpmp.md docs/README_fpmp_spec.md docs/README_fp64_tool.md`` Directories/files to include
``RELEASE_EXCLUDE`` patterns ``./Makefile examples/fp64_tool.cpp include/fp64_tool.h``                                Patterns to exclude
``OUT``             path     ``_out``                                                                                   Output directory for tarball
=================== ======== ========================================================================================== =========================================

Default Exclusions
~~~~~~~~~~~~~~~~~~

The script automatically excludes:

-  ``_out/``, ``_out/*`` — Build output directories
-  ``*.o``, ``*.a``, ``*.so``, ``*.exe``, ``*.log`` — Build artifacts
-  ``.git/``, ``.gitignore`` — Version control
-  ``__pycache__/``, ``*.pyc`` — Python cache
-  ``.vscode/``, ``.idea/`` — IDE settings
-  ``*.swp``, ``*.swo``, ``*~`` — Editor temp files
-  ``.DS_Store`` — macOS metadata
-  ``scripts/`` — Development scripts

License Injection
~~~~~~~~~~~~~~~~~

The script injects appropriate license headers based on release type:

========================================================================== =================================
File Type                                                                  Header File
========================================================================== =================================
C/C++/CUDA sources (``.c``, ``.cpp``, ``.cu``, ``.h``, ``.hpp``, ``.cuh``) ``scripts/header_src_{type}.h``
Makefiles                                                                  ``scripts/header_make_{type}.mi``
Root LICENSE.txt                                                           ``scripts/license_{type}.txt``
========================================================================== =================================


Usage Examples
~~~~~~~~~~~~~~

.. code:: bash

   # Create default NDA release
   make tar

   # Create release with custom name
   make tar RELEASE_NAME=fpmp_v1.0.0_nda.tar.gz

   # Create release with specific includes
   make tar RELEASE_INCLUDE="include examples docs/README_fpmp.md"

   # Create release excluding additional files
   make tar RELEASE_EXCLUDE="examples/experimental.cpp"

   # Verbose output showing each processed file
   make tar VERBOSE=1

Direct Script Usage
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # Basic usage
   python3 scripts/make_release.py --release-type nda --output-dir _out

   # With specific includes
   python3 scripts/make_release.py --include include src examples --output-dir _out

   # With exclusions
   python3 scripts/make_release.py --exclude "test*" "*.tmp" --output-dir _out

   # Verbose with custom name
   python3 scripts/make_release.py --name my_release.tar.gz --verbose

Output
~~~~~~

-  ``cuda_multi_fp_{type}_{date}.tar.gz`` — Release tarball containing:

   -  ``cuda_multi_fp/`` — Root directory

      -  ``LICENSE.txt`` — License file
      -  ``include/`` — Header files with injected license
      -  ``examples/`` — Example files with injected license
      -  ``docs/README_fpmp.md`` — Main documentation
      -  ``docs/README_fpmp_spec.md`` — Library specification
      -  ``docs/README_fp64_tool.md`` — FP64 Tool documentation

--------------


5. Specification Generator (make spec)
--------------------------------------

The specification generator creates a comprehensive ``README_fpmp_spec.md`` document from
fpmp test set results (``ts/fpmp/`` only — fpemu data is intentionally not consumed).


Script Location
~~~~~~~~~~~~~~~

::

   scripts/make_spec.py


Parameters
~~~~~~~~~~

=============== ====== ============================== =========================================
Parameter       Values Default                        Description
=============== ====== ============================== =========================================
``ACC``         path   ``$(OUT)``                     Directory with accuracy test results
``PERF1``       path   -                              First GPU performance results (optional)
``PERF2``       path   -                              Second GPU performance results (optional)
``PERF3``       path   -                              Third GPU performance results (optional)
``SPEC_OUTPUT`` path   ``README_fpmp_spec.md`` in ACC Output file path
=============== ====== ============================== =========================================

Input Files
~~~~~~~~~~~

The script parses files from the ``ts/fpmp/`` subdirectory of each input folder:

============ ===========================================================
File Pattern Content
============ ===========================================================
``*.log``    Accuracy logs with classification tables and special values
``*.csv``    Performance data (GFLOPS, ev/clk/SM, clk/ev)
============ ===========================================================

Generated Content
~~~~~~~~~~~~~~~~~

The output ``README_fpmp_spec.md`` includes:

1. **Table of Contents** — Quick navigation to all functions
2. **Supported Data Types** — Description of ``fp32mp2`` and ``fp64mp2``
3. **Function Categories** — Organized sections:

   -  Arithmetic Operations (add, sub, mul, div, sqrt, etc.)
   -  Math Functions (exp, etc.)
   -  Comparison Operations (eq, ne, lt, le, gt, ge)
   -  Conversions (mp2int, int2mp, etc.)

4. **Per-Function Specifications**:

   -  Accuracy summary tables (per method/type)
   -  Special values tables
   -  Performance tables with multiple GPUs
   -  Performance ratios vs base and reference types

5. **Appendix: Legends**:

   -  Special Values Legend (floating-point and integer)
   -  Measured Accuracy Legend
   -  Performance Metrics Legend


Usage Examples
~~~~~~~~~~~~~~

.. code:: bash

   # Generate spec from local test results
   make spec

   # Generate spec with custom accuracy folder
   make spec ACC=_out_l40

   # Generate spec with multiple GPU results
   make spec ACC=_out_l40 PERF1=_out_l40 PERF2=_out_a100 PERF3=_out_h100

   # Generate spec with custom output path
   make spec SPEC_OUTPUT=docs/SPEC.md


Direct Script Usage
~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # Basic usage
   python3 scripts/make_spec.py --acc _out/ts/fpmp

   # With multiple GPU results
   python3 scripts/make_spec.py --acc _out/ts/fpmp --perf1 results/L40 --perf2 results/A100

   # With custom output
   python3 scripts/make_spec.py --acc _out/ts/fpmp --output README_fpmp_spec.md

   # Verbose mode
   python3 scripts/make_spec.py --acc _out/ts/fpmp --verbose

Typical Workflow
~~~~~~~~~~~~~~~~

.. code:: bash

   # 1. Run FPMP tests on first GPU (e.g., L40)
   make tsmp OUT=_out_l40

   # 2. Run FPMP tests on second GPU (e.g., A100)
   make tsmp OUT=_out_a100

   # 3. Generate comprehensive spec
   make spec ACC=_out_l40 PERF1=_out_l40 PERF2=_out_a100

   # 4. Review generated spec
   cat _out_l40/README_fpmp_spec.md

   # 5. Move into docs/ after verification (matches the release manifest entry
   #    docs/README_fpmp_spec.md in the top-level Makefile RELEASE_INCLUDE list)
   cp _out_l40/README_fpmp_spec.md ./docs/README_fpmp_spec.md

--------------

Common Workflows
----------------

Full CI Build
~~~~~~~~~~~~~

.. code:: bash

   # Run comprehensive CI across all configurations
   make ci

This executes:

-  Examples with inline and LTO linkage
-  Unit tests with inline and LTO linkage
-  Benchmarks with inline and LTO linkage
-  FPMP test suite with inline and LTO linkage
-  Host builds with all above

Creating a Release
~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 1. Run all FPMP tests to verify correctness
   make tsmp TIMING=n

   # 2. Run all FPEMU tests to verify correctness
   make tsemu

   # 3. Run FPMP performance tests
   make tsmp ACCURACY=n

   # 4. Generate specification document
   make spec

   # 5. Review the spec
   less _out/README_fpmp_spec.md

   # 6. Create release tarball
   make tar

   # 7. Verify tarball contents
   tar -tzf _out/cuda_multi_fp_nda_*.tar.gz | head -20

Performance Comparison Across GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # Run tests on different GPUs (requires SSH or separate machines)
   # On L40 machine:
   make tsmp OUT=_results_l40

   # On A100 machine:
   make tsmp OUT=_results_a100

   # On H100 machine:
   make tsmp OUT=_results_h100

   # Collect results and generate comparison spec
   make spec ACC=_results_l40 PERF1=_results_l40 PERF2=_results_a100 PERF3=_results_h100

--------------

Prerequisites
-------------

Required Software
~~~~~~~~~~~~~~~~~

-  **CUDA Toolkit** (11.0+) for device builds
-  **GCC** (9.0+) for host builds
-  **Python 3** for scripts
-  **libquadmath** for FP128 support

Installation
~~~~~~~~~~~~

.. code:: bash

   # Ubuntu/Debian
   sudo apt install libquadmath0

   # RHEL/CentOS
   sudo yum install libquadmath

--------------

Environment Variables
---------------------

============= =================================================
Variable      Description
============= =================================================
``CUDA_HOME`` CUDA installation path (auto-detected if not set)
``PATH``      Must include ``nvcc`` for CUDA builds
============= =================================================

--------------

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**"nvcc: command not found"**

.. code:: bash

   export PATH=/usr/local/cuda/bin:$PATH

**"libquadmath not found"**

.. code:: bash

   sudo apt install libquadmath0

**"OUT directory contains old results"**

.. code:: bash

   make clean
   # or
   rm -rf _out

**"Tests running slowly on host"**

.. code:: bash

   # Disable classification for faster host runs
   make -C ts/fpmp TARGET=host CLASSIFY=n RIGOR=20 rerun
