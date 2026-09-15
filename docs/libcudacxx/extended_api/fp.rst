.. _libcudacxx-extended-api-fp:

FP Component
============

.. toctree::
   :hidden:
   :maxdepth: 1

   fp/fpmp

The FP component provides floating-point types that give you arithmetic the hardware does not
offer directly: more precision than a ``double``, ``double`` precision without FP64 units, less
precision than any hardware implements, or the same arithmetic with a record of what it did.
Every type works in both host and device code from the same source, so a computation written
against one of them needs no separate host implementation.

At a high level, the component provides:

.. list-table::
   :widths: 20 25 35 20 20
   :header-rows: 1

   * - **Sub-component**
     - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`fpmp <libcudacxx-extended-api-fp-fpmp>`
     - ``<cuda/fpmp>``, ``<cuda/fpmp_math>``
     - Double-word arithmetic built from pairs of IEEE floats, reaching more mantissa than the
       hardware has: ``fp32mp2`` (46 significand bits = 2×24 − 2) and ``fp64mp2``
       (104 = 2×53 − 2)
     - CCCL 3.6.0
     - CUDA 13.6

   * - fpemu
     - ``<cuda/fpemu>``
     - IEEE-754 double precision emulated with integer and single-precision operations, for
       targets where FP64 is slow or absent: ``fp64emu``, and ``fp64emu_unpacked`` for chains
       of operations
     - CCCL 3.6.0
     - CUDA 13.6

   * - fptool
     - ``<cuda/fptool>``
     - Instrumentation rather than new arithmetic: any narrower format emulated on native FP64
       (``fp64_custom<E, M>``), and ``fpmp2`` with a record of the operations it performed
       (``fp32mp2_stat``, ``fp64mp2_stat``)
     - CCCL 3.6.0
     - CUDA 13.6

The first two rows and ``fp64_custom`` change the arithmetic; the ``_stat`` types only observe it.

Why the component exists
------------------------

Precision is decided by the hardware, and the hardware is not obliged to offer the precision an
algorithm needs at a price the algorithm can pay. The component exists because that gap opens in
four different directions.

**Below FP64.** On many recent parts the FP64 pipelines are a fraction of the FP32 ones, so a
computation that needs more than ``float`` can pay an order of magnitude for asking for
``double``. The ratio is a property of the part rather than of the code, and it is measured per
platform in the :doc:`fpmp specification <fp/fpmp_spec>`. Where it is
steep, ``fp32mp2`` does the same work on the FP32 units the part has in abundance.

**Above FP64.** There is typically no IEEE-754 binary128 hardware at all, and a fully
IEEE-correct software binary128 is expensive. ``fp64mp2`` reaches 104 significand bits
(2×53 − 2) out of ordinary FP64 operations, which covers many of the uses a quad-precision type
would be reached for.

**Below FP32.** Hardware implements a handful of narrow formats and no more, which makes the
question "how little precision does this algorithm actually need?" difficult to ask. Answering
it is what ``fp64_custom<E, M>`` is for: it rounds to an arbitrary exponent and mantissa width
after every operation, so a computation runs as though the hardware had that format, and the
exponent and mantissa can be varied independently to find out which of the two matters.

**Alongside the arithmetic.** When a result is wrong, native arithmetic keeps no record of how it
got that way. The ``_stat`` types compute bit-identical results to their ``fpmp2`` counterparts
while recording operation counts, cancellation and overflow events, and what the operands looked
like.

Which one do I want?
--------------------

**More precision than** ``double``. ``fp64mp2``, a double-double reaching 104 significand bits
(2×53 − 2) out of FP64 operations.

**Roughly** ``double`` **precision, but FP64 is slow on the target.** Two answers, and the choice
turns on what you need from ``double``:

- ``fp32mp2`` if 46 significand bits (2×24 − 2) are enough and you can live with ``float``'s
  exponent range. It is 8 bytes and runs on FP32 units.
- ``fp64emu`` if you need ``double``'s full semantics — its range, its specials, and results
  identical to ``double`` bit for bit. Also 8 bytes in the packed form.

``fp64emu`` comes in two representations. The packed one is the drop-in: 8 bytes holding the
same bit pattern a ``double`` holds, implicit conversion from the built-in types, and results
identical to ``double`` at the highest accuracy level. ``fp64emu_unpacked`` instead keeps sign,
exponent and mantissa in separate fields, which avoids re-packing between operations and extends
the 53-bit significand by 9 bits, so a chain of operations runs on 62 significand bits and rounds
to the storage format once, at the pack, rather than after every operation. That makes it both
faster and more accurate than the packed form over a run of operations. The costs are 16 bytes
per value instead of 8, which can cost occupancy where many values are live; a layout that is not
the IEEE one, so conversions into it are written out rather than implicit; and results that are
therefore no longer bit-identical to ``double`` across a chain, even at the highest accuracy
level.

**Less precision, on purpose.** ``fp64_custom<E, M>`` rounds to a narrower format after every
operation. This is how to find out whether an algorithm survives BF16, and whether it is the
mantissa or the dynamic range that matters.

**To know what the arithmetic is currently doing.** ``fp32mp2_stat`` and ``fp64mp2_stat`` compute
identical results to ``fp32mp2`` and ``fp64mp2`` and record what the operations did, including
whether the second limb is carrying anything at all.

Accuracy levels
---------------

``fpmp`` and ``fpemu`` each come at three accuracy levels — ``low``, ``mid`` and ``high`` —
selected through a template parameter or through the named aliases, such as ``fp32mp2_high`` and
``fp64emu_low``, with each step trading speed against accuracy per operation. The unsuffixed name
takes a default that differs between the two: ``mid`` for ``fpmp``, and ``high`` for ``fpemu``, so
emulated ``double`` is IEEE-correct unless asked otherwise. ``fp64_custom`` has no levels, since
its accuracy is set by the exponent and mantissa widths. What each level does, and what it costs,
is covered on the sub-component pages.

One source for host and device
------------------------------

The operations on every type in the component are ``__host__ __device__``, so a single build
carries both copies and one run can exercise both. A function written against these types is
therefore ordinary C++ that happens to be callable from a kernel, and writing it as a template
is the form worth copying: one definition covers every type in the component, and the precision
of the computation becomes a single line to change.

.. code-block:: cuda

    #include <cuda/fpmp>

    template <typename FpType>
    __host__ __device__ FpType dot(const FpType* a, const FpType* b, int n)
    {
      FpType acc{0.0};
      for (int i = 0; i < n; ++i)
      {
        acc = acc + a[i] * b[i];
      }
      return acc;
    }

    // The one place the precision is chosen.
    using fp_t = cuda::experimental::fp64mp2;

    __global__ void dot_kernel(const fp_t* a, const fp_t* b, int n, fp_t* out)
    {
      *out = dot(a, b, n);   // FpType deduced from the arguments
    }

    fp_t dot_on_host(const fp_t* a, const fp_t* b, int n)
    {
      return dot(a, b, n);   // the same definition, no device involved
    }

The types construct from the built-in arithmetic types and overload the usual operators, so the
body of a ``double`` computation keeps its shape when the type underneath it is swapped. Nothing
in ``dot`` names a precision, so the same definition serves ``fp32mp2`` on a part where FP64 is
rationed and ``fp64mp2`` where the algorithm needs more than ``double``.

Namespace and stability
-----------------------

The component lives in the ``cuda::experimental`` namespace, which will be promoted to ``cuda::``
later. The standard-named math functions are found by argument-dependent lookup, so they can be
called unqualified and a body of ``double`` code keeps its call sites. The component's own
functions, which have no ``double`` counterpart, are found the same way, but the sub-component
pages spell out the namespace on them to mark them as belonging to the component:

.. code-block:: cuda

    namespace cudax = cuda::experimental;

    cudax::fp64mp2 x{2.0};

    auto r = sqrt(x);                  // a standard name: a body of double code keeps this call site
    auto s = cudax::renormalize(x);    // no double counterpart; renormalize(x) also compiles

Examples
--------

Four complete programs live in
`examples/cudax/fp <https://github.com/NVIDIA/cccl/tree/main/examples/cudax/fp>`__, one per
sub-component. Each prints the inputs and the result of every operation it performs, once from
the host and once from the device, so the two can be compared directly.
