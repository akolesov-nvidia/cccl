.. _libcudacxx-extended-api-fp-fptool:

fptool: Instruments for Floating-Point Arithmetic
=================================================

.. toctree::
   :hidden:
   :maxdepth: 1

   fp64_custom <fptool_custom>
   fpmp2_stat <fptool_stat>

The other two sub-components give you arithmetic to compute with. ``fptool`` gives you two
instruments for finding out what a computation *needs* and what it is *doing* — questions that are
otherwise answered by guesswork, or by rewriting the algorithm in a narrower type and hoping the
rewrite was faithful.

.. list-table::
   :widths: 30 34 36
   :header-rows: 1

   * - **Tool**
     - **Answers**
     - **How**

   * - :ref:`fp64_custom\<E, M\> <libcudacxx-extended-api-fp-fptool-custom>`
     - How little precision does this algorithm need? Is it the mantissa or the dynamic range that
       matters?
     - Rounds to an arbitrary exponent and mantissa width after every operation, so the
       computation runs as though the hardware had that format

   * - :ref:`fp32mp2_stat, fp64mp2_stat <libcudacxx-extended-api-fp-fptool-stat>`
     - What did the arithmetic actually do? Is the second limb carrying anything? Is anything
       silently corrupt?
     - Wraps an ``fpmp2`` type, computes bit-identical results, and records the operations and
       numerical events on the device

The difference between them is worth stating plainly, because it decides how each is used:
**``fp64_custom`` changes the arithmetic, and the ``_stat`` types only observe it.** One is there
to make results worse in a controlled way, so you can see how much worse the algorithm tolerates.
The other is there to leave results alone — bit-identical to the type it wraps, by construction —
so that an instrumented run can be compared against a plain one, and a difference between them
means a race or an uninitialized value rather than a rounding change.

Neither is a type to ship in
----------------------------

Both are diagnostics, and both cost rather than save:

- ``fp64_custom`` emulates a narrow format *on native FP64*, adding a rounding step to each
  operand and to each result. Modelling BF16 with it is therefore slower than ``double``, not
  faster — in the integration :ref:`measured here <libcudacxx-extended-api-fp-fptool-custom-pi>`
  every reduced format costs about 10% over plain ``double``. What it buys is the answer to
  "would this work in BF16", on hardware that has no BF16 arithmetic, for a format that no
  hardware need ever implement.
- an instrumented ``_stat`` run is **two to three orders of magnitude slower** than the plain
  type, every operation updating one device-wide record through atomics.

So the shape of a study is: reach for one of these to answer a question, read the answer, and put
it back. Both are designed for that — ``fp64_custom`` is a drop-in for ``double`` and the ``_stat``
types are drop-ins for the ``fpmp2`` types they wrap, so in each case the change is to a type
alias and the algorithm is left alone. Keeping that alias behind a build flag is the arrangement
worth copying.

Using the header
----------------

.. code-block:: cuda

    #include <cuda/fptool>

One header carries both tools and their math functions, as ``<cuda/fpmp>`` does. The one
consequence to know is that including it costs about a fifth more than the types alone, since the
statistics math wrappers rest on the fpmp math surface.

Both tools work in host and device code from the same source, but with an asymmetry that matters
in opposite directions:

- ``fp64_custom`` keeps **independent host and device sizes**, deliberately, so a full-precision
  host reference can run alongside a reduced device computation.
- ``_stat`` collection is **device-only**. The same source compiles and runs on the host, where
  the wrapper is a transparent pass-through and nothing is gathered — so a program doing part of
  its arithmetic on the host reports a count below what its closed form predicts, with no
  diagnostic.

Everything either tool names is specific to the component and carries the namespace. Unlike
``fpmp`` and ``fpemu``, there is little here to find by argument-dependent lookup: the setters,
getters and readout functions take only an ``int`` or a stream, so there is no operand of a
component type to look in.

.. seealso::
   :ref:`fp64_custom <libcudacxx-extended-api-fp-fptool-custom>` — **emulating a narrower format
   on native FP64**, for sensitivity studies.

   :ref:`fpmp2_stat <libcudacxx-extended-api-fp-fptool-stat>` — **recording what the arithmetic
   did**, without changing what it computes.
