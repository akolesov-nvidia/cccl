# Floating-point component benchmarks

NVBench benchmarks of the `cuda/fp` component. `fpmp/` covers the multi-precision pair types
(`cuda::fp32mp2`, `cuda::fp64mp2`) and their math library; one file per function family, one
executable per file:

| target | functions |
|---|---|
| `libcudacxx.bench.fp.fpmp.arith.base` | add, sub, mul, div, fma, sqrt, rsqrt |
| `libcudacxx.bench.fp.fpmp.explog.base` | exp, exp2, exp10, expm1, log, log2, log10, log1p |
| `libcudacxx.bench.fp.fpmp.pow.base` | pow, cbrt, rcbrt, hypot, rhypot |
| `libcudacxx.bench.fp.fpmp.trig.base` | sin, cos, tan, sinpi, cospi |
| `libcudacxx.bench.fp.fpmp.invtrig.base` | asin, acos, atan, atan2 |
| `libcudacxx.bench.fp.fpmp.hyperbolic.base` | sinh, cosh, tanh, asinh, acosh, atanh |
| `libcudacxx.bench.fp.fpmp.special.base` | erf, erfc, erfcx, erfinv, erfcinv, normcdf, normcdfinv, boys_f0 |
| `libcudacxx.bench.fp.fpmp.rounding.base` | floor, ceil, round, trunc, rint, nearbyint, fmod, remainder |

Every executable holds two benchmarks, `throughput` and `latency`, over the type axes `Func`,
`Type` (`fp32mp2`, `fp64mp2`) and `Accuracy` (`low`, `mid`, `high`), so a row of the CSV output is
one (function, type, level) cell, and two more, `ref_throughput` and `ref_latency`, that time the
reference precision over `Func` alone:

- **throughput**: 2048 × 256 threads, each a chain of 1024 dependent evaluations. nvbench reports
  the GPU time and elements per second; the summary `ev/clk/SM` is evaluations per SM clock per SM,
  from the mean GPU time and the device's default SM clock.
- **latency**: one thread, a chain of 4096 dependent evaluations; `clk/ev` is SM clocks per evaluation.
- **ref_throughput, ref_latency**: the same chains on the reference precision, the double
  expression of the declaration on a `double` (Type `fp64`), which is what fp32mp2 stands in for
  on a device whose fp64 rate is low. fp64mp2 would be compared with binary128, which the device
  does not have yet. The table turns them into **Gain tp** (the cell's `ev/clk/SM` over the
  reference's) and **Gain lat** (the reference's `clk/ev` over the cell's): both above 1 when the
  pair is faster than the reference precision.
- **u2**: the maximum relative error over 2^20 random normalized pairs against a double reference,
  in units of u² = 2^-48, the fp32mp2 pair's own half-ulp (the unit of the fpmp error-bounds page).
  The fp64mp2 types would need a binary128 reference, which the device does not have yet: they are
  checked against double, show `-` here and cap `Bits` at 53.
- **Bits**: -log2 of that maximum relative error.
- **Check**: a rough verdict on that error with the thresholds of the cccl-fpdev ts/fpmp suite:
  `FAIL` above 1e-7 for fp32mp2 (the extended precision is gone) and above 1e-15 for fp64mp2,
  `WARN` above 1e-13 for fp32mp2 (the fp64mp2 WARN level, 1e-28, needs the binary128 reference),
  `PASS` otherwise; a result that is not finite where the reference is finite also fails. The
  check catches a broken function; a regression below these levels shows in `u2` between runs.
- **Regs**: registers per thread of the chain kernel.
- **Worst at**: the arguments of the sample that gave the maximum relative error, as C99
  hexadecimal floats (`0x1.8p+1`, exact; in the CSV and JSON only), to see where an accuracy
  change comes from.

The CUDA double math library is accurate to a few ulps of a double, which is about a tenth of a
u²: the accuracy column is for tracking regressions, not for the last digit of the specification.

## Building and running

```bash
cmake --preset libcudacxx-benchmark -B build/libcudacxx-benchmark -DCMAKE_CUDA_ARCHITECTURES=native
ninja -C build/libcudacxx-benchmark libcudacxx.bench.fp.fpmp.arith.base
build/libcudacxx-benchmark/bin/libcudacxx.bench.fp.fpmp.arith.base --csv out/arith.csv --md stdout
```

A family takes 7 to 20 seconds to run; `--min-time 0.1 --max-noise 2` is enough for a quick look.
Single cells: `-b throughput -a Func=div -a Type=fp32mp2 -a Accuracy=mid`.

## The table

The SASS instruction count of one evaluation is not a run-time quantity. Each cell also compiles
a `__noinline__` function with exactly one evaluation, which `nvdisasm` lists as a labelled block
of its own, and `benchmarks/scripts/fp_table.py` counts its instructions from the executable and
joins them to the CSV, merging the throughput and latency rows:

```bash
for f in arith explog pow trig invtrig hyperbolic special rounding; do
  build/libcudacxx-benchmark/bin/libcudacxx.bench.fp.fpmp.$f.base --csv out/$f.csv --json out/$f.json --quiet
done
python3 benchmarks/scripts/fp_table.py --bin-dir build/libcudacxx-benchmark/bin --csv-dir out --out out/fpmp.csv --md out/fpmp.md
```

giving one row per cell: `Func, Type, Accuracy, u2, Bits, Check, ev/clk/SM, Gain tp, clk/ev,
Gain lat, SASS, Regs, Worst at, Family, Device`, plus one row per function for the reference precision
(Type `fp64`, Accuracy `-`, with its own `SASS` and `Regs`). nvbench's own columns (GPU time,
elements per second) stay in the per-family CSVs. The JSON files are what `ci/bench` compares
between two refs.

## Adding a function

Declare it in its family file with `FPB_OP1`, `FPB_OP2` or `FPB_OP3`: the pair expression, the
double reference, the accuracy input range of each argument (`fpb::lin(lo, hi)` uniform in value,
`fpb::logr(lo, hi[, signed])` uniform in the exponent) and the performance input
(`FPB_PERF_DEFAULT` is the suite's `1 + tid * 1e-4`; give the function's own start and stride when
that leaves its domain), then add it to the family's `type_list`.

Two things the shared header (`fpmp_bench.cuh`) gets right that are easy to break: the chain
feeds both limbs of the result into the next input and the final sink reads both limbs, or the
compiler deletes the low-limb arithmetic of the `low` level (whose high limb does not depend on
it) and `low` add measures as one FADD; and `nvbench::state::add_summary` returns a reference into
a vector, so each summary is filled before the next one is added.
