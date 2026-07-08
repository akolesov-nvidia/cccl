# FPMP (float-float) NaN issue in LULESH

## Problem

When running LULESH with `DATATYPE=fpmp` (`fpmp2<float>`), the simulation
produced NaN for energy values starting at cycle 2, eventually corrupting all
elements. The same code works correctly with native `double`, `float`, and `fpemu`.

## Root cause

The double-float (float-float) multiplication algorithm produces **NaN instead
of inf** when the result overflows float's representable range (~3.4e38).

The standard double-float multiply computes:

```
res_hi = a_hi * b_hi
res_lo = FMA(a_hi, b_hi, -res_hi) + a_hi * b_lo + a_lo * b_hi
```

When `a_hi * b_hi` overflows to `inf`, the error-correction term becomes:

```
FMA(a_hi, b_hi, -inf) = inf - inf = NaN
```

This is an inherent property of the double-float algorithm, not a library bug.
Adding an overflow check to every multiply would degrade performance on the
critical arithmetic path.

## How it manifests in LULESH

In `CalcMonotonicQRegionForElems_kernel`, the artificial viscosity calculation
normalizes velocity divergence ratios:

```c
Real_t norm = Real_t(1.) / (delv_xi[i] + ptiny);  // ptiny = 1e-36
delvm = delv_xi[neighbor] * norm;
```

For elements far from the shock front, `delv_xi[i] ≈ 0`, so `norm ≈ 1e36`.
When a neighboring element has a large velocity divergence from the shock wave
(e.g., `delv_xi[neighbor] ≈ -1014`), the product overflows:

```
-1014 * 1e36 = -1.014e39  (exceeds float max ≈ 3.4e38)
```

- **float**: correctly returns `-inf`, which the subsequent clamp
  `if (phi < 0) phi = 0` catches.
- **fpmp**: returns `NaN` due to the overflow mechanism above. Since
  `NaN < 0` is `false`, the clamp does not trigger, and NaN propagates
  through `qq`/`ql` → energy → pressure → forces → all elements.

## Fix applied

Changed the clamping comparisons from:

```c
if (phixi < Real_t(0.)) phixi = Real_t(0.);
```

to NaN-safe form:

```c
if (!(phixi >= Real_t(0.))) phixi = Real_t(0.);
```

The `!(x >= 0)` idiom catches both negative values and NaN (since all NaN
comparisons return false). This has zero performance cost — same instruction,
inverted branch.

The same pattern was applied to the SQRT threshold checks in the EOS:

```c
// before: if (ssc <= Real_t(.1111111e-36))
// after:
if (!(ssc > Real_t(.1111111e-36)))
```

## Guideline for fpmp users

Any code using `fpmp2<float>` should be aware that intermediate
products must stay within float's representable range. When porting code that
relies on `inf` propagation semantics (common in IEEE-754 compliant code),
replace comparisons like `x < 0` with NaN-safe equivalents `!(x >= 0)` at
clamping/guarding points.
