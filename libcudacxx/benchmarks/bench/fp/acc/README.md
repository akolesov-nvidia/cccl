# ACC vs ADD Performance Benchmark

This benchmark compares the performance of the optimized single-component accumulate operation (`__fpmp2_acc`, `operator+=`) against using full mp2+mp2 addition when adding a single-precision value to a multi-precision number.

## Overview

The `ACC` operation is optimized for the common case of accumulating a single float/double into a multi-precision (hi, lo) pair. It saves approximately **6 floating-point operations** compared to full addition by avoiding the 2Sum for the low parts (since the single-component contribution has `lo=0`).

### Algorithm Comparison

**ACC (Optimized Accumulate):**
```cpp
(new_hi, err) = 2Sum(acc_hi, c)        // Add c to high part
new_lo = acc_lo + err                   // Accumulate error into low part
(res_hi, res_lo) = Fast2Sum(new_hi, new_lo)  // Normalize
```

**ADD (Full mp2+mp2 Addition):**
```cpp
(s_hi, s_lo) = 2Sum(a_hi, b_hi)        // Add high parts
(t_hi, t_lo) = 2Sum(a_lo, b_lo)        // Add low parts (unnecessary when b_lo=0!)
c = s_lo + t_hi                         // Merge middle terms
(v_hi, v_lo) = Fast2Sum(s_hi, c)       // Normalize
w = t_lo + v_lo                         // Absorb error
(r_hi, r_lo) = Fast2Sum(v_hi, w)       // Final normalize
```

## Test Variants

| Variant | ACC Implementation | ADD Implementation |
|---------|-------------------|-------------------|
| FAST | `__fpmp2_acc_fast` (no normalization) | `__fpmp2_add_fast` with (c, 0.0) |
| DEFAULT | `__fpmp2_acc` (Dekker-style) | `__fpmp2_add` with (c, 0.0) |
| ACCURATE | `__fpmp2_acc_accurate` (FPAN-style) | `__fpmp2_add_accurate` with (c, 0.0) |

## Usage

```bash
# Build and run on GPU (default)
make

# Run with custom parameters
make REPS=2048 NUM_BLOCKS=2048 run

# Build for specific architecture
make ARCH=80 rebuild   # A100
make ARCH=89 rebuild   # RTX 40xx

# Clean up
make clean
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TARGET` | device | Target platform (device/host) |
| `ARCH` | 86 | CUDA architecture (e.g., 75=Turing, 80=Ampere, 86=RTX 30xx) |
| `NUM_ITER` | 10 | Number of timing iterations |
| `THREADS` | 256 | Threads per CUDA block |
| `NUM_BLOCKS` | 1024 | Number of CUDA blocks |
| `REPS` | 1024 | Operations per thread |
| `UNROLL` | 64 | Loop unroll factor |

## Expected Results

The ACC operation should be faster than ADD because it:
1. Avoids 2Sum on low parts (contribution has `lo=0`)
2. Uses direct error accumulation into existing low part
3. Has shorter critical path for normalization

Expected speedup: **~1.3-1.5x** for normalized methods (DEFAULT, ACCURATE), less for FAST method.

## Example Output

```
================================================================================
  ACCUMULATE (ACC) vs ADDITION (ADD) BENCHMARK
  Comparing Optimized Single-Component Accumulate vs Full mp2+mp2 Addition
================================================================================

Configuration:
  Repetitions/thread:   1024
  Unroll factor:        64
  Iterations:           10
  Execution:            NVIDIA GeForce RTX 3090
  ...

================================================================================
  BENCHMARK RESULTS
================================================================================
Method       | ACC (ms)   | ADD (ms)    | ACC GOPS  | ADD GOPS  | Speedup
-------------+------------+-------------+-----------+-----------+----------
FAST         |     0.8234 |      1.0123 |    319.45 |    259.89 |    1.23x
DEFAULT      |     1.2456 |      1.8901 |    211.23 |    139.15 |    1.52x
ACCURATE     |     1.5678 |      2.3456 |    167.89 |    112.14 |    1.50x
================================================================================
```

## Use Cases

The optimized `ACC` operation is particularly useful for:

1. **Summation loops**: Accumulating many single-precision values into a running sum
   ```cpp
   fp32mp2 sum(0.0f);
   for (float x : values) {
       sum += x;  // Uses optimized ACC, not full mp2+mp2 ADD
   }
   ```

2. **Kahan-style compensated summation**: Adding error compensation terms
3. **Dot product accumulation**: Accumulating partial products
4. **Running averages and statistics**: Incremental updates
