# Float-Float Atomic Operations Benchmark

This benchmark measures the performance of atomic operations for `fp32mp2` (float-float) compared to native double-precision atomics.

## Overview

The benchmark creates **extreme contention** by having all threads atomically update the **same single memory location**. This represents a worst-case scenario but is realistic for:
- Global counters
- Single accumulation points
- Shared statistics

## What This Measures

1. **Native double atomicAdd** - Baseline performance
2. **fp32mp2 atomicAdd** - Float-float emulation with `fpmp2_accuracy::def`

Both use the same contention pattern for fair comparison.

## Building

```bash
# Build and run with defaults
make

# Custom configuration
make ARCH=80 NUM_BLOCKS=32 THREADS=128 NUM_ITER=100

# Just build
make build

# Rebuild from scratch
make rebuild
```

### Parameters

- `ARCH` - CUDA architecture (default: 86 for RTX 30xx/A100)
- `NUM_ITER` - Iterations per thread (default: 50)
- `THREADS` - Threads per block (default: 64)
- `NUM_BLOCKS` - Number of blocks (default: 16)
- `VERBOSE` - Verbosity level (0=silent, 1=minimal, 2=full, 3=very verbose)
- `LINKAGE` - Linkage mode (inline|static|lto)

## Expected Results

### Under High Contention (This Benchmark)

```
Operation Type                     Time (ms)    Slowdown  Result
----------------------------------------------------------------------
Native double atomicAdd                X.XX        1.00x   51.2
fp32mp2 atomicAdd                   XX.XX       10-50x   51.2
----------------------------------------------------------------------
```

**Expected slowdown: 10-50x** under extreme contention.

This is **not a bug** - it's the fundamental cost of more complex arithmetic when every CAS retry recomputes everything.

### Why So Slow?

Each failed atomicCAS attempt must recompute the entire addition:

**Native double:**
```cpp
old = atomicCAS(addr, assumed, new_val);  // 1 operation
```

**fp32mp2:**
```cpp
// Every retry does ~12 float operations:
- fabsf comparison
- fast_2_sum (3 ops)
- 3x fadd_rn
- another fast_2_sum (3 ops)
- atomicCAS
```

With 90% retry rate (common under high contention):
- Native: ~10 operations total
- fp32mp2: ~120 operations total
- **Result: 12x slowdown**

## Real-World Performance

In properly written code with hierarchical reductions:

| Contention Level | Expected Slowdown |
|------------------|-------------------|
| **Low** (distributed targets) | 1.2-1.5x |
| **Medium** (some sharing) | 2-5x |
| **High** (single target) | 10-50x |
| **Extreme** (this benchmark) | 50-100x |

## How to Improve Performance in Real Code

### 1. Use Hierarchical Reductions ✅

```cpp
__global__ void good_kernel(double* global_sum, const float* data, int n) {
    __shared__ efloat32_mp2_t block_sum;
    
    // Reduce within block first
    efloat32_mp2_t thread_sum = reduce_thread_data(data);
    efloat32_mp2_t result = blockReduce(thread_sum);
    
    // Only one atomic per block (not per thread!)
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, result);  // Much lower contention
    }
}
```

### 2. Use Shared Memory Atomics

Shared memory atomics are 10-100x faster than global memory atomics.

### 3. Batch Operations

Accumulate locally, atomic once:
```cpp
efloat32_mp2_t local_sum = 0;
for (int i = 0; i < N; i++) {
    local_sum += values[i];  // No atomic
}
atomicAdd(global_sum, local_sum);  // One atomic
```

### 4. Use Multiple Accumulators

Distribute work across multiple accumulation points:
```cpp
constexpr int NUM_ACCUMULATORS = 32;
int acc_idx = threadIdx.x % NUM_ACCUMULATORS;
atomicAdd(&accumulators[acc_idx], value);
// Reduce accumulators at the end
```

## Documentation

See the following files in this directory for detailed information:

### Quick Start
- **QUICK_REFERENCE.md** ⚡ - 30-second overview, formulas, and decision tree

### Understanding Performance
- **WHY_SO_SLOW.md** ⭐ - Simple explanation: Hardware vs software atomics
- **ATOMIC_CONTENTION_MATH.md** 📊 - Detailed mathematical analysis of 250x slowdown
- **ATOMIC_PERFORMANCE_NOTES.md** - Performance under different contention levels

### Implementation Details
- **ATOMIC_OPTIMIZATIONS.md** - Technical deep dive into atomic implementations
- **OPTIMIZATION_LIMITS.md** 🔬 - Can we optimize further? Analysis of limits
- **ATOMIC_BENCHMARK_NOTES.md** - Understanding the benchmark scenarios
- **BENCHMARK_FIXES.md** - History of bugs fixed during development
- **MIGRATION_NOTES.md** - How this benchmark was organized

## Key Takeaways

1. ✅ The "slow" performance is **correct behavior** under extreme contention
2. ✅ The atomic implementation is working as designed
3. ✅ Real applications should use hierarchical reductions
4. ✅ For low contention: ~1.5x slower than native double
5. ✅ For high contention: 10-50x slower (shown here)

**Both are correct!** The difference is contention level.

## Integration with 8-Byte Alignment

This benchmark validates the 8-byte alignment implementation:
- ✅ Atomic operations use 64-bit `atomicCAS` directly
- ✅ No manual alignment handling needed
- ✅ Works on both host and device code

## Clean Up

```bash
make clean
```

## See Also

- Main documentation: `../../README_ALIGNMENT.md`
- Atomic test suite: `../../examples/float_float_atomic.cpp`
- Alignment compatibility: `../../ALIGNAS_COMPATIBILITY.md`

