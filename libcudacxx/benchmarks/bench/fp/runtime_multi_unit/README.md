Runtime Multi-Unit Test
=======================

This benchmark verifies that `fp64_tool.h` with `FP64_TOOL_RUNTIME_SIZE` can be safely included in **multiple compilation units** within a single binary without causing linker errors, symbol conflicts, or overdefinition issues.

Overview
--------

When `fp64_tool.h` is included in more than one `.cpp` file, the runtime-size global variables and setter functions must have the correct linkage to avoid duplicate-symbol errors at link time. This test compiles two separate translation units — each including `fp64_tool.h` — and links them into one executable.

Test Structure
--------------

| File | Role |
|------|------|
| `runtime_multi_unit.cpp` | Main unit — includes `fp64_tool.h`, runs all tests, calls functions from both units |
| `runtime_multi_unit_aux.cpp` | Auxiliary unit — also includes `fp64_tool.h`, exposes `test_from_aux_unit()` and `set_mantissa_from_aux_unit()` |
| `Makefile` | Compiles each unit separately, then links them |

Test Cases
----------

All tests use the operation `1 + 2^-52` (ULP of 1.0 in FP64):

| # | Description | Mantissa | Expected |
|---|-------------|----------|----------|
| 1 | Main unit, full precision | 52 bits | `1 + 2^-52` (small term preserved) |
| 2 | Auxiliary unit, full precision | 52 bits | `1 + 2^-52` |
| 3 | Main unit, reduced precision | 50 bits | `1.0` (small term lost) |
| 4 | Auxiliary unit, precision changed only in main unit | 52 bits | `1 + 2^-52` (isolation check) |
| 5 | Auxiliary unit, reduced via its own setter | 50 bits | `1.0` |

Test 4 is the key multi-unit check: because each compilation unit has its own `static` copy of the host mantissa variable, changing it in the main unit must **not** affect the auxiliary unit.

Building
--------

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `TARGET` | Target device (`device`, `host`) | `device` |
| `ARCH` | CUDA compute capability (e.g. `86`) | `86` |
| `OUT` | Output directory | `_out` |
| `VERBOSE` | Verbose mode (`0`=silent, `1`=minimal, `2`=full) | `1` |
| `EXTRA_FLAGS` | Extra compiler/linker flags | — |

### Build Commands

```bash
# Build and run on GPU (default)
make

# Build and run on CPU
make TARGET=host

# Build only (no run)
make build

# Rebuild from scratch and run
make rerun

# Clean generated files
make clean
```

### Manual Compilation

CPU:
```bash
g++ -std=c++17 -O2 -I../../include -c runtime_multi_unit.cpp -o runtime_multi_unit.o
g++ -std=c++17 -O2 -I../../include -c runtime_multi_unit_aux.cpp -o runtime_multi_unit_aux.o
g++ -std=c++17 -O2 runtime_multi_unit.o runtime_multi_unit_aux.o -o runtime_multi_unit.exe
```

CUDA:
```bash
nvcc -std=c++17 -O2 -I../../include -x cu -rdc=true -c runtime_multi_unit.cpp -o runtime_multi_unit.o
nvcc -std=c++17 -O2 -I../../include -x cu -rdc=true -c runtime_multi_unit_aux.cpp -o runtime_multi_unit_aux.o
nvcc -std=c++17 -O2 -arch=sm_86 -rdc=true runtime_multi_unit.o runtime_multi_unit_aux.o -o runtime_multi_unit.exe
```

Expected Output
---------------

```
FP64 Tool Runtime Size Multi-Unit Test
======================================

This test verifies that fp64_tool.h can be included in
multiple compilation units without linker errors.

Test 1: Main unit, full mantissa (52 bits)
  Expected: 1.00000000000000022e+00
  Got:      1.00000000000000022e+00
  Status:   PASS

Test 2: Auxiliary unit, full mantissa (52 bits)
  ...
  Status:   PASS

Test 3: Main unit, reduced mantissa (50 bits)
  ...
  Status:   PASS

Test 4: Auxiliary unit, mantissa changed in main compilation unit should have no effect
  ...
  Status:   PASS

Test 5: Auxiliary unit, reduced mantissa (50 bits)
  ...
  Status:   PASS

========================================
Overall Test: PASS
========================================
```
