Examples
========

This directory contains examples demonstrating the usage of the CUDA
floating-point libraries on NVIDIA GPUs and CPUs.

Components
----------

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **FPMP**      | Multi-precision arithmetic using pairs of IEEE-754 floats (`fp32mp2`, `fp64mp2`) | [README_fpmp.md](README_fpmp.md) |
| **FPEMU**     | Emulated IEEE-754 double-precision arithmetic via integer compute units          | [README_fpemu.md](README_fpemu.md) |
| **FP64 Tool** | Drop-in `double` replacement for precision sensitivity analysis                  | [README_fp64_tool.md](README_fp64_tool.md) |
| **Fortran**   | Calling FPMP from Fortran (host with `gfortran`, device with `nvfortran`)        | [fortran/README.md](fortran/README.md) |

Building Examples
-----------------

### Build Options

| Option    | Description                                          | Default    |
|-----------|------------------------------------------------------|------------|
| `TARGET`  | Target device (`device` for GPU, `host` for CPU)     | `device`   |
| `EXAMPLE` | Specific example(s) to build (e.g., `fp32mp2`)       | all *.cpp  |
| `ARCH`    | CUDA compute capability (e.g., `86` for SM 8.6)      | `86`       |
| `OUT`     | Output directory for build artifacts                 | `_out`     |
| `VERBOSE` | Verbosity (`0`=silent, `1`=minimal, `2`=full, `3`=very verbose); aliases: `V`, `VERB` | `0` |

### Build Commands

```bash
# Build all examples for GPU
make

# Build all examples for CPU only
make TARGET=host

# Build a specific example
make EXAMPLE=fp32mp2

# Build with verbose output
make VERBOSE=1

# Rebuild from scratch
make rebuild
```

Running Examples
----------------

```bash
# Build and run all examples
make run

# Run a specific example
make run EXAMPLE=fp32mp2

# Rebuild and run
make rerun

# Run on CPU
make run TARGET=host
```

### Expected Output

Each example displays:
- Input values used for calculations
- Results of arithmetic operations
- Comparison with reference precision values
- Absolute error between computed and reference results
