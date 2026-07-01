FP64 Tool Examples
==================

Examples demonstrating **FP64 Tool** — a drop-in `double` replacement for
precision sensitivity analysis (compile-time and runtime mantissa control).

For build/run instructions and the wider examples landscape, see the
[examples README](README.md).

Available Examples
------------------

### fp64_tool.cpp — FP64 Precision Tool Demo

Demonstrates compile-time precision reduction using `fp64_tool.h`:

- **Use case**: Algorithm sensitivity analysis, mixed-precision research

Features demonstrated:
- Setting mantissa and exponent bits via `#define` macros
- Creating `fp64_tool` variables (drop-in `double` replacement)
- Arithmetic with automatic precision callbacks
- Side-by-side comparison with native `double`

### fp64_tool_runtime.cpp — FP64 Precision Tool Runtime Demo

Demonstrates runtime-configurable precision reduction:

- **Use case**: Exploring precision thresholds without recompilation

Features demonstrated:
- Changing mantissa precision at runtime via `FP64_TOOL_RUNTIME_SIZE`
- Testing precision boundaries (e.g., 1 + 2⁻⁵² with full vs. reduced mantissa)
- Verifying that precision reduction correctly rounds away small values
