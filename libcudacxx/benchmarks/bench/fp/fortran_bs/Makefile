###########################################################################################
#
#  Makefile for FPMP `fortran_bs` Fortran benchmark (CUDA Fortran + OpenACC)
#
#  Compares accuracy and throughput of GPU Black-Scholes option pricing
#  across two arithmetic types — real(real64), type(fp32mp2) — across two
#  execution paths sharing the same workload, reference, and FPMP library:
#
#    device  - CUDA Fortran (nvfortran -cuda),    attributes(global) kernels
#    acc     - OpenACC      (nvfortran -acc=gpu), !$acc parallel loop kernels
#
#  There is no host-only target; the host portion of each binary just
#  generates inputs, computes the fp64 reference, and prints the result
#  table.  See README.md for the full description.
#
#  Makefile parameters:
#  --------------------
#    `ARCH`        = CUDA sm_xx (unset: nvidia-smi, else 86)
#    `OUT`         = output directory (default: _out)
#    `NUM_OPTIONS` = number of options to price per kernel launch
#                    (default: 1048576 = 1M)
#    `N_REPS`      = timing repetitions, fastest is reported (default: 5)
#    `BLOCK_SIZE`  = CUDA threads per block, also OpenACC `vector_length`
#                    (default: 256)
#    `NUM_BLOCKS`  = CUDA grid size; for CUDA Fortran's grid-stride loop,
#                    NB * BLOCK_SIZE threads cooperate on N options.  For
#                    OpenACC's one-thread-per-option scheduling this just
#                    sets the vector_length and the partials-buffer size
#                    (default: 2048)
#    `MIN_BATCH_MS_DEV` = CUDA Fortran `device` auto-tune target: minimum
#                        milliseconds per timed cudaEvent window per
#                        kernel (default: 500; the device path's per-launch
#                        overhead is ~10 us so a 500 ms window keeps the
#                        per-kernel launch count well into the thousands)
#    `MIN_BATCH_MS_ACC` = OpenACC `acc` auto-tune target: minimum
#                        milliseconds per timed cudaEvent window per
#                        kernel (default: 100; OpenACC's per-`!$acc
#                        parallel` overhead is ~1 ms so a 100 ms window
#                        keeps total wallclock reasonable)
#    `MIN_BATCH_MS`    = convenience override.  When set on the make
#                        command line it overrides BOTH MIN_BATCH_MS_DEV
#                        and MIN_BATCH_MS_ACC at once.
#    `KERNEL_LAUNCHES` = GPU kernel launches per timed cudaEvent window;
#                        0 = auto-tune to MIN_BATCH_MS_{DEV,ACC} per
#                        kernel; >0 = fixed override (default: 0; applies
#                        to both the `device` and `acc` targets)
#    `VERBOSE`     = verbose mode (`0` (silent), `1` (minimal), `2` (full),
#                    `3` (very verbose; also enables nvfortran `-Minfo`));
#                    aliases: `V`, `VERB`
#
#  Build targets:
#  --------------
#    `all`     - build and run the CUDA Fortran `device` benchmark (default).
#                The OpenACC `acc` benchmark is OPT-IN and not built by `all`
#                because the natural overloaded-operator surface for
#                `type(fp32mp2)` arithmetic inside an `!$acc parallel loop`
#                is affected by known nvfortran 25.11 -acc=gpu codegen
#                bugs.  Two independent failure modes have been isolated
#                into minimal reproducers and filed upstream; see:
#                  issues/nvfortran_openacc_indexed_global_array_store_marshalling/
#                  issues/nvfortran_openacc_chained_derived_type_return/
#                Set `WITH_ACC=1` to include the OpenACC benchmark in
#                `all` once the compiler ships a fix, or invoke `make
#                acc` directly to exercise the failing path explicitly.
#    `build`   - build the device executable only (also acc with WITH_ACC=1)
#    `device`  - build and run CUDA Fortran device benchmark
#    `acc`     - build and run OpenACC device benchmark (explicit
#                invocation always works; results are currently
#                affected by the upstream nvfortran 25.11 codegen bugs
#                referenced above)
#    `run`     - same as `all`
#    `rerun`   - clean + build + run
#    `rebuild` - clean + build
#    `clean`   - remove generated files
#    `help`    - show this message
#
#  Compiler availability (CI-friendly auto-skip):
#  ----------------------------------------------
#    nvfortran is auto-detected at parse time via `command -v`.  If it is
#    missing, both targets become no-op stubs that print a warning and exit
#    successfully so the root `ci` target stays green on environments
#    without the NVIDIA HPC SDK.
#
###########################################################################################

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Dev-only library build (lib/make/lib.mk + lib/src/ + lib/include/) lives in the sibling cccl-fpdev
# repo; overridable (the cccl-fpdev root Makefile passes it explicitly).
FPDEV_DIR ?= $(abspath $(THIS_DIR)/../../../../../../cccl-fpdev)
MFP_DIR  := $(THIS_DIR)/../..
OUT      ?= $(THIS_DIR)/_out

# CUDA sm_xx: use ARCH if set; else first GPU (nvidia-smi); else 86.
ifndef ARCH
    _fpmp_cuda_arch_cap := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
    ifeq ($(strip $(_fpmp_cuda_arch_cap)),)
        ARCH := 86
    else
        ARCH := $(_fpmp_cuda_arch_cap)
    endif
endif

LIB_BASE     ?= cufp
NUM_OPTIONS  ?= 1048576
N_REPS       ?= 5
BLOCK_SIZE   ?= 256
NUM_BLOCKS   ?= 2048

# GPU timing window controls.  Each timed cudaEvent window covers a
# per-kernel batch of launches sized to run for at least MIN_BATCH_MS_*
# milliseconds; the measurement therefore stays well above timer / launch-
# overhead noise.  The CUDA Fortran (`device`) and OpenACC (`acc`)
# targets need different defaults:
#
#   MIN_BATCH_MS_DEV = CUDA Fortran target (default: 500).
#   MIN_BATCH_MS_ACC = OpenACC target      (default: 100).
#   MIN_BATCH_MS     = convenience override (no default); when set on the
#                      make command line, overrides both at once.
#   KERNEL_LAUNCHES  = 0 = auto-tune, >0 = fixed batch count for every kernel.
MIN_BATCH_MS_DEV ?= 500
MIN_BATCH_MS_ACC ?= 100
ifneq ($(origin MIN_BATCH_MS),undefined)
    MIN_BATCH_MS_DEV := $(MIN_BATCH_MS)
    MIN_BATCH_MS_ACC := $(MIN_BATCH_MS)
endif
KERNEL_LAUNCHES ?= 0

# Library `EXTRA_FLAGS` -- fp32mp2 is always enabled in the lib; fp64mp2
# is enabled too (it costs nothing at compile time and lets users tweak
# the source to add an fp64mp2 BS row if they want a really accurate
# device reference).
LIB_EXTRA_FLAGS :=

# Compiler availability detection (parse-time, run once).
HAS_NVFORTRAN := $(shell command -v nvfortran 2>/dev/null)

DEVICE_SUFFIX ?=

DEV_LIB     = $(OUT)/lib/lib$(LIB_BASE)$(DEVICE_SUFFIX).a
FORTRAN_DIR = $(MFP_DIR)/include/fortran

DEV_EXE = $(OUT)/fortran_bs_device
ACC_EXE = $(OUT)/fortran_bs_acc

DEV_SRC = $(THIS_DIR)/fortran_bs_device.cuf
ACC_SRC = $(THIS_DIR)/fortran_bs_acc.f90

# Common -D flags that apply unchanged to both build targets.  The two
# GPU paths extend it with their own MIN_BATCH_MS value.
COMMON_CPP_FLAGS = -DNUM_OPTIONS=$(NUM_OPTIONS) -DN_REPS=$(N_REPS) \
                   -DBLOCK_SIZE=$(BLOCK_SIZE) -DNUM_BLOCKS=$(NUM_BLOCKS) \
                   -DKERNEL_LAUNCHES=$(KERNEL_LAUNCHES)

# Per-target -DMIN_BATCH_MS so the CUDA Fortran and OpenACC benchmarks
# can pick window targets independently.  Both source files use the
# same `MIN_BATCH_MS` preprocessor name internally; the per-target
# value lands in the `-D` for that build's nvfortran invocation.
DEV_CPP_FLAGS = $(COMMON_CPP_FLAGS) -DMIN_BATCH_MS=$(MIN_BATCH_MS_DEV)
ACC_CPP_FLAGS = $(COMMON_CPP_FLAGS) -DMIN_BATCH_MS=$(MIN_BATCH_MS_ACC)

# verbose mode
VERBOSE_MODE = 0
ifneq (,$(filter $(V) $(VERBOSE) $(VERB), 1))
    VERBOSE_MODE = 1
else ifneq (,$(filter $(V) $(VERBOSE) $(VERB), 2))
    VERBOSE_MODE = 2
else ifneq (,$(filter $(V) $(VERBOSE) $(VERB), 3 y Y yes Yes full Full f F))
    VERBOSE_MODE = 3
endif

# nvfortran -Minfo diagnostics (OpenACC kernel schedule + cross-procedure
# inline decisions for the FPMP routines).  Noisy; gate behind V=3.
NVF_MINFO :=
ifeq ($(VERBOSE_MODE),3)
    NVF_MINFO := -Minfo=accel,inline
endif

.PHONY: all build run rerun rebuild device acc clean help

# OpenACC opt-in switch.  By default the fortran_bs OpenACC benchmark
# is excluded from `all`/`build`/`run` because the natural overloaded-
# operator surface for `type(fp32mp2)` arithmetic inside an `!$acc
# parallel loop` is affected by known nvfortran 25.11 -acc=gpu codegen
# bugs (see the bug reports at
# issues/nvfortran_openacc_indexed_global_array_store_marshalling/ and
# issues/nvfortran_openacc_chained_derived_type_return/).  Pass
# `WITH_ACC=1` on the make command line to include `acc` in the
# default chain again.  The `acc` target itself remains available for
# explicit invocation regardless of this flag.
WITH_ACC ?= 0

ifeq ($(WITH_ACC),1)
    DEFAULT_TARGETS := device acc
    BUILD_TARGETS   := build_device build_acc
else
    DEFAULT_TARGETS := device
    BUILD_TARGETS   := build_device
endif

all: run

build: $(BUILD_TARGETS)

run: $(DEFAULT_TARGETS)

rerun:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile all

rebuild:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build

$(OUT):
	mkdir -p $(OUT)

# ============================================================================
# Library build via the customer-facing make/lib.mk -- same device LTO
# library that fortran_sum, examples/fortran, and other Fortran consumers
# use.  Built once with TARGET=device LINKAGE=lto so both the CUDA Fortran
# and OpenACC binaries link the same FPMP device objects.
# ============================================================================

$(DEV_LIB):
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/lib/make/lib.mk \
		TARGET=device LINKAGE=lto ARCH=$(ARCH) OUT=$(OUT) \
		EXTRA_FLAGS='$(LIB_EXTRA_FLAGS)' lib

# ============================================================================
# Device benchmark - uses nvfortran (NVIDIA HPC SDK)
# ============================================================================

ifneq ($(strip $(HAS_NVFORTRAN)),)

build_device: $(DEV_EXE)

$(DEV_EXE): $(DEV_SRC) $(DEV_LIB) | $(OUT)
	@echo "fortran_bs: building device benchmark (arch=sm_$(ARCH), LTO)..."
	mkdir -p $(OUT)/mod_dev
	nvfortran -cuda -gpu=lto,cc$(ARCH) -c \
		$(FORTRAN_DIR)/fpmp_api.f90 \
		-o $(OUT)/fpmp_api.o -module $(OUT)/mod_dev
	nvfortran -cpp $(DEV_CPP_FLAGS) -cuda -gpu=lto,cc$(ARCH) -O3 \
		-o $@ $(DEV_SRC) \
		$(OUT)/fpmp_api.o \
		-I$(OUT)/mod_dev -module $(OUT)/mod_dev \
		$(OUT)/obj/fpmp_lib$(DEVICE_SUFFIX).o $(OUT)/obj/fpemu_lib$(DEVICE_SUFFIX).o -lstdc++
	@echo "fortran_bs: $(notdir $@) built"

device: $(DEV_EXE)
	$(DEV_EXE)

else

build_device:
	@echo "fortran_bs: skipping device build - nvfortran not found in PATH (NVIDIA HPC SDK required)"

device:
	@echo "fortran_bs: skipping device run - nvfortran not found in PATH (NVIDIA HPC SDK required)"

endif

# ============================================================================
# OpenACC benchmark - uses nvfortran (NVIDIA HPC SDK)
# ============================================================================
# Same device LTO library as the CUDA Fortran `device` target.  Build
# flow is split into THREE steps:
#
#   1. Compile the unified bindings TU `fpmp_api.f90` with `-acc=gpu`
#      only (NO `-cuda`).  Keeps `!@cuf attributes(device) &` dormant
#      so every FPMP wrapper carries only `!$acc routine seq` -- not
#      the dual-attribute (`attributes(device)` + `!$acc routine
#      seq`) state that triggers an nvfortran ABI bug for derived-
#      type returns inside `!$acc parallel loop`.
#   2. Compile `fortran_bs_acc.f90` with `-acc=gpu` only (NO `-cuda`).
#      Same reason.  This also means `fortran_bs_acc.f90` cannot `use
#      cudafor` -- it relies on Fortran's intrinsic `system_clock`
#      for timing, with the `!$acc wait(0)` discipline before the
#      closing clock read so we measure real GPU time and not just
#      CPU dispatch latency.
#   3. Link with `-acc=gpu -cuda`.  `-cuda` IS required at link time
#      because the FPMP library was built by nvcc with `-dc -dlto`
#      and embeds CUDA runtime registration calls
#      (`__cudaRegisterLinkedBinary_*`, `__cudaRegisterManagedVar`,
#      `__cudaInitModule`, ...).  `-cuda` at link time wires
#      nvfortran into nvcc's device linker AND pulls in `libcudart`
#      so those symbols resolve.  `-cuda` at LINK time does not
#      re-process the source files, so `!@cuf` remains dormant in
#      the compiled bindings object.
#
# This way the OpenACC kernel body in `fortran_bs_acc.f90` stays
# end-user-friendly: natural `+`/`-`/`*`/`/`/`sqrt`/`log`/`exp`/`erfc`
# syntax on `type(fp32mp2)`, no raw subroutine plumbing.  The
# bindings object lands in `fpmp_api_acc.o` so it does not collide
# with the device-side `fpmp_api.o` above (different `-acc=gpu`
# flag).
#
# Companion benchmarks (`fortran_sum_acc`) still pass `-cuda` at
# COMPILE time because they `use cudafor` for `cudaEvent`-based
# batched timing; their kernels use `call fpmp_acc(...)`
# (subroutine, out parameter) and never trigger the by-value-return
# ABI path, so the dual-attribute state is safe there.  Only by-value
# derived-type returns in parallel regions need the pure-OpenACC
# compile flag combination.
#
# A SECOND nvfortran 25.11 bug is also sidestepped, but in the source
# rather than the build:  even under pure-OpenACC compilation,
# nvfortran miscompiles the "by-value derived-type return stored
# directly into an indexed global-array slot" ABI path -- every
# `%lo` field in the returned struct gets the `%hi` source value.
# `fortran_bs_acc.f90`'s `bs_kernel_fp32mp2` works around this by
# staging every `type(fp32mp2)` expression through a named private
# scalar declared in the `!$acc parallel loop private(...)` clause,
# so the only store into `prices_ff(:)` is a same-type intrinsic
# copy from a private scalar (`prices_ff(i) = price`), never a
# direct by-value function-return into the array.  Standalone
# reproducer: `../../nvfortran_bug_temp/expert_repro/` --
# `simple_loop.f90` FAILs, `simple_loop_via_priv.f90` PASSes.
#
# Inlining diagnostic: pass V=3 to enable nvfortran's `-Minfo=accel,inline`
# output (kernel schedule + every cross-procedure inline decision).  Watch
# for `Local memory used for <name>` -- those flag wrapper-call accumulator
# spills that drive a row much slower than the equivalent CUDA Fortran
# kernel.  Silenced by default to keep the build log clean.

ifneq ($(strip $(HAS_NVFORTRAN)),)

build_acc: $(ACC_EXE)

$(ACC_EXE): $(ACC_SRC) $(DEV_LIB) | $(OUT)
	@echo "fortran_bs: building OpenACC benchmark (arch=sm_$(ARCH), LTO)..."
	mkdir -p $(OUT)/mod_acc
	nvfortran -acc=gpu -gpu=cc$(ARCH),lto -O3 $(NVF_MINFO) -c \
		$(FORTRAN_DIR)/fpmp_api.f90 \
		-o $(OUT)/fpmp_api_acc.o -module $(OUT)/mod_acc
	nvfortran -cpp $(ACC_CPP_FLAGS) -acc=gpu -gpu=cc$(ARCH),lto -O3 \
		$(NVF_MINFO) -c \
		$(ACC_SRC) -o $(OUT)/fortran_bs_acc_main.o \
		-I$(OUT)/mod_acc -module $(OUT)/mod_acc
	nvfortran -acc=gpu -cuda -gpu=cc$(ARCH),lto -O3 $(NVF_MINFO) \
		-o $@ \
		$(OUT)/fortran_bs_acc_main.o $(OUT)/fpmp_api_acc.o \
		$(OUT)/obj/fpmp_lib$(DEVICE_SUFFIX).o $(OUT)/obj/fpemu_lib$(DEVICE_SUFFIX).o -lstdc++
	@echo "fortran_bs: $(notdir $@) built"

acc: $(ACC_EXE)
	$(ACC_EXE)

else

build_acc:
	@echo "fortran_bs: skipping acc build - nvfortran not found in PATH (NVIDIA HPC SDK required)"

acc:
	@echo "fortran_bs: skipping acc run - nvfortran not found in PATH (NVIDIA HPC SDK required)"

endif

clean:
	rm -rf $(OUT)

help:
	@echo "FPMP fortran_bs benchmark"
	@echo ""
	@echo "Usage: make [all|build|run|rerun|rebuild|device|acc|clean|help] [OPTIONS]"
	@echo ""
	@echo "Execution paths (both built from the same FPMP device LTO library):"
	@echo "  device  CUDA Fortran, attributes(global) kernels (nvfortran -cuda)"
	@echo "  acc     OpenACC, !\$$acc parallel loop kernels      (nvfortran -acc=gpu)"
	@echo ""
	@echo "  NOTE: 'acc' is excluded from 'all'/'run'/'build' by default because"
	@echo "        the natural overloaded-operator surface for type(fp32mp2)"
	@echo "        arithmetic inside an !\$$acc parallel loop is affected by"
	@echo "        known nvfortran 25.11 -acc=gpu codegen bugs (see"
	@echo "        issues/nvfortran_openacc_indexed_global_array_store_marshalling/"
	@echo "        and issues/nvfortran_openacc_chained_derived_type_return/)."
	@echo "        Pass WITH_ACC=1 to include 'acc' in the default chain, or"
	@echo "        invoke 'make acc' directly to exercise the failing path."
	@echo ""
	@echo "Options:"
	@echo "  WITH_ACC=<0|1>     include OpenACC 'acc' target in all/build/run"
	@echo "                     (default: 0; gated by upstream nvfortran 25.11"
	@echo "                      -acc=gpu codegen bugs filed under issues/)"
	@echo "  ARCH=<sm>          CUDA architecture (default: $(ARCH))"
	@echo "  OUT=<dir>          output directory (default: $(OUT))"
	@echo "  NUM_OPTIONS=<int>  options per kernel launch (default: $(NUM_OPTIONS))"
	@echo "  N_REPS=<int>       timing reps, fastest reported (default: $(N_REPS))"
	@echo "  BLOCK_SIZE=<int>   CUDA threads per block; also OpenACC vector_length"
	@echo "                     (default: $(BLOCK_SIZE))"
	@echo "  NUM_BLOCKS=<int>   CUDA grid size (default: $(NUM_BLOCKS))"
	@echo "  MIN_BATCH_MS_DEV=<int>"
	@echo "                     CUDA Fortran 'device' auto-tune target: minimum"
	@echo "                     milliseconds per timed cudaEvent window per kernel"
	@echo "                     (default: $(MIN_BATCH_MS_DEV) ms)"
	@echo "  MIN_BATCH_MS_ACC=<int>"
	@echo "                     OpenACC 'acc' auto-tune target: same meaning as"
	@echo "                     MIN_BATCH_MS_DEV, but for the OpenACC benchmark"
	@echo "                     (default: $(MIN_BATCH_MS_ACC) ms)"
	@echo "  MIN_BATCH_MS=<int> convenience override: when set, overrides BOTH"
	@echo "                     MIN_BATCH_MS_DEV and MIN_BATCH_MS_ACC at once."
	@echo "  KERNEL_LAUNCHES=<int>"
	@echo "                     GPU kernel launches per timed cudaEvent window;"
	@echo "                     0 = auto-tune to MIN_BATCH_MS_{DEV,ACC} per kernel,"
	@echo "                     >0 = fixed override (default: $(KERNEL_LAUNCHES))"
	@echo "  V=<int>            verbosity level (aliases: VERBOSE, VERB)"
	@echo "                     0 = silent (default), 1 = minimal, 2 = full,"
	@echo "                     3 = very verbose (also enables nvfortran"
	@echo "                     -Minfo=accel,inline diagnostics)"
	@echo ""
	@echo "Compiler availability (parse-time):"
	@echo "  HAS_NVFORTRAN = $(if $(HAS_NVFORTRAN),yes,no)"

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
