###########################################################################################
#
#  Makefile for FPMP `fortran_sum` Fortran benchmark (host + CUDA Fortran + OpenACC)
#
#  Compares accuracy and throughput of recursive summation across four
#  accumulator types - real(real32), real(real64), type(fp32mp2),
#  type(fp64mp2) - across three execution paths sharing the same
#  workload, reference, and FPMP library:
#
#    host    - pure Fortran (gfortran/nvfortran), two modes (serial + blocked)
#    device  - CUDA Fortran (nvfortran -cuda),  attributes(global) kernels
#    acc     - OpenACC      (nvfortran -acc=gpu), !$acc parallel loop kernels
#
#  See README.md for details.
#
#  Makefile parameters:
#  --------------------
#    `ARCH`        = CUDA sm_xx (unset: nvidia-smi, else 86)
#    `OUT`         = output directory (default: _out)
#    `FC`          = host Fortran compiler (default: gfortran)
#    `N_ELEMENTS`  = number of array elements (default: 10000000)
#    `N_REPS`      = timing repetitions, fastest is reported (default: 5)
#    `BLOCK_SIZE`  = CUDA threads per block (also used by host Mode 2 and
#                    by the OpenACC `vector_length`, default: 256)
#    `NUM_BLOCKS`  = CUDA grid size (also used by host Mode 2 and by the
#                    OpenACC `NTH = NUM_BLOCKS * BLOCK_SIZE`, default: 1024)
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
#    `FP2MP` / `MP2FP` = build the lib with the integer-only `double <-> fp32mp2`
#                        conversion paths (CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP /
#                        CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE).  In `fortran_sum`
#                        the only conversion-affected op is `fpmp_to_double`
#                        in the kernel epilogue (1 thread/block), so these
#                        are not expected to move throughput here -- see
#                        `benchmarks/d2ff/` for the isolated speedup numbers.
#    `VERBOSE`     = verbose mode (`0` (silent), `1` (minimal), `2` (full),
#                    `3` (very verbose)); aliases: `V`, `VERB`
#
#  Build targets:
#  --------------
#    `all`     - build and run host + device + acc benchmarks (default)
#    `build`   - build all three executables only
#    `host`    - build and run host benchmark
#    `device`  - build and run CUDA Fortran device benchmark
#    `acc`     - build and run OpenACC device benchmark
#    `run`     - same as `all`
#    `rerun`   - clean + build + run
#    `rebuild` - clean + build
#    `clean`   - remove generated files
#    `help`    - show this message
#
#  Compiler availability (CI-friendly auto-skip):
#  ----------------------------------------------
#    Fortran compilers are auto-detected at parse time via `command -v`:
#      - $(FC)     - host Fortran compiler (default: gfortran)
#      - nvfortran - CUDA Fortran / OpenACC compiler (NVIDIA HPC SDK)
#    If a compiler is missing, the corresponding targets become no-op stubs
#    that print a warning and exit successfully so the root `ci` target
#    stays green on environments without Fortran tooling.
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
# Force gfortran when FC is unset or only set by the make-builtin default
# (origin == "default", i.e. `FC = f77` from GNU make's implicit rules).
# Command-line / environment / per-invocation FC=... still wins.  We need
# to know which compiler we are talking to so we can pass the matching
# strict-IEEE flag (see HOST_STRICT_FP below) - the accuracy story this
# benchmark tells depends on strict left-to-right accumulation.
ifeq ($(origin FC),default)
    override FC := gfortran
endif
ifeq ($(origin FC),undefined)
    override FC := gfortran
endif
N_ELEMENTS      ?= 10000000
N_REPS          ?= 5
SCALE_BITS      ?= 0
BLOCK_SIZE      ?= 256
NUM_BLOCKS      ?= 1024
# GPU timing window controls.  Each timed cudaEvent window covers a
# per-kernel batch of launches sized to run for at least MIN_BATCH_MS_*
# milliseconds, so the measurement stays well above timer / launch-
# overhead noise.  The CUDA Fortran (`device`) and OpenACC (`acc`)
# targets need different defaults:
#
#   MIN_BATCH_MS_DEV = CUDA Fortran target (default: 500).
#     The device path's per-launch overhead is ~10 us, so even at 500 ms
#     per window the per-kernel launch count stays in the thousands and
#     the four kernels together complete in well under a minute of
#     wallclock.  500 ms keeps the run-to-run variance noticeably tighter
#     than 100 ms.
#
#   MIN_BATCH_MS_ACC = OpenACC target (default: 100).
#     The OpenACC path has ~1 ms of per-`!$acc parallel` overhead, so
#     even a 100 ms window covers ~100 launches at minimum and runs for
#     long enough to drown out cudaEvent timer noise.  Pushing this much
#     higher unnecessarily slows the benchmark on the slow rows
#     (fp64mp2 ~200 us each).
#
#   MIN_BATCH_MS = convenience override (no default).  When set on the
#     make command line, overrides BOTH MIN_BATCH_MS_DEV and
#     MIN_BATCH_MS_ACC at once.  Existing scripts that pass
#     `MIN_BATCH_MS=X` continue to work unchanged.
#
#   KERNEL_LAUNCHES = override.  0 = auto-tune to MIN_BATCH_MS_* per
#     kernel (default).  Set to a positive integer to disable auto-tuning
#     and use that fixed batch count instead.
MIN_BATCH_MS_DEV ?= 500
MIN_BATCH_MS_ACC ?= 100
ifneq ($(origin MIN_BATCH_MS),undefined)
    # User passed MIN_BATCH_MS on the command line or via the environment.
    MIN_BATCH_MS_DEV := $(MIN_BATCH_MS)
    MIN_BATCH_MS_ACC := $(MIN_BATCH_MS)
endif
KERNEL_LAUNCHES ?= 0

# Optional integer-only conversion paths (forwarded to the lib build via
# EXTRA_FLAGS).  These swap the FP64-pipe `double <-> fp32mp2` casts for
# integer bit manipulation -- relevant only when conversions appear in
# the hot path.  In `fortran_sum` the only hit is `fpmp_to_double` at
# the end of each block (1 thread/block), so they are ~free here; the
# `benchmarks/d2ff` micro-benchmark is the right place to see their
# isolated speedup.
#   FP2MP=1  -> -DCCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=1
#   MP2FP=1  -> -DCCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE=1
FP2MP ?= 0
MP2FP ?= 0

LIB_EXTRA_FLAGS :=
ifeq ($(FP2MP),1)
    LIB_EXTRA_FLAGS += -DCCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=1
endif
ifeq ($(MP2FP),1)
    LIB_EXTRA_FLAGS += -DCCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE=1
endif

# Compiler availability detection (parse-time, run once).
HAS_HOST_FC   := $(shell command -v $(FC) 2>/dev/null)
HAS_NVFORTRAN := $(shell command -v nvfortran 2>/dev/null)

DEVICE_SUFFIX ?=
HOST_SUFFIX   ?= _host

HOST_LIB    = $(OUT)/lib/lib$(LIB_BASE)$(HOST_SUFFIX).so
DEV_LIB     = $(OUT)/lib/lib$(LIB_BASE)$(DEVICE_SUFFIX).a
FORTRAN_DIR = $(MFP_DIR)/include/fortran

HOST_EXE = $(OUT)/fortran_sum_host
DEV_EXE  = $(OUT)/fortran_sum_device
ACC_EXE  = $(OUT)/fortran_sum_acc

HOST_SRC = $(THIS_DIR)/fortran_sum_host.f90
DEV_SRC  = $(THIS_DIR)/fortran_sum_device.cuf
ACC_SRC  = $(THIS_DIR)/fortran_sum_acc.f90

# Common -D flags that apply unchanged to every build target.  The host
# benchmark does not look at MIN_BATCH_MS / KERNEL_LAUNCHES at all
# (those are GPU-only), so it can use COMMON_CPP_FLAGS directly.  The
# two GPU paths extend it with their own MIN_BATCH_MS value.
COMMON_CPP_FLAGS = -DN_ELEMENTS=$(N_ELEMENTS) -DN_REPS=$(N_REPS) \
                   -DSCALE_BITS=$(SCALE_BITS) \
                   -DBLOCK_SIZE=$(BLOCK_SIZE) -DNUM_BLOCKS=$(NUM_BLOCKS) \
                   -DKERNEL_LAUNCHES=$(KERNEL_LAUNCHES)

# Per-target -DMIN_BATCH_MS so the CUDA Fortran and OpenACC benchmarks
# can pick window targets independently.  Both source files use the
# same `MIN_BATCH_MS` preprocessor name internally - the per-target
# value lands in the `-D` for that build's nvfortran invocation.
HOST_CPP_FLAGS = $(COMMON_CPP_FLAGS)
DEV_CPP_FLAGS  = $(COMMON_CPP_FLAGS) -DMIN_BATCH_MS=$(MIN_BATCH_MS_DEV)
ACC_CPP_FLAGS  = $(COMMON_CPP_FLAGS) -DMIN_BATCH_MS=$(MIN_BATCH_MS_ACC)

# Fortran module output directory flag (gfortran uses -J, nvfortran uses -module)
FC_MODDIR_FLAG = -J
ifneq (,$(findstring nvfortran,$(FC)))
    FC_MODDIR_FLAG = -module
endif
ifneq (,$(findstring ifort,$(FC)))
    FC_MODDIR_FLAG = -module
endif

# Strict IEEE / no-reassociation flags for the host benchmark.
#
#   The benchmark's accuracy story depends on the SP accumulator visibly
#   saturating around 2^24.  Compilers that reassociate FP reductions by
#   default (notably nvfortran -O3 / -fast) split the loop into independent
#   SIMD-lane partials, which masks the saturation and makes single
#   precision look bit-exact.  Lock the reduction down to source order.
#
#   Override with HOST_STRICT_FP="" to opt out (e.g. to compare un-pinned
#   throughput).
ifndef HOST_STRICT_FP
    ifneq (,$(findstring nvfortran,$(FC)))
        # PGI/NVIDIA HPC: -Kieee turns off FP reassociation and other
        # accuracy-affecting transforms; -Mnofma keeps FMA contraction off
        # so the SP loop sees true add-by-add rounding.
        HOST_STRICT_FP := -Kieee -Mnofma
    else ifneq (,$(filter-out ,$(findstring ifort,$(FC))$(findstring ifx,$(FC))))
        HOST_STRICT_FP := -fp-model precise -no-fma
    else
        # gfortran (and flang) preserve reduction order at -O3 by default;
        # this just makes the intent explicit and rejects -ffast-math
        # leaking in via env CFLAGS/FFLAGS.
        HOST_STRICT_FP := -fno-fast-math -fno-associative-math
    endif
endif

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
# inline decisions for the FPMP routines).  They produce ~750 lines of
# output per build for `build_acc` -- useful for spotting issues like
# `Local memory used for <name>` (FPMP wrapper spill) but noisy by
# default.  Gate them behind V=3 ("very verbose").
NVF_MINFO :=
ifeq ($(VERBOSE_MODE),3)
    NVF_MINFO := -Minfo=accel,inline
endif

.PHONY: all build run rerun rebuild host device acc clean help

# OpenACC opt-in switch.  By default the fortran_sum OpenACC benchmark
# is excluded from `all`/`build`/`run` because the natural overloaded-
# operator surface for `type(fp32mp2)` / `type(fp64mp2)` arithmetic is
# affected by known nvfortran 25.11 -acc=gpu codegen bugs (see the
# bug reports at issues/nvfortran_openacc_indexed_global_array_store_marshalling/
# and issues/nvfortran_openacc_chained_derived_type_return/).  Pass
# `WITH_ACC=1` on the make command line to include `acc` in the
# default chain.  The `acc` target itself remains available for
# explicit invocation regardless of this flag.
WITH_ACC ?= 0

ifeq ($(WITH_ACC),1)
    DEFAULT_TARGETS := host device acc
    BUILD_TARGETS   := build_host build_device build_acc
else
    DEFAULT_TARGETS := host device
    BUILD_TARGETS   := build_host build_device
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
# Library builds via the customer-facing make/lib.mk
# ============================================================================

$(HOST_LIB):
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/lib/make/lib.mk \
		TARGET=host OUT=$(OUT) \
		EXTRA_FLAGS='$(LIB_EXTRA_FLAGS)' lib

$(DEV_LIB):
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/lib/make/lib.mk \
		TARGET=device LINKAGE=lto ARCH=$(ARCH) OUT=$(OUT) \
		EXTRA_FLAGS='$(LIB_EXTRA_FLAGS)' lib

# ============================================================================
# Host benchmark - uses gfortran/$(FC)
# ============================================================================

ifneq ($(strip $(HAS_HOST_FC)),)

build_host: $(HOST_EXE)

$(HOST_EXE): $(HOST_SRC) $(HOST_LIB) | $(OUT)
	@echo "fortran_sum: building host benchmark with $(FC) ($(HOST_STRICT_FP))..."
	$(FC) -c $(FORTRAN_DIR)/fpmp_api.f90 \
		-o $(OUT)/fpmp_api_host.o \
		$(FC_MODDIR_FLAG) $(OUT)
	$(FC) -cpp $(HOST_CPP_FLAGS) -O3 $(HOST_STRICT_FP) -o $@ $(HOST_SRC) \
		$(OUT)/fpmp_api_host.o \
		-I$(OUT) $(HOST_LIB)
	@echo "fortran_sum: $(notdir $@) built"

host: $(HOST_EXE)
	LD_LIBRARY_PATH=$(OUT)/lib:$$LD_LIBRARY_PATH $(HOST_EXE)

else

build_host:
	@echo "fortran_sum: skipping host build - Fortran compiler '$(FC)' not found in PATH"

host:
	@echo "fortran_sum: skipping host run - Fortran compiler '$(FC)' not found in PATH"

endif

# ============================================================================
# Device benchmark - uses nvfortran (NVIDIA HPC SDK)
# ============================================================================

ifneq ($(strip $(HAS_NVFORTRAN)),)

build_device: $(DEV_EXE)

$(DEV_EXE): $(DEV_SRC) $(DEV_LIB) | $(OUT)
	@echo "fortran_sum: building device benchmark (arch=sm_$(ARCH), LTO)..."
	mkdir -p $(OUT)/mod_dev
	nvfortran -cuda -gpu=lto,cc$(ARCH) -c \
		$(FORTRAN_DIR)/fpmp_api.f90 \
		-o $(OUT)/fpmp_api_dev.o -module $(OUT)/mod_dev
	nvfortran -cpp $(DEV_CPP_FLAGS) -cuda -gpu=lto,cc$(ARCH) -O3 \
		-o $@ $(DEV_SRC) \
		$(OUT)/fpmp_api_dev.o \
		-I$(OUT)/mod_dev -module $(OUT)/mod_dev \
		$(OUT)/obj/fpmp_lib$(DEVICE_SUFFIX).o $(OUT)/obj/fpemu_lib$(DEVICE_SUFFIX).o -lstdc++
	@echo "fortran_sum: $(notdir $@) built"

device: $(DEV_EXE)
	$(DEV_EXE)

else

build_device:
	@echo "fortran_sum: skipping device build - nvfortran not found in PATH (NVIDIA HPC SDK required)"

device:
	@echo "fortran_sum: skipping device run - nvfortran not found in PATH (NVIDIA HPC SDK required)"

endif

# ============================================================================
# OpenACC benchmark - uses nvfortran (NVIDIA HPC SDK)
# ============================================================================
# Build flow: this is a PURE OpenACC benchmark.  The main program does
# NOT `use cudafor`; per-launch batched timing uses `system_clock` (an
# int64 Fortran intrinsic with nanosecond resolution on Linux), with
# `!$acc wait(0)` between the launch loop and the closing clock read
# so the elapsed-time interval brackets pure GPU time -- see the file
# header in `fortran_sum_acc.f90` for the full rationale.
#
# Because the source does not need `-cuda`, both the bindings and the
# main program are compiled with `-acc=gpu -gpu=cc$(ARCH),lto` ONLY.
# Withholding `-cuda` at compile time keeps the `!@cuf
# attributes(device) &` directives inside the unified `fpmp_api.f90`
# dormant -- every FPMP wrapper carries only `!$acc routine seq` and
# stays host-callable, so the host-side `fpmp_to_double(partials_*(j))`
# folds after `!$acc update self` resolve cleanly (passing `-cuda`
# would activate `!@cuf`, give every specific procedure
# `attributes(device)`, and trigger
# `NVFORTRAN-S-0155-Could not resolve generic procedure
# fpmp_to_double` on the host).
#
# `-cuda` IS passed at the final link step.  The FPMP library objects
# (`nvcc -dc -dlto`) embed CUDA runtime registration calls that need
# `libcudart`; `-cuda` wires nvfortran into nvcc's device linker so
# `libcudart` is pulled in.  `-cuda` at link time does not re-process
# the already-compiled `.o` files, so `!@cuf` remains dormant.
#
# `-gpu=cc$(ARCH),lto` performs cross-module device LTO across the
# OpenACC kernel and the FPMP runtime, inlining FPMP arithmetic into
# the kernel body wherever possible.  `-acc=gpu` enables offload for
# the `!$acc` directives in the source.  The bindings object lands in
# a separate `mod_acc/` module-output directory so it does not collide
# with the gfortran-compiled `fpmp_api_host.o` from the host rules
# (build-variant labels, not module names: the unified `fpmp_api.f90`
# defines a single `module fpmp_api`).
#
# Inlining diagnostic:
#   `-Minfo=accel,inline` prints (a) the OpenACC kernel schedule and (b) every
#   cross-procedure inline decision.  Watch for `Local memory used for <name>`
#   lines under each `reduce_*_kernel` - those flag the wrapper-call
#   address-of-accumulator spill that drives the `fp32mp2` row much slower
#   than the equivalent CUDA Fortran kernel.  These messages are silenced
#   by default (they generate ~750 lines per build through the FPMP
#   bindings); pass `V=3` on the make command line to re-enable them.

ifneq ($(strip $(HAS_NVFORTRAN)),)

build_acc: $(ACC_EXE)

$(ACC_EXE): $(ACC_SRC) $(DEV_LIB) | $(OUT)
	@echo "fortran_sum: building OpenACC benchmark (arch=sm_$(ARCH), LTO)..."
	mkdir -p $(OUT)/mod_acc
	nvfortran -acc=gpu -gpu=cc$(ARCH),lto -O3 $(NVF_MINFO) -c \
		$(FORTRAN_DIR)/fpmp_api.f90 \
		-o $(OUT)/fpmp_api_acc.o -module $(OUT)/mod_acc
	nvfortran -cpp $(ACC_CPP_FLAGS) -acc=gpu -gpu=cc$(ARCH),lto -O3 \
		$(NVF_MINFO) -c \
		-o $(OUT)/fortran_sum_acc.o $(ACC_SRC) \
		-I$(OUT)/mod_acc -module $(OUT)/mod_acc
	nvfortran -acc=gpu -cuda -gpu=cc$(ARCH),lto -O3 \
		-o $@ \
		$(OUT)/fortran_sum_acc.o \
		$(OUT)/fpmp_api_acc.o \
		$(OUT)/obj/fpmp_lib$(DEVICE_SUFFIX).o $(OUT)/obj/fpemu_lib$(DEVICE_SUFFIX).o -lstdc++
	@echo "fortran_sum: $(notdir $@) built"

acc: $(ACC_EXE)
	$(ACC_EXE)

else

build_acc:
	@echo "fortran_sum: skipping acc build - nvfortran not found in PATH (NVIDIA HPC SDK required)"

acc:
	@echo "fortran_sum: skipping acc run - nvfortran not found in PATH (NVIDIA HPC SDK required)"

endif

clean:
	rm -rf $(OUT)

help:
	@echo "FPMP fortran_sum benchmark"
	@echo ""
	@echo "Usage: make [all|build|run|rerun|rebuild|host|device|acc|clean|help] [OPTIONS]"
	@echo ""
	@echo "Execution paths (built from the same FPMP device LTO library):"
	@echo "  host    pure Fortran (gfortran/nvfortran), Mode 1 + Mode 2"
	@echo "  device  CUDA Fortran, attributes(global) kernels (nvfortran -cuda)"
	@echo "  acc     OpenACC, !\$acc parallel loop kernels (nvfortran -acc=gpu)"
	@echo ""
	@echo "  NOTE: 'acc' is excluded from 'all'/'run'/'build' by default because"
	@echo "        the natural overloaded-operator surface for type(fp32mp2)/"
	@echo "        type(fp64mp2) arithmetic is affected by known nvfortran"
	@echo "        25.11 -acc=gpu codegen bugs (see"
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
	@echo "  FC=<compiler>      host Fortran compiler (default: $(FC))"
	@echo "  N_ELEMENTS=<int>   array size (default: $(N_ELEMENTS))"
	@echo "  N_REPS=<int>       timing reps, fastest reported (default: $(N_REPS))"
	@echo "  SCALE_BITS=<int>   max input ≈ 2^SCALE_BITS (default: $(SCALE_BITS))"
	@echo "  BLOCK_SIZE=<int>   CUDA threads per block; also host Mode 2 and"
	@echo "                     OpenACC vector_length (default: $(BLOCK_SIZE))"
	@echo "  NUM_BLOCKS=<int>   CUDA grid size; also host Mode 2 and OpenACC NTH"
	@echo "                     (default: $(NUM_BLOCKS))"
	@echo "  MIN_BATCH_MS_DEV=<int>"
	@echo "                     CUDA Fortran 'device' auto-tune target: minimum"
	@echo "                     milliseconds per timed cudaEvent window per kernel;"
	@echo "                     auto-tuning picks the per-window launch count so each"
	@echo "                     measurement runs at least this long (default: $(MIN_BATCH_MS_DEV) ms)"
	@echo "  MIN_BATCH_MS_ACC=<int>"
	@echo "                     OpenACC 'acc' auto-tune target: same meaning as"
	@echo "                     MIN_BATCH_MS_DEV, but for the OpenACC benchmark."
	@echo "                     Kept shorter because OpenACC has ~1 ms per-launch"
	@echo "                     overhead (default: $(MIN_BATCH_MS_ACC) ms)"
	@echo "  MIN_BATCH_MS=<int> convenience override: when set, overrides BOTH"
	@echo "                     MIN_BATCH_MS_DEV and MIN_BATCH_MS_ACC at once."
	@echo "                     (unset by default; pre-existing scripts still work)"
	@echo "  KERNEL_LAUNCHES=<int>"
	@echo "                     GPU kernel launches per timed cudaEvent window;"
	@echo "                     0 = auto-tune to MIN_BATCH_MS_{DEV,ACC} per kernel,"
	@echo "                     >0 = fixed override (default: $(KERNEL_LAUNCHES))."
	@echo "                     Applies to both 'device' and 'acc'; per-launch time"
	@echo "                     = elapsed / launches"
	@echo "  HOST_STRICT_FP=<flags>   override the auto-picked strict-IEEE flags"
	@echo "                           for the host build.  Default for $(FC):"
	@echo "                             $(HOST_STRICT_FP)"
	@echo "                           Pass HOST_STRICT_FP= (empty) to opt out and"
	@echo "                           let the compiler reassociate the SP loop."
	@echo "  FP2MP=1            build the lib with CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=1"
	@echo "                     (integer-only double->fp32mp2 conversion)"
	@echo "  MP2FP=1            build the lib with CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE=1"
	@echo "                     (integer-only fp32mp2->double conversion)"
	@echo "                     Note: only fpmp_to_double in the kernel epilogue"
	@echo "                     uses these paths -- expected to be a no-op on"
	@echo "                     fortran_sum throughput.  See benchmarks/d2ff/"
	@echo "                     for the isolated conversion-throughput benchmark."
	@echo "  V=<int>            verbosity level (aliases: VERBOSE, VERB)"
	@echo "                     0 = silent (default), 1 = minimal, 2 = full,"
	@echo "                     3 = very verbose (also enables nvfortran"
	@echo "                     -Minfo=accel,inline diagnostics for the 'acc' build)"
	@echo ""
	@echo "Compiler availability (parse-time):"
	@echo "  HAS_HOST_FC   = $(if $(HAS_HOST_FC),yes,no)"
	@echo "  HAS_NVFORTRAN = $(if $(HAS_NVFORTRAN),yes,no)"

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
