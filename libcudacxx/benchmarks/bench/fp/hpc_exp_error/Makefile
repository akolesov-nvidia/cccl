###############################################################################
#  Makefile - composite-exp() error & GPU performance benchmark
#  -------------------------------------------------------------
#  Layout & conventions:
#    * default TARGET=device (CPU build via TARGET=host)
#    * all build artefacts live in $(OUT) (default: _out)
#    * targets: all, build, run, rebuild, rerun, clean, help
#
#  Pointing at the cuda_multi_fp headers
#  -------------------------------------
#  This benchmark needs the cuda_multi_fp headers (fpmp.hpp, fpmp_math.hpp).
#  cuda_multi_fp is header-only for the templates we use here, so nothing
#  needs to be linked from it. Two ways to point at a checkout:
#
#    make CUDA_MULTI_FP_DIR=/path/to/cuda_multi_fp rerun        # preferred
#    make FPMP_INC=/path/to/cuda_multi_fp/include   rerun       # lower-level
#
#  When neither is set the Makefile falls back to ../../include relative to
#  its own location (this matches the layout when the benchmark is shipped
#  inside a cuda_multi_fp checkout). If the resolved include path doesn't
#  contain fpmp.hpp the build aborts immediately with a one-line hint.
#
#  Performance-measurement knobs (compile-time defaults, all overridable):
#    NUM_OPTIONS        = chains per launch (default: 10000 GPU / 1000 host)
#    CHAIN_LEN          = exponents per chain (default: 10000 GPU / 1000 host)
#    REPS               = inner repetitions inside ONE kernel launch
#                         (default: 8 GPU / 2 host) - amortises launch overhead
#    NUM_ITER           = number of timed launches averaged (default: 10)
#    THREADS            = CUDA threads per block (default: 256)
#    NUM_BLOCKS         = CUDA blocks (default: 2048)
#    SWEEP_NUM_DEPTHS   = depth checkpoints in the alpha sweep (default: 20)
#    UNROLL             = per-thread unroll factor / # of independent
#                         accumulators in compose / expsum (default: 4)
#
#  Usage:
#    make CUDA_MULTI_FP_DIR=<path>                        # build + run on GPU
#    make CUDA_MULTI_FP_DIR=<path> TARGET=host            # build + run on CPU
#    make CUDA_MULTI_FP_DIR=<path> REPS=32 NUM_ITER=20 rerun
#    make CUDA_MULTI_FP_DIR=<path> NUM_OPTIONS=4194304 rerun
#    make ARCH=89                                          # explicit CUDA arch
#    make OUT=/tmp/foo run                                 # custom output dir
###############################################################################

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

TARGET   ?= device
CXX      ?= g++
# Float128 reference is a GCC + libquadmath feature; always built with g++.
REF_CXX  ?= g++
NVCC     ?= nvcc
OPT      ?= -O3
WARN     ?= -Wall -Wextra -Wpedantic
CXXSTD   ?= -std=c++17
# Resolve cuda_multi_fp headers. Two ways for the user to point the build at
# an out-of-tree checkout:
#   * CUDA_MULTI_FP_DIR=<root of cuda_multi_fp checkout>  (preferred)
#   * FPMP_INC=<path that already contains fpmp.hpp>      (lower-level override)
# When neither is set, fall back to the in-tree default (this Makefile lives
# at $(CUDA_MULTI_FP)/benchmarks/hpc_exp_error/, so headers are at ../../include).
ifdef CUDA_MULTI_FP_DIR
    FPMP_INC ?= $(CUDA_MULTI_FP_DIR)/include
else
    FPMP_INC ?= $(THIS_DIR)/../../include
endif
OUT      ?= $(THIS_DIR)/_out

# Validate that fpmp.hpp is actually findable at $(FPMP_INC). The check runs
# only for build-y targets (skip help/clean so users can always inspect/clean).
ifeq (,$(filter help clean,$(MAKECMDGOALS)))
ifeq (,$(wildcard $(FPMP_INC)/fpmp.hpp))
$(error cuda_multi_fp headers not found at FPMP_INC=$(FPMP_INC). \
Pass CUDA_MULTI_FP_DIR=<path to cuda_multi_fp checkout> or FPMP_INC=<path to its include dir>. \
Example: make CUDA_MULTI_FP_DIR=$$HOME/src/cuda_multi_fp rerun)
endif
endif

# --- Performance-measurement defaults ---------------------------------------
# Host and GPU share the same problem-size defaults so that the host run on
# its own (`make TARGET=host`) reports maxRelErr / alpha at exactly the same
# (numSamples, chainLen) as the GPU run, making the two reports directly
# comparable. Defaults match the original problem statement this benchmark
# was derived from:
#     numSamples = 10000, chainLen = 10000, depthStep = 500 (-> 20 checkpoints)
#
# At 10000 x 10000 device memory totals ~3.6 GB across float / double /
# fp32mp2 / fp128-ref buffers, which fits any modern GPU comfortably. The
# host build is OMP-parallel across independent chains and finishes the
# accuracy pass in a few seconds at this size; if you want a faster host
# smoke test, override on the command line, e.g.
#     make TARGET=host NUM_OPTIONS=1000 CHAIN_LEN=1000
NUM_OPTIONS ?= 10000
CHAIN_LEN   ?= 10000
ifneq (,$(filter $(T) $(TARGET), host))
    REPS    ?= 2
else
    REPS    ?= 8
endif
NUM_ITER         ?= 10
THREADS          ?= 256
NUM_BLOCKS       ?= 2048
SWEEP_NUM_DEPTHS ?= 20
# Per-thread unroll factor for compose / expsum kernels. Each iteration of
# the unrolled inner loop runs UNROLL independent partial accumulators,
# which breaks the serial latency chain through the accumulator and lifts
# fp throughput on Ada/Hopper-class GPUs. Setting UNROLL=1 disables ILP
# (single accumulator, naive sequential loop). 4 is a good default; on
# higher-occupancy hardware UNROLL=8 occasionally helps.
UNROLL           ?= 4

# Compile-time defines exposed to the C++ source (overridable on CLI).
BENCH_DEFINES = \
    -DNUM_OPTIONS="$(NUM_OPTIONS)" \
    -DCHAIN_LEN=$(CHAIN_LEN) \
    -DREPS=$(REPS) \
    -DNUM_ITERATIONS=$(NUM_ITER) \
    -DTHREADS_PER_BLOCK=$(THREADS) \
    -DNUM_BLOCKS=$(NUM_BLOCKS) \
    -DSWEEP_NUM_DEPTHS=$(SWEEP_NUM_DEPTHS) \
    -DHPC_UNROLL_FACTOR=$(UNROLL)

# Auto-detect CUDA arch when needed (digits-only filter; nvidia-smi can print
# error text on stdout when the driver is missing). "Device" includes any
# TARGET that isn't `host`. The top-level Makefile of an enclosing tree may
# pass TARGET=all by default, which must still build the CUDA TU.
ifeq (,$(filter $(T) $(TARGET), host))
    ifndef ARCH
        _ARCH_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -dc '0-9')
        ifeq ($(shell echo "$(_ARCH_CAP)" | grep -E '^[0-9]{2,3}$$'),)
            ARCH := 86
        else
            ARCH := $(_ARCH_CAP)
        endif
    endif
endif

EXE := $(OUT)/hpc_exp_error

# Verbose knob (V / VERBOSE / VERB; values 0..3).
VERBOSE_MODE = 1
ifneq (,$(filter 0 n N no No,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 0
else ifneq (,$(filter 1 y Y yes Yes,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 1
else ifneq (,$(filter 2,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 2
else ifneq (,$(filter 3 full Full f F,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 3
endif

ifeq ($(VERBOSE_MODE),1)
    RULE_DONE = -@echo "hpc_exp_error: $(notdir $@) built"
else
    RULE_DONE =
endif

ifeq ($(VERBOSE_MODE),0)
    Q = @
endif

# Position-independent code: required because nvcc/-cuda link a PIE binary on
# modern Linux distros and reject non-PIC host objects.
PIC_FLAG ?= -fPIC

# OpenMP for the host accuracy snapshot (TARGET=device flow).
# Compiler-specific flag: g++/clang++ use -fopenmp, nvc++ uses -mp. Set OMP=0
# to disable (host snapshot then runs single-threaded; pragmas degrade silently).
OMP ?= 1
ifeq ($(OMP),1)
    ifneq (,$(findstring nvc++,$(notdir $(CXX))))
        OMP_FLAG := -mp
    else
        OMP_FLAG := -fopenmp
    endif
else
    OMP_FLAG :=
endif

HOST_CXXFLAGS := $(CXXSTD) $(OPT) $(WARN) $(PIC_FLAG) $(OMP_FLAG) -I$(THIS_DIR) -I$(FPMP_INC) $(BENCH_DEFINES)
REF_CXXFLAGS  := $(CXXSTD) $(OPT) $(PIC_FLAG) -I$(THIS_DIR) -I$(FPMP_INC) -fext-numeric-literals

OBJS_HOST := $(OUT)/hpc_exp_error_host.o $(OUT)/main_exp_error.o
OBJS_REF  := $(OUT)/hpc_exp_error_ref.o
OBJS_GPU  :=
LDLIBS    := -lquadmath

ifeq (,$(filter $(T) $(TARGET), host))
    OBJS_GPU      += $(OUT)/hpc_exp_error_gpu.o
    HOST_CXXFLAGS += -DHPC_EXP_ERROR_HAVE_CUDA=1
    REF_CXXFLAGS  += -DHPC_EXP_ERROR_HAVE_CUDA=1

    NVCCFLAGS     ?= $(CXXSTD) $(OPT) -arch=sm_$(ARCH) -x cu --extended-lambda \
                      -I$(THIS_DIR) -I$(FPMP_INC) -DHPC_EXP_ERROR_HAVE_CUDA=1 $(BENCH_DEFINES)

    # Pick a linker driver that matches the compiler that produced the host
    # objects, otherwise its proprietary runtime symbols (vector math like
    # __fs_exp_8_avx2 for nvc++, libstdc++ ABI specifics for clang++, etc.)
    # won't be resolved.
    #
    # OpenMP runtime must also be linked: nvc++ needs -mp at link time;
    # nvcc-as-linker forwards -Xcompiler -fopenmp so libgomp gets pulled in.
    ifneq (,$(findstring nvc++,$(notdir $(CXX))))
        LINK ?= $(CXX) $(CXXSTD) $(OPT) $(PIC_FLAG) -cuda -gpu=cc$(ARCH) $(OMP_FLAG)
    else
        ifeq ($(OMP),1)
            LINK ?= $(NVCC) $(CXXSTD) -arch=sm_$(ARCH) $(OPT) -Xcompiler -fopenmp
        else
            LINK ?= $(NVCC) $(CXXSTD) -arch=sm_$(ARCH) $(OPT)
        endif
    endif
else
    LINK ?= $(CXX) $(CXXSTD) $(OPT) $(OMP_FLAG)
endif

OBJS := $(OBJS_HOST) $(OBJS_REF) $(OBJS_GPU)

.PHONY: all build run rebuild rerun clean help

# Default target: build + run.
all: run

build: $(EXE)

$(EXE): $(OBJS)
	$(Q)$(LINK) -o $@ $^ $(LDLIBS)
	$(RULE_DONE)

# Generic host TUs.
$(OUT)/hpc_exp_error_host.o: $(THIS_DIR)/hpc_exp_error_host.cpp $(THIS_DIR)/hpc_exp_error.h $(THIS_DIR)/hpc_exp_error_kernels.hpp | $(OUT)
	$(Q)$(CXX) $(HOST_CXXFLAGS) -c -o $@ $<
	$(RULE_DONE)

$(OUT)/main_exp_error.o: $(THIS_DIR)/main_exp_error.cpp $(THIS_DIR)/hpc_exp_error.h | $(OUT)
	$(Q)$(CXX) $(HOST_CXXFLAGS) -c -o $@ $<
	$(RULE_DONE)

# Float128 reference: always g++ (libquadmath, __float128, -fext-numeric-literals).
$(OUT)/hpc_exp_error_ref.o: $(THIS_DIR)/hpc_exp_error_ref.cpp $(THIS_DIR)/hpc_exp_error.h $(THIS_DIR)/hpc_exp_error_kernels.hpp | $(OUT)
	$(Q)$(REF_CXX) $(REF_CXXFLAGS) -c -o $@ $<
	$(RULE_DONE)

# CUDA TU.
$(OUT)/hpc_exp_error_gpu.o: $(THIS_DIR)/hpc_exp_error_gpu.cu $(THIS_DIR)/hpc_exp_error.h $(THIS_DIR)/hpc_exp_error_kernels.hpp | $(OUT)
	$(Q)$(NVCC) $(NVCCFLAGS) -c -o $@ $<
	$(RULE_DONE)

$(OUT):
	$(Q)mkdir -p $(OUT)

run: $(EXE)
	$(Q)$(EXE) 2>&1

rebuild:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build

rerun:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build
	MAKEFLAGS= $(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile run -j1

clean:
	$(Q)rm -rf $(OUT)

help:
	@echo "Composite-exp() error & GPU performance benchmark"
	@echo "==================================================="
	@echo ""
	@echo "Build parameters:"
	@echo "  TARGET     = device|host                (default: device)"
	@echo "  ARCH       = CUDA SM (auto-detected via nvidia-smi, fallback 86)"
	@echo "  CXX        = host C++ compiler          (default: g++)"
	@echo "  REF_CXX    = compiler for fp128 ref TU  (default: g++; pinned)"
	@echo "  NVCC       = CUDA compiler              (default: nvcc)"
	@echo "  OPT        = optimisation flag          (default: -O3)"
	@echo "  OUT        = output directory           (default: _out)"
	@echo "  CUDA_MULTI_FP_DIR = root of an external cuda_multi_fp checkout"
	@echo "                                          (use when this benchmark is built standalone)"
	@echo "  FPMP_INC   = direct override for the include path"
	@echo "                                          (default: \$$(CUDA_MULTI_FP_DIR)/include or"
	@echo "                                                    \$$(THIS_DIR)/../../include)"
	@echo "  VERBOSE    = 0|1|2|3                    (default: 1)"
	@echo ""
	@echo "Performance-measurement parameters (compile-time defaults):"
	@echo "  NUM_OPTIONS      = chains per launch     (default: 10000 GPU / 1000 host)"
	@echo "  CHAIN_LEN        = exponents per chain   (default: 10000 GPU / 1000 host)"
	@echo "  REPS             = inner reps per launch (default: 8 GPU / 2 host)"
	@echo "  NUM_ITER         = timed launches        (default: 10)"
	@echo "  THREADS          = CUDA threads/block    (default: 256)"
	@echo "  NUM_BLOCKS       = CUDA grid blocks      (default: 2048)"
	@echo "  SWEEP_NUM_DEPTHS = alpha-sweep depths    (default: 20)"
	@echo "  UNROLL           = per-thread unroll factor in compose / expsum"
	@echo "                                            (default: 4; UNROLL=1 disables ILP)"
	@echo ""
	@echo "GPU defaults (10000 x 10000) match the original problem statement"
	@echo "this benchmark was derived from."
	@echo "Memory footprint is ~3.6 GB; fits any modern GPU. Push higher to"
	@echo "saturate larger GPUs (e.g. NUM_OPTIONS=100000 CHAIN_LEN=10000)."
	@echo ""
	@echo "Targets:"
	@echo "  all      - build + run (default)"
	@echo "  build    - build only"
	@echo "  run      - run (builds if needed)"
	@echo "  rebuild  - clean + build"
	@echo "  rerun    - clean + build + run"
	@echo "  clean    - remove $(OUT)"
	@echo "  help     - show this message"

# silent mode
ifneq (,$(filter $(VERBOSE_MODE),0 1))
.SILENT:
endif
