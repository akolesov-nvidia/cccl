###########################################################################################
#
#  File:    Makefile
#  Purpose: Build and run mixed precision throughput benchmarks
#  Author:  Andrei Kolesov
#  Date:    2025
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `TARGET`      = target device (`device` (default), `host`)
#    `ARCH`        = CUDA architecture (default: 86 for RTX 30xx/A100)
#    `NUM_ITER`    = number of timing iterations (default: 10)
#    `THREADS`     = threads per block for CUDA (default: 256)
#    `NUM_BLOCKS`  = number of CUDA blocks (default: 1024)
#    `REPS`        = operations per thread (default: 1024)
#    `UNROLL`      = loop unroll factor (default: 64)
#    `OUT`         = output directory (default: _out)
#    `VERBOSE`     = verbose mode (`0` (silent), `1` (minimal), `2` (full), `3` (very verbose)); aliases: `V`, `VERB`
#    `LINKAGE`     = linkage mode (`inline` (default), `static`, `lto`)
#    `TYPE`       = data types to benchmark (space-separated list or `all`)
#                    Options: fp32, fp32-fp32, fp64, fpemu_acc, fpemu_def, fpemu_fast, fp64-fp64, fp128
#                    Default: fp32 fp32-fp32 fp64 fp64-fp64
#
#  Build targets:
#  --------------
#    `all` (default) - build and run benchmark
#    `build`         - build benchmark executable only
#    `run`           - run the benchmark (builds if needed)
#    `rebuild`       - clean and build
#    `rerun`         - clean, build, and run
#    `clean`         - delete generated files
#    `help`          - show help message
#
#  Usage examples:
#  ---------------
#    `make`                              - build and run on GPU
#    `make ARCH=89 run`                  - run on Ada Lovelace (RTX 40xx)
#    `make ARCH=100 run`                 - run on Blackwell with fp128 support
#    `make REPS=2048 run`                - run with 2048 ops per thread
#    `make NUM_BLOCKS=2048 THREADS=512`  - run with more threads
#    `make VERBOSE=3 rebuild`            - rebuild with full verbose output
#    `make clean`                        - clean up generated files
#    `make TYPE=all run`                - benchmark all data types
#    `make TYPE="fp32 fp64" run`        - benchmark only fp32 and fp64
#    `make TYPE="fpemu_acc fpemu_def fpemu_fast" run` - benchmark FPEMU types
#
#  Dependencies:
#  -------------
#    - cuda_multi_fp library (fpmp.h) for float-float and double-double
#    - fpemu library (fpemu.h) for emulated FP64 types
#
###########################################################################################

# Default parameters
TARGET      ?= device
# CUDA sm_xx: use ARCH if set; else first GPU (nvidia-smi); else 86.
ifndef ARCH
    _fpmp_cuda_arch_cap := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
    ifeq ($(strip $(_fpmp_cuda_arch_cap)),)
        ARCH := 86
    else
        ARCH := $(_fpmp_cuda_arch_cap)
    endif
endif
DEVICE_SUFFIX ?=
HOST_SUFFIX   ?= _host
NUM_ITER    ?= 10
THREADS     ?= 256
NUM_BLOCKS  ?= 8192
REPS        ?= 4096
UNROLL      ?= 64

# Current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Dev-only library build (lib/make/lib.mk + lib/src/ + lib/include/) lives in the sibling cccl-fpdev
# repo; overridable (the cccl-fpdev root Makefile passes it explicitly).
FPDEV_DIR ?= $(abspath $(THIS_DIR)/../../../../../../cccl-fpdev)

# libquadmath availability — only x86 GCC distributions ship it.
# Override with HAS_QUADMATH=1/0 to bypass auto-detection.
HOST_ARCH := $(shell uname -m 2>/dev/null)
ifndef HAS_QUADMATH
    ifneq (,$(filter $(HOST_ARCH),x86_64 i386 i686 amd64))
        HAS_QUADMATH := 1
    else
        HAS_QUADMATH := 0
    endif
endif
ifeq ($(HAS_QUADMATH),1)
    QUADMATH_LIB := -lquadmath
else
    QUADMATH_LIB :=
endif

# Custom ABI: ABI=x,y (forces LINKAGE=static)
COMMA := ,
ifdef ABI
    ABI_DATA    := $(word 1,$(subst $(COMMA), ,$(ABI)))
    ABI_CONTROL := $(word 2,$(subst $(COMMA), ,$(ABI)))
    LINKAGE := static
    ABI_FLAGS = -D_CCCL_FPMP_ABI_PRESERVE_N_DATA=$(ABI_DATA) -D_CCCL_FPMP_ABI_PRESERVE_N_CONTROL=$(ABI_CONTROL) \
                -D_CCCL_FPEMU_ABI_PRESERVE_N_DATA=$(ABI_DATA) -D_CCCL_FPEMU_ABI_PRESERVE_N_CONTROL=$(ABI_CONTROL)
else
    ABI_FLAGS =
endif

# Source and include directories
INCLUDE_DIR = $(THIS_DIR)/../../../../include
SRC_FILE    = $(THIS_DIR)/mixed.cpp
MFP_DIR     = $(THIS_DIR)/../..

LINKAGE ?= inline

# Output directory
OUT ?= $(THIS_DIR)/_out

# Executable name
EXE = $(OUT)/mixed.exe
OBJ = $(OUT)/mixed.o


# Compiler settings
ifneq (,$(filter $(T) $(TARGET), host))
    TARGET_SUFFIX = $(HOST_SUFFIX)
    # Host (CPU) settings
    CXX = g++
    COMPILEFLAGS = -std=c++17 -O3 -Wall -Wextra
    LINKFLAGS = -std=c++17 -O3 -Wall -Wextra
    DEFINES  = 
    LINKLIBS = $(QUADMATH_LIB)
else
    TARGET_SUFFIX = $(DEVICE_SUFFIX)
    # CUDA (GPU) settings
    CXX = nvcc
    COMPILEFLAGS = -std=c++17 -arch=sm_$(ARCH) -O3 -x cu --extended-lambda -c
    LINKFLAGS = -std=c++17 -arch=sm_$(ARCH) -O3
    DEFINES  = 
    LINKLIBS = $(QUADMATH_LIB)
endif

ifneq (,$(filter $(LINKAGE), static lto))
    LIB_BASE ?= cufp
    LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
    COMPILEFLAGS += -DCCCL_FPMP_LIB=1 -DCCCL_FPEMU_LIB=1
    ifneq (,$(filter $(LINKAGE), lto))
	    ifeq (,$(filter $(T) $(TARGET), host))
            COMPILEFLAGS += -dlto
	    	LINKFLAGS += -dlto
		endif
    else
        # Static linkage without LTO requires relocatable device code
        ifeq (,$(filter $(T) $(TARGET), host))
          COMPILEFLAGS += -rdc=true
          LINKFLAGS += -rdc=true
		endif
    endif
else
    LIB_NAME =
endif

COMPILEFLAGS += $(EXTRA_FLAGS) $(ABI_FLAGS)
LINKFLAGS    += $(EXTRA_FLAGS)
# ===========================================================================================
# Data Types Selection via TYPE parameter
# ===========================================================================================
# TYPE can be: fp32, fp32-fp32, fp64, fpemu_acc, fpemu_def, fpemu_fast, fp64-fp64, fp128, or "all"
# Default: fp32 fp32-fp32 fp64 fp64-fp64 (no emulated types by default)

# Default types if not specified
TYPE ?= fp32 fp32-fp32 fp64 fp64-fp64

# Expand "all" to include all types
ifneq (,$(filter all ALL,$(TYPE)))
    override TYPE = fp32 fp32-fp32 fp64 fpemu_acc fpemu_def fpemu_fast fp64-fp64 fp128
endif

# Enable types based on TYPE parameter (cpp defaults all to 0)
ifneq (,$(filter fp32 FP32,$(TYPE)))
    DEFINES += -D__FPMP_FP32_ENABLED__=1
endif
ifneq (,$(filter fp32-fp32 FP32-FP32 fp32mp2 FP32MP2,$(TYPE)))
    DEFINES += -D__FPMP_FP32MP2_ENABLED__=1
endif
ifneq (,$(filter fp64 FP64,$(TYPE)))
    DEFINES += -D__FPMP_FP64_NATIVE_ENABLED__=1
endif
ifneq (,$(filter fpemu_acc FPEMU_ACC,$(TYPE)))
    DEFINES += -D__FPEMU_ACCURATE_ENABLED__=1
endif
ifneq (,$(filter fpemu_def FPEMU_DEF,$(TYPE)))
    DEFINES += -D__FPEMU_DEF_ENABLED__=1
endif
ifneq (,$(filter fpemu_fast FPEMU_FAST,$(TYPE)))
    DEFINES += -D__FPEMU_FAST_ENABLED__=1
endif
ifneq (,$(filter fp64-fp64 FP64-FP64 fp64mp2 FP64MP2,$(TYPE)))
    DEFINES += -D__FPMP_FP64MP2_ENABLED__=1
endif
# FP128 only enabled when FP128_EXTRA_FLAGS is set (requires FP128=1 at root level)
ifneq (,$(filter fp128 FP128,$(TYPE)))
    ifneq (,$(FP128_EXTRA_FLAGS))
        DEFINES += -D__FPMP_FP128_ENABLED__=1
        COMPILEFLAGS += $(FP128_EXTRA_FLAGS)
        LINKFLAGS    += $(FP128_EXTRA_FLAGS)
    endif
endif

# Add compile-time definitions
DEFINES += -DNUM_ITERATIONS=$(NUM_ITER)
DEFINES += -DTHREADS_PER_BLOCK=$(THREADS)
DEFINES += -DNUM_BLOCKS=$(NUM_BLOCKS)
DEFINES += -DREPS=$(REPS)
DEFINES += -DUNROLL=$(UNROLL)

# Include directories
INCLUDES = -I$(INCLUDE_DIR)
# verbose mode
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

# Verbose mode
ifeq ($(VERBOSE_MODE), 1)
    RULE_DONE = -@echo "benchmark: $(notdir $@) built"
else
    RULE_DONE = 
endif

# maximum verbose mode with -v flag
ifeq ($(VERBOSE_MODE), 3)
	COMPILEFLAGS += -v -Xptxas="-v"  --keep --keep-dir $(OUT)
	LINKFLAGS    += -v -Xptxas="-v" --keep --keep-dir $(OUT)
endif

ifeq ($(VERBOSE_MODE), 0)
	Q = @
endif

# Default target
all: run

# Build the executable
build: $(EXE)

ifneq (,$(filter $(T) $(TARGET), host))
# Host build: single-step compilation
$(EXE): $(SRC_FILE) $(LIB_NAME)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $< $(LIB_NAME)
	$(RULE_DONE)
else
# CUDA build: separate compilation and linking to avoid -x cu affecting library
$(OBJ): $(SRC_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

$(EXE): $(OBJ) $(LIB_NAME)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $< $(LIB_NAME) $(LINKLIBS)
	$(RULE_DONE)
endif

# Run the benchmark
run: $(EXE)
	$(Q)$(EXE)

rebuild:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build

rerun:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build
	MAKEFLAGS= $(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile run -j1

# Create output directory
$(OUT):
	$(Q)mkdir -p $(OUT)

$(LIB_NAME):
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/lib/make/lib.mk TARGET=$(TARGET) OUT=$(OUT) lib

# Clean up generated files
clean:
	$(Q)rm -rf $(OUT)

# Help target
help:
	@echo "Mixed Precision Throughput Benchmark"
	@echo "====================================="
	@echo ""
	@echo "Compares: float, float-float, double, FPEMU (accurate/def/fast), double-double, fp128"
	@echo ""
	@echo "Parameters:"
	@echo "  TARGET      = device|host (default: device)"
	@echo "  ARCH        = CUDA architecture (default: 86, use 100+ for fp128)"
	@echo "  NUM_ITER    = timing iterations (default: 10)"
	@echo "  THREADS     = threads per block (default: 256)"
	@echo "  NUM_BLOCKS  = number of CUDA blocks (default: 1024)"
	@echo "  REPS        = operations per thread (default: 1024)"
	@echo "  UNROLL      = unroll factor (default: 64)"
	@echo "  OUT         = output directory (default: _out)"
	@echo "  VERBOSE     = 0|1|2|3 (default: 1); aliases: V, VERB"
	@echo "  LINKAGE     = inline | static | lto (default: inline)"
	@echo "  TYPE       = data types to benchmark (space-separated or 'all')"
	@echo "                Options: fp32, fp32-fp32, fp64, fpemu_acc, fpemu_def, fpemu_fast, fp64-fp64, fp128"
	@echo "                Default: fp32 fp32-fp32 fp64 fp64-fp64"
	@echo ""
	@echo "Targets:"
	@echo "  all      - build and run (default)"
	@echo "  build    - build only"
	@echo "  run      - run (builds if needed)"
	@echo "  rebuild  - clean and build"
	@echo "  rerun    - clean, build, and run"
	@echo "  clean    - remove generated files"
	@echo "  help     - show this message"
	@echo ""
	@echo "Examples:"
	@echo "  make ARCH=89 run              # RTX 40xx (Ada Lovelace)"
	@echo "  make ARCH=100 run             # Blackwell with fp128"
	@echo "  make REPS=2048 NUM_BLOCKS=2048 run"
	@echo "  make TYPE=all run            # Benchmark all data types"
	@echo "  make TYPE=\"fp32 fp64\" run    # Benchmark only fp32 and fp64"
	@echo "  make TYPE=\"fpemu_acc fpemu_def fpemu_fast\" run # Benchmark FPEMU types"
	@echo ""
	@echo "Prerequisites:"
	@echo "  When FP128 support is enabled (_CCCL_FPMP_FP128_ENABLE=1),"
	@echo "  libquadmath is required for linking (provides expq, sinq, etc.)."
	@echo "  libquadmath is auto-detected (x86 only). Override with HAS_QUADMATH=1/0."
	@echo "  Current detection: HOST_ARCH=$(HOST_ARCH), HAS_QUADMATH=$(HAS_QUADMATH)"

.PHONY: all build run rebuild rerun clean help

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
