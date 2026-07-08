###########################################################################################
#
#  File:    Makefile
#  Purpose: Build and run atomic operations benchmarks
#  Author:  Andrei Kolesov
#  Date:    2025
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `ARCH`        = CUDA architecture (default: 86 for RTX 30xx/A100)
#    `NUM_ITER`    = iterations per thread (default: 50)
#    `THREADS`     = threads per block for CUDA (default: 64)
#    `NUM_BLOCKS`  = number of CUDA blocks (default: 16)
#    `LINKAGE`     = linkage mode (`inline` (default), `static`, `lto`)
#    `EXTRA_FLAGS` = extra compiler/linker flags (inherited from root Makefile if exported)
#    `OUT`         = output directory (default: _out)
#    `VERBOSE`     = verbose mode (`0` (silent), `1` (minimal), `2` (full), `3` (very verbose)); aliases: `V`, `VERB`
#
#  Build targets:
#  --------------
#    `all` (default) - build and run benchmark
#    `build`         - build benchmark executable only
#    `run`           - run the benchmark (builds if needed)
#    `rebuild`       - clean and build
#    `rerun`         - clean, build, and run
#    `clean`         - delete generated files
#
#  Usage examples:
#  ---------------
#    `make`                              - build and run on GPU
#    `make NUM_ITER=100 run`             - run with 100 iterations per thread
#    `make NUM_BLOCKS=32 THREADS=128`    - run with more threads
#    `make ARCH=75 rebuild`              - rebuild for Turing architecture
#    `make clean`                        - clean up generated files
#
#  Note: Build process uses two-stage compilation (source -> object -> executable)
#        to avoid -x cu flag treating libraries as CUDA source files
#
###########################################################################################

# Default parameters
# CUDA sm_xx: use ARCH if set; else first GPU (nvidia-smi); else 86.
ifndef ARCH
    _fpmp_cuda_arch_cap := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
    ifeq ($(strip $(_fpmp_cuda_arch_cap)),)
        ARCH := 86
    else
        ARCH := $(_fpmp_cuda_arch_cap)
    endif
endif
TARGET_SUFFIX ?=
NUM_ITER    ?= 50
THREADS     ?= 64
NUM_BLOCKS  ?= 16

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
MFP_DIR     = $(THIS_DIR)/../..
INCLUDE_DIR = $(THIS_DIR)/../../../../include
SRC_FILE    = $(THIS_DIR)/atomic.cpp

LINKAGE ?= inline

# Output directory
OUT ?= $(THIS_DIR)/_out

# Object and executable names
OBJ_FILE = $(OUT)/atomic.o
EXE      = $(OUT)/atomic.exe

# Compiler settings (CUDA required)
CXX = nvcc
CXXFLAGS = -std=c++17 -arch=sm_$(ARCH) -O3
CUFLAGS  = -x cu
LINKFLAGS = 
DEFINES  =
LINKLIBS  = $(QUADMATH_LIB)

# Add compile-time definitions
DEFINES += -DNUM_ITERATIONS=$(NUM_ITER)
DEFINES += -DTHREADS_PER_BLOCK=$(THREADS)
DEFINES += -DNUM_BLOCKS=$(NUM_BLOCKS)

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

ifeq ($(VERBOSE_MODE), 0)
	Q = @
endif

# Verbose mode
ifeq ($(VERBOSE_MODE), 1)
    RULE_DONE = -@echo "benchmark: $(notdir $@) built"
else
    RULE_DONE = 
endif

# maximum verbose mode with -v flag
ifeq ($(VERBOSE_MODE), 3)
	CXXFLAGS     += -v -Xptxas="-v" --keep --keep-dir $(OUT)
	LINKFLAGS    += -v -Xptxas="-v" --keep --keep-dir $(OUT)
endif


ifneq (,$(filter $(LINKAGE), static lto))
    LIB_BASE ?= cufp
    LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
    DEFINES += -DCCCL_FPMP_LIB=1 -DCCCL_FPEMU_LIB=1
    ifneq (,$(filter $(LINKAGE), lto))
        CXXFLAGS  += -dlto
	    LINKFLAGS += -dlto
    else
        # Static linkage without LTO requires relocatable device code
        CXXFLAGS  += -rdc=true
        LINKFLAGS += -rdc=true
    endif
else
    LIB_NAME =
endif

CXXFLAGS  += $(EXTRA_FLAGS) $(ABI_FLAGS)
LINKFLAGS  += $(EXTRA_FLAGS)

# Default target
all: run

# Build the executable
build: $(EXE)

# Compile source to object file
$(OBJ_FILE): $(SRC_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(CXXFLAGS) $(CUFLAGS) $(DEFINES) $(INCLUDES) -c -o $@ $<
	$(RULE_DONE)

# Link object file to executable
$(EXE): $(OBJ_FILE) $(LIB_NAME)
	$(Q)$(CXX) $(CXXFLAGS) $(LINKFLAGS) -o $@ $< $(LIB_NAME) $(LINKLIBS)
	$(RULE_DONE)

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
	@echo "Float-Float Atomic Operations Benchmark Makefile"
	@echo "================================================="
	@echo ""
	@echo "Parameters:"
	@echo "  ARCH        = CUDA architecture (default: 86)"
	@echo "  NUM_ITER    = iterations per thread (default: 50)"
	@echo "  THREADS     = threads per block (default: 64)"
	@echo "  NUM_BLOCKS  = number of blocks (default: 16)"
	@echo "  LINKAGE     = linkage mode (inline|static|lto, default: inline)"
	@echo "  EXTRA_FLAGS = extra compiler/linker flags"
	@echo "  OUT         = output directory (default: _out)"
	@echo "  VERBOSE     = 0|1|2|3 (default: 1); aliases: V, VERB"
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
	@echo "Note: This benchmark measures atomic operations under high contention"
	@echo "      All threads update the SAME memory location (worst-case scenario)"
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

