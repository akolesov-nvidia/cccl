###########################################################################################
#
#  File:    Makefile
#  Purpose: Build and run accuracy & performance test for standalone FP64 add emulation
#  Author:  Andrei Kolesov
#  Date:    2025
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `TARGET`      = target device (`host` (default), `device`)
#    `ARCH`        = CUDA architecture (default: 86 for RTX 30xx/A100)
#    `RANDOM_LEN`  = number of random test pairs (default: 65536)
#    `ACCUM_LEN`   = number of accumulation steps (default: 16384)
#    `SEED`        = RNG seed (default: 42)
#    `NUM_ITER`    = number of timing iterations (default: 10)
#    `THREADS`     = threads per block for CUDA (default: 256)
#    `NUM_BLOCKS`  = number of CUDA blocks (default: 1024)
#    `REPS`        = operations per thread for perf (default: 1024)
#    `UNROLL`      = loop unroll factor (default: 64)
#    `OUT`         = output directory (default: _out)
#    `VERBOSE`     = verbose mode (`0` (silent), `1` (minimal), `2` (full), `3` (very verbose)); aliases: `V`, `VERB`
#    `EXTRA_FLAGS` = extra compiler/linker flags
#
#  Build targets:
#  --------------
#    `all` (default) - build and run test
#    `build`         - build test executable only
#    `run`           - run the test (builds if needed)
#    `rebuild`       - clean and build
#    `rerun`         - clean, build, and run
#    `clean`         - delete generated files
#
#  Usage examples:
#  ---------------
#    `make`                                   - build and run on host (accuracy only)
#    `make TARGET=device ARCH=80`             - build and run on GPU (accuracy + perf)
#    `make TARGET=device REPS=2048 run`       - run with 2048 ops per thread
#    `make RANDOM_LEN=1048576 run`            - run with 1M random tests
#    `make SEED=123 rerun`                    - rerun with different seed
#    `make clean`                             - clean up generated files
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
RANDOM_LEN  ?= 65536
ACCUM_LEN   ?= 16384
SEED        ?= 42
NUM_ITER    ?= 10
THREADS     ?= 256
NUM_BLOCKS  ?= 1024
REPS        ?= 1024
UNROLL      ?= 64

# Current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
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
INCLUDE_DIR = $(THIS_DIR)
SRC_FILE    = $(THIS_DIR)/test_fpemu_add.cpp

# Output directory
OUT ?= $(THIS_DIR)/_out

# Executable name
EXE = $(OUT)/test_fpemu_add.exe
OBJ = $(OUT)/test_fpemu_add.o


# Compiler settings
ifneq (,$(filter $(T) $(TARGET), host))
    TARGET_SUFFIX = $(HOST_SUFFIX)
    # Host (CPU) settings — single-step compile+link, no -c flag
    CXX = g++
    COMPILEFLAGS = -std=c++17 -O3 -Wall -Wextra
    LINKFLAGS = -std=c++17 -O3 -Wall -Wextra
    DEFINES  = 
    LINKLIBS =
else
    TARGET_SUFFIX = $(DEVICE_SUFFIX)
    # CUDA (GPU) settings
    CXX = nvcc
    COMPILEFLAGS = -std=c++17 -arch=sm_$(ARCH) -O3 -x cu --extended-lambda -c
    LINKFLAGS = -std=c++17 -arch=sm_$(ARCH) -O3
    DEFINES  = 
    LINKLIBS =
endif

COMPILEFLAGS += $(EXTRA_FLAGS) $(ABI_FLAGS)
LINKFLAGS    += $(EXTRA_FLAGS)

# Treat warnings as errors when WERROR is enabled (default 0 for benchmarks).
# host: -Wall -Wextra -Werror; device (nvcc): forwarded via -Xcompiler. Enable with WERROR=1.
WERROR ?= 0
ifeq (,$(filter 0 n N no No false False,$(WERROR)))
  ifneq (,$(filter $(T) $(TARGET), host))
    COMPILEFLAGS += -Wall -Wextra -Wno-strict-aliasing -Werror
  else
    COMPILEFLAGS += -Xcompiler -Wall,-Wextra,-Wno-strict-aliasing,-Werror
  endif
endif

# Add compile-time definitions
DEFINES += -DRANDOM_LEN=$(RANDOM_LEN)
DEFINES += -DACCUMULATE_LEN=$(ACCUM_LEN)
DEFINES += -DSEED=$(SEED)
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
    RULE_DONE = -@echo "test: $(notdir $@) built"
else
    RULE_DONE = 
endif

# maximum verbose mode with -v flag
ifeq ($(VERBOSE_MODE), 3)
	COMPILEFLAGS += -v
	LINKFLAGS    += -v
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
$(EXE): $(SRC_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)
else
# CUDA build: separate compilation and linking
$(OBJ): $(SRC_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

$(EXE): $(OBJ)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $< $(LINKLIBS)
	$(RULE_DONE)
endif

# Run the test
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

# Clean up generated files
clean:
	$(Q)rm -rf $(OUT)

# Help target
help:
	@echo "FP64 Add Emulation Test Makefile"
	@echo "================================"
	@echo ""
	@echo "Parameters:"
	@echo "  TARGET      = host|device (default: host)"
	@echo "  ARCH        = CUDA architecture (default: 86)"
	@echo "  RANDOM_LEN  = random test count (default: 65536)"
	@echo "  ACCUM_LEN   = accumulation steps (default: 16384)"
	@echo "  SEED        = RNG seed (default: 42)"
	@echo "  NUM_ITER    = timing iterations (default: 10)"
	@echo "  THREADS     = threads per block (default: 256)"
	@echo "  NUM_BLOCKS  = number of CUDA blocks (default: 1024)"
	@echo "  REPS        = operations per thread (default: 1024)"
	@echo "  UNROLL      = unroll factor (default: 64)"
	@echo "  OUT         = output directory (default: _out)"
	@echo "  VERBOSE     = 0|1|2|3 (default: 1); aliases: V, VERB"
	@echo "  EXTRA_FLAGS = extra compiler/linker flags"
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
	@echo "This test validates FP64 add emulation accuracy and performance"
	@echo "by comparing against native double-precision addition."
	@echo ""
	@echo "  HOST build:   accuracy tests only (no GPU needed)"
	@echo "  DEVICE build: accuracy + performance (GPU required)"

.PHONY: all build run rebuild rerun clean help

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
