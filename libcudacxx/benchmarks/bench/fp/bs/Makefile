###########################################################################################
#
#  File:    Makefile
#  Purpose: Build and run Black-Scholes option pricing benchmark
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `TARGET`      = target device (`device` (default), `host`)
#    `ARCH`        = CUDA architecture (default: 86 for RTX 30xx/A100)
#    `NUM_OPTIONS` = options per launch (default: 1048576 = 1M GPU / 32K host)
#    `REPS`        = inner repetitions per kernel launch (default: 128 GPU / 16 host)
#                    Higher = longer kernels = more stable timing
#    `NUM_ITER`    = number of timing iterations (default: 10)
#    `THREADS`     = threads per block for CUDA (default: 256)
#    `NUM_BLOCKS`  = number of CUDA blocks (default: 2048)
#    `OUT`         = output directory (default: _out)
#    `VERBOSE`     = verbose mode (`0`..`3`); aliases: `V`, `VERB`
#    `LINKAGE`     = linkage mode (`inline` (default), `static`, `lto`)
#    `FP128`       = include fp128_t type in benchmark (`0` (default), `1`)
#                    Requires ARCH=100+ for GPU builds
#    `FP128_REF`   = use quad-precision reference prices (`1` (default), `0`)
#    `EXTRA_FLAGS` = extra compiler/linker flags
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
#    `make REPS=256 run`                 - run with higher rigor
#    `make NUM_OPTIONS=4194304 run`      - run with 4M options
#    `make T=host rerun`                 - rebuild and run on CPU
#    `make ARCH=75 rebuild`              - rebuild for Turing
#    `make ARCH=100 FP128=1 rerun`       - Blackwell GPU with fp128
#    `make T=host FP128=1 rerun`         - CPU with fp128
#    `make FP128_REF=0 rerun`            - use double reference instead of float128
#    `make clean`                        - clean up
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
NUM_BLOCKS  ?= 2048

ifneq (,$(filter $(T) $(TARGET), host))
    NUM_OPTIONS ?= 1024*32
    REPS        ?= 2
else
    NUM_OPTIONS ?= 1024*1024
    REPS        ?= 128
endif

# Current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

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
    ABI_FLAGS = -D__FPMP_ABI_PRESERVE_N_DATA__=$(ABI_DATA) -D__FPMP_ABI_PRESERVE_N_CONTROL__=$(ABI_CONTROL) \
                -D__FPEMU_ABI_PRESERVE_N_DATA__=$(ABI_DATA) -D__FPEMU_ABI_PRESERVE_N_CONTROL__=$(ABI_CONTROL)
else
    ABI_FLAGS =
endif

# Source and include directories
INCLUDE_DIR = $(THIS_DIR)/../../../../include
SRC_FILE    = $(THIS_DIR)/bs.cpp
REF_FILE    = $(THIS_DIR)/bs_ref.cpp
MFP_DIR     = $(THIS_DIR)/../..

LINKAGE ?= inline

# Float128 reference: set FP128_REF=1 to use quad-precision reference
FP128_REF ?= 1

# Float128 benchmark type: set FP128=1 to include fp128_t in benchmarked types
# Requires ARCH=100+ for GPU builds
FP128 ?= 0

# Output directory
OUT ?= $(THIS_DIR)/_out

# Executable name
EXE = $(OUT)/bs.exe
OBJ = $(OUT)/bs.o
REF_OBJ = $(OUT)/bs_ref.o

# Compiler settings
ifneq (,$(filter $(T) $(TARGET), host))
    TARGET_SUFFIX = $(HOST_SUFFIX)
    CXX = g++
    COMPILEFLAGS = -std=c++17 -O3 -Wall -Wextra -c
    LINKFLAGS = -std=c++17 -O3 -Wall -Wextra
    DEFINES  =
    LINKLIBS = $(QUADMATH_LIB)
else
    TARGET_SUFFIX = $(DEVICE_SUFFIX)
    CXX = nvcc
    COMPILEFLAGS = -std=c++17 -arch=sm_$(ARCH) -O3 -x cu --extended-lambda -c
    LINKFLAGS = -std=c++17 -arch=sm_$(ARCH) -O3
    DEFINES  =
    LINKLIBS = $(QUADMATH_LIB)
endif

ifneq (,$(filter $(LINKAGE), static lto))
    LIB_BASE ?= cufp
    LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
    COMPILEFLAGS += -DFPMP_LIB=1 -DFPEMU_LIB=1
    ifneq (,$(filter $(LINKAGE), lto))
        ifeq (,$(filter $(T) $(TARGET), host))
            COMPILEFLAGS += -dlto
            LINKFLAGS += -dlto
        endif
    else
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

# Add compile-time definitions
DEFINES += -DNUM_OPTIONS="$(NUM_OPTIONS)"
DEFINES += -DREPS=$(REPS)
DEFINES += -DNUM_ITERATIONS=$(NUM_ITER)
DEFINES += -DTHREADS_PER_BLOCK=$(THREADS)
DEFINES += -DNUM_BLOCKS=$(NUM_BLOCKS)

# Float128 reference with custom erfcq
ifeq ($(FP128_REF),1)
    DEFINES += -D__USE_FP128_REFERENCE__
endif

# Float128 benchmark type
ifeq ($(FP128),1)
    DEFINES += -D__FPMP_FP128_ENABLED__=1
    COMPILEFLAGS += $(FP128_EXTRA_FLAGS)
    LINKFLAGS    += $(FP128_EXTRA_FLAGS)
endif

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

ifeq ($(VERBOSE_MODE), 1)
    RULE_DONE = -@echo "bs: $(notdir $@) built"
else
    RULE_DONE =
endif

ifeq ($(VERBOSE_MODE), 3)
	COMPILEFLAGS += -v -Xptxas="-v" --keep --keep-dir $(OUT)
	LINKFLAGS    += -v -Xptxas="-v" --keep --keep-dir $(OUT)
endif

ifeq ($(VERBOSE_MODE), 0)
	Q = @
endif

# Default target
all: run

# Build the executable
build: $(EXE)

# Separate compilation and linking
$(OBJ): $(SRC_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

# Float128 reference — always compiled by g++ (uses libquadmath where available)
$(REF_OBJ): $(REF_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)g++ -std=c++17 -O3 -fext-numeric-literals -c -o $@ $<
	$(RULE_DONE)

ifeq ($(FP128_REF),1)
$(EXE): $(OBJ) $(REF_OBJ) $(LIB_NAME)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $(OBJ) $(REF_OBJ) $(LIB_NAME) $(LINKLIBS)
	$(RULE_DONE)
else
$(EXE): $(OBJ) $(LIB_NAME)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $< $(LIB_NAME) $(LINKLIBS)
	$(RULE_DONE)
endif

# Run the benchmark
run: $(EXE)
	$(Q)$(EXE) 2>&1

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
	$(MAKE) --no-print-directory -f $(MFP_DIR)/make/lib.mk TARGET=$(TARGET) OUT=$(OUT) lib

# Clean up generated files
clean:
	$(Q)rm -rf $(OUT)

# Help target
help:
	@echo "Black-Scholes Option Pricing Benchmark"
	@echo "======================================="
	@echo ""
	@echo "Parameters:"
	@echo "  TARGET      = device|host (default: device)"
	@echo "  ARCH        = CUDA architecture (default: 86)"
	@echo "  NUM_OPTIONS = options per launch (default: 1048576)"
	@echo "  REPS        = inner reps per kernel (default: 128, controls rigor)"
	@echo "  NUM_ITER    = timing iterations (default: 10)"
	@echo "  THREADS     = threads per block (default: 256)"
	@echo "  NUM_BLOCKS  = CUDA blocks (default: 2048)"
	@echo "  OUT         = output directory (default: _out)"
	@echo "  VERBOSE     = 0|1|2|3 (default: 1)"
	@echo "  LINKAGE     = inline|static|lto (default: inline)"
	@echo "  FP128       = 0|1 (default: 0, include fp128_t, requires ARCH=100+ for GPU)"
	@echo "  FP128_REF   = 0|1 (default: 1, use quad-precision reference)"
	@echo ""
	@echo "Targets:"
	@echo "  all      - build and run (default)"
	@echo "  build    - build only"
	@echo "  run      - run (builds if needed)"
	@echo "  rebuild  - clean and build"
	@echo "  rerun    - clean, build, and run"
	@echo "  clean    - remove generated files"
	@echo "  help     - show this message"

.PHONY: all build run rebuild rerun clean help

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
