###########################################################################################
#
#  File:    Makefile
#  Purpose: Build and run Monte Carlo European option pricing benchmark
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `TARGET`      = target device (`device` (default), `host`)
#    `ARCH`        = CUDA architecture (default: 86 for RTX 30xx/A100)
#    `NUM_PATHS`   = total Monte Carlo paths (default: 16777216 = 16M)
#    `REPS`        = inner repetitions per kernel launch (default: 128)
#                    Higher = longer kernels = more stable timing
#    `NUM_ITER`    = number of timing iterations (default: 10)
#    `THREADS`     = threads per block for CUDA (default: 256)
#    `NUM_BLOCKS`  = number of CUDA blocks (default: 2048)
#    `OUT`         = output directory (default: _out)
#    `VERBOSE`     = verbose mode (`0`..`3`); aliases: `V`, `VERB`
#    `STRIKE`      = option strike price (default: 100.0)
#                    Increase to make the option deep OTM and stress tail precision
#    `SIGMA`       = volatility (default: 0.2)
#                    Increase to amplify exp() arguments and stress arithmetic
#    `ICDF`        = inverse CDF method for inline RNG (`0` (default), `1`)
#                    0 = Box-Muller (curand_normal_double / std::normal_distribution)
#                    1 = uniform + normcdfinv
#    `FP128_REF`   = use quad-precision BS analytical reference (`1` (default), `0`)
#                    Uses __float128 / libquadmath for maximum reference accuracy
#    `LINKAGE`     = linkage mode (`inline` (default), `static`, `lto`)
#    `EXTRA_FLAGS` = extra compiler/linker flags
#
#  Both pre-generated and inline RNG modes run in every execution.
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
#    `make REPS=256 run`                 - run with higher rigor (longer kernels)
#    `make NUM_PATHS=67108864 run`       - run with 64M paths
#    `make ICDF=1 rerun`                 - rebuild with inverse CDF method
#    `make STRIKE=200 SIGMA=0.4 rerun`  - deep OTM with higher volatility
#    `make FP128_REF=0 rerun`           - use double-precision BS reference
#    `make ARCH=75 rebuild`              - rebuild for Turing
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
ICDF        ?= 0
STRIKE      ?= 100.0
SIGMA       ?= 0.2

ifneq (,$(filter $(T) $(TARGET), host))
    NUM_PATHS ?= 1024*16
    REPS      ?= 2
else
    NUM_PATHS ?= 1024*1024*16
    REPS      ?= 128
endif

# Current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Dev-only library build (make/lib.mk + src/) lives in the sibling cccl-fpdev
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
SRC_FILE    = $(THIS_DIR)/mc.cpp
HOST_FILE   = $(THIS_DIR)/mc_host.cpp
MFP_DIR     = $(THIS_DIR)/../..

LINKAGE ?= inline

# Float128 reference: set FP128_REF=1 to use quad-precision BS analytical price
FP128_REF ?= 1

# Output directory
OUT ?= $(THIS_DIR)/_out

# Executable name
EXE = $(OUT)/mc.exe
OBJ = $(OUT)/mc.o
HOST_OBJ = $(OUT)/mc_host.o

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
    COMPILEFLAGS += -DCCCL_FPMP_LIB=1 -DCCCL_FPEMU_LIB=1
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
DEFINES += -DNUM_PATHS=$(NUM_PATHS)
DEFINES += -DREPS=$(REPS)
DEFINES += -DNUM_ITERATIONS=$(NUM_ITER)
DEFINES += -DTHREADS_PER_BLOCK=$(THREADS)
DEFINES += -DNUM_BLOCKS=$(NUM_BLOCKS)
DEFINES += -DMC_STRIKE=$(STRIKE)
DEFINES += -DMC_SIGMA=$(SIGMA)
ifneq ($(ICDF), 0)
    DEFINES += -DMC_RNG_ICDF=$(ICDF)
endif

# Float128 reference
ifeq ($(FP128_REF),1)
    DEFINES += -D__USE_FP128_REFERENCE__
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
    RULE_DONE = -@echo "mc: $(notdir $@) built"
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

# Separate compilation and linking (both host and device)
$(OBJ): $(SRC_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

# Host helpers — always compiled by g++ (needs __float128 / libquadmath)
$(HOST_OBJ): $(HOST_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)g++ -std=c++17 -O3 -fext-numeric-literals -c -o $@ $<
	$(RULE_DONE)

$(EXE): $(OBJ) $(HOST_OBJ) $(LIB_NAME)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $(OBJ) $(HOST_OBJ) $(LIB_NAME) $(LINKLIBS)
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
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/make/lib.mk TARGET=$(TARGET) OUT=$(OUT) lib

# Clean up generated files
clean:
	$(Q)rm -rf $(OUT)

# Help target
help:
	@echo "Monte Carlo Option Pricing Benchmark"
	@echo "====================================="
	@echo ""
	@echo "Parameters:"
	@echo "  TARGET     = device|host (default: device)"
	@echo "  ARCH       = CUDA architecture (default: 86)"
	@echo "  NUM_PATHS  = Monte Carlo paths (default: 16777216)"
	@echo "  REPS       = inner reps per kernel (default: 128, controls rigor)"
	@echo "  STRIKE     = strike price (default: 100.0)"
	@echo "               Increase for deep OTM options (e.g. 160, 200)"
	@echo "  SIGMA      = volatility (default: 0.2)"
	@echo "               Increase to amplify tail effects (e.g. 0.5, 1.0)"
	@echo "  ICDF       = 0|1 (default: 0) Gaussian method for inline RNG"
	@echo "               0=Box-Muller 1=normcdfinv"
	@echo "  NUM_ITER   = timing iterations (default: 10)"
	@echo "  THREADS    = threads per block (default: 256)"
	@echo "  NUM_BLOCKS = CUDA blocks (default: 2048)"
	@echo "  OUT        = output directory (default: _out)"
	@echo "  VERBOSE    = 0|1|2|3 (default: 1)"
	@echo "  FP128_REF  = 0|1 (default: 1, use quad-precision BS reference)"
	@echo "  LINKAGE    = inline|static|lto (default: inline)"
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
	@echo "Runtime options (passed to executable):"
	@echo "  --s0 X  --k X  --r X  --q X  --sigma X  --t X"
	@echo "  --call | --put"
	@echo "  --paths N"
	@echo "  --antithetic | --no-antithetic"

.PHONY: all build run rebuild rerun clean help

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
