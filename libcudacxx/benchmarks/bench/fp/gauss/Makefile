###########################################################################################
#
#  File:    Makefile
#  Purpose: Build and run double-precision Gaussian RNG performance benchmark
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `ARCH`        = CUDA architecture (default: 86 for RTX 30xx/A100)
#    `NUM_SAMPLES` = samples per launch (default: 16777216 = 16M)
#    `REPS`        = inner repetitions per kernel launch (default: 128)
#                    Higher = longer kernels = more stable timing
#    `NUM_ITER`    = number of timing iterations (default: 10)
#    `THREADS`     = threads per block for CUDA (default: 256)
#    `NUM_BLOCKS`  = number of CUDA blocks (default: 2048)
#    `OUT`         = output directory (default: _out)
#    `VERBOSE`     = verbose mode (`0`..`3`); aliases: `V`, `VERB`
#    `LINKAGE`     = linkage mode (`inline` (default), `static`, `lto`)
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
#    `make`                              - build and run
#    `make REPS=256 run`                 - run with higher rigor
#    `make NUM_SAMPLES=67108864 run`     - run with 64M samples
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
NUM_SAMPLES ?= 1024*1024*1024
REPS        ?= 8
NUM_ITER    ?= 8
THREADS     ?= 256
NUM_BLOCKS  ?= 2048

# Current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
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
SRC_FILE    = $(THIS_DIR)/gauss.cpp
MFP_DIR     = $(THIS_DIR)/../..

LINKAGE ?= inline

# Output directory
OUT ?= $(THIS_DIR)/_out

# Executable name
EXE = $(OUT)/gauss.exe
OBJ = $(OUT)/gauss.o

# Compiler settings
ifneq (,$(filter $(T) $(TARGET), host))
    TARGET_SUFFIX = $(HOST_SUFFIX)
    CXX = g++
    COMPILEFLAGS = -std=c++17 -O3 -Wall -Wextra -c
    LINKFLAGS = -std=c++17 -O3 -Wall -Wextra
    DEFINES  =
    LINKLIBS =
    LIB_NAME =
else
    TARGET_SUFFIX = $(DEVICE_SUFFIX)
    CXX = nvcc
    COMPILEFLAGS = -std=c++17 -arch=sm_$(ARCH) -O3 -x cu --extended-lambda -c
    LINKFLAGS = -std=c++17 -arch=sm_$(ARCH) -O3
    DEFINES  =
    LINKLIBS =

    ifneq (,$(filter $(LINKAGE), static lto))
        LIB_BASE ?= cufp
        LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
        COMPILEFLAGS += -DFPMP_LIB=1 -DFPEMU_LIB=1
        ifneq (,$(filter $(LINKAGE), lto))
            COMPILEFLAGS += -dlto
            LINKFLAGS += -dlto
        else
            COMPILEFLAGS += -rdc=true
            LINKFLAGS += -rdc=true
        endif
    else
        LIB_NAME =
    endif
endif

COMPILEFLAGS += $(EXTRA_FLAGS) $(ABI_FLAGS)
LINKFLAGS    += $(EXTRA_FLAGS)

# Add compile-time definitions
DEFINES += -DNUM_SAMPLES="$(NUM_SAMPLES)"
DEFINES += -DREPS=$(REPS)
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

ifeq ($(VERBOSE_MODE), 1)
    RULE_DONE = -@echo "gauss: $(notdir $@) built"
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

$(EXE): $(OBJ) $(LIB_NAME)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $< $(LIB_NAME) $(LINKLIBS)
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
	$(MAKE) --no-print-directory -f $(MFP_DIR)/make/lib.mk TARGET=device OUT=$(OUT) lib

# Clean up generated files
clean:
	$(Q)rm -rf $(OUT)

# Help target
help:
	@echo "Gaussian RNG Performance Benchmark"
	@echo "==================================="
	@echo ""
	@echo "Parameters:"
	@echo "  ARCH        = CUDA architecture (default: 86)"
	@echo "  NUM_SAMPLES = samples per launch (default: 16777216)"
	@echo "  REPS        = inner reps per kernel (default: 128)"
	@echo "  NUM_ITER    = timing iterations (default: 10)"
	@echo "  THREADS     = threads per block (default: 256)"
	@echo "  NUM_BLOCKS  = CUDA blocks (default: 2048)"
	@echo "  OUT         = output directory (default: _out)"
	@echo "  VERBOSE     = 0|1|2|3 (default: 1)"
	@echo "  LINKAGE     = inline|static|lto (default: inline)"
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
