###########################################################################################
#
#  File:    Makefile
#  Purpose: Build and run PCM off-diagonal matrix-vector product benchmark
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    Note: this benchmark is GPU-only (CUDA/nvcc).
#    `ARCH`        = CUDA architecture (default: 86)
#    `NPOINT`      = number of surface points (default: 10000, range: 8000-30000)
#    `THREADS`     = threads per block for CUDA (default: 256)
#    `NUM_BLOCKS`  = CUDA grid size (default: 512)
#    `NUM_ITER`    = number of timing iterations (default: 10)
#    `OUT`         = output directory (default: _out)
#    `VERBOSE`     = verbose mode (`0`..`3`); aliases: `V`, `VERB`
#    `LINKAGE`     = linkage mode (`inline` (default), `static`, `lto`)
#    `ACCURACY`      = ffloat (fp32mp2) accuracy (`def` (default), `low`, `mid`, `high`)
#    `FMAD_FLAG`   = -fmad= flag (default: true)
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
#    `make`                              - build and run (npoint=10000)
#    `make NPOINT=8000 rerun`            - run with 8000 points
#    `make NPOINT=20000 rerun`           - run with 20000 points
#    `make NPOINT=30000 rerun`           - run with 30000 points
#    `make ARCH=90 rerun`                - rebuild for Hopper
#    `make clean`                        - clean up
#
###########################################################################################

.NOTPARALLEL:

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
NPOINT      ?= 10000
THREADS     ?= 256
NUM_BLOCKS  ?= 512
NUM_ITER    ?= 10
FMAD_FLAG   ?= true
DEVICE_SUFFIX ?=
HOST_SUFFIX   ?= _host

# Current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Dev-only library build (make/lib.mk + src/) lives in the sibling cccl-fpdev
# repo; overridable (the cccl-fpdev root Makefile passes it explicitly).
FPDEV_DIR ?= $(abspath $(THIS_DIR)/../../../../../../cccl-fpdev)
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
SRC_FILE    = $(THIS_DIR)/pcm.cu
MFP_DIR     = $(THIS_DIR)/../..

LINKAGE ?= inline

# Output directory
OUT ?= $(THIS_DIR)/_out

# Executable name
EXE  = $(OUT)/pcm.exe
OBJ  = $(OUT)/pcm.o

# Compiler settings
ifneq (,$(filter $(T) $(TARGET), host))
    TARGET_SUFFIX = $(HOST_SUFFIX)
    CXX = g++
    CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -x c++ -c
    LINKFLAGS = -std=c++17 -O3 -Wall -Wextra
    LIBFLAGS =
    LIBS =
    LIB_NAME =
else
    TARGET_SUFFIX = $(DEVICE_SUFFIX)
    CXX := nvcc
    CXXFLAGS := -std=c++17 -O3 -arch=sm_$(ARCH) -fmad=$(FMAD_FLAG) --extended-lambda
    LINKFLAGS :=
    LIBFLAGS :=
    LIBS :=

    DEVCOMP := -dc

    ifneq (,$(filter $(LINKAGE), static lto))
        LIB_BASE ?= cufp
        LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
        CXXFLAGS += -DCCCL_FPMP_LIB=1
        ifneq (,$(filter $(LINKAGE), lto))
            DEVCOMP := -dc -dlto
            CXXFLAGS += -dlto
            LIBFLAGS += -dlto
        else
            CXXFLAGS += -rdc=true
            LIBFLAGS += -rdc=true
        endif
        LIBS += $(LIB_NAME)
    else
        LIB_NAME =
    endif
endif

CXXFLAGS  += $(EXTRA_FLAGS) $(ABI_FLAGS)
LIBFLAGS  += $(EXTRA_FLAGS)

# Compile-time definitions
DEFINES  = -DNPOINT=$(NPOINT)
DEFINES += -DTHREADS_PER_BLOCK=$(THREADS)
DEFINES += -DNUM_BLOCKS=$(NUM_BLOCKS)
DEFINES += -DNUM_ITERATIONS=$(NUM_ITER)

# ACC: short alias for ACCURACY (e.g. ACC=mid); ACCURACY takes precedence
ifdef ACC
ACCURACY ?= $(ACC)
endif

# ffloat accuracy: def (default), low, mid, high
ifneq (,$(ACCURACY))
    ifneq (,$(filter $(ACCURACY), def low mid high))
        DEFINES += -D__METHOD__=$(ACCURACY)
    endif
else
    DEFINES += -D__METHOD__=def
endif

ifneq (,$(MIN_BLOCKS))
    DEFINES += -DMIN_BLOCKS_PER_SM=$(MIN_BLOCKS)
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
    RULE_DONE = -@echo "pcm: $(notdir $@) built"
else
    RULE_DONE =
endif

ifeq ($(VERBOSE_MODE), 3)
	CXXFLAGS += -v -Xptxas="-v" --keep --keep-dir $(OUT)
	LIBFLAGS += -v -Xptxas="-v" --keep --keep-dir $(OUT)
endif

ifeq ($(VERBOSE_MODE), 0)
	Q = @
endif

# Default target
all: run

# Build the executable
build: $(EXE)

ifeq (,$(filter $(T) $(TARGET), host))
# --- Device (GPU) build ---
$(OBJ): $(SRC_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) $(DEVCOMP) $< -o $@
	$(RULE_DONE)

$(EXE): $(OBJ) $(LIB_NAME)
	$(Q)$(CXX) -arch=sm_$(ARCH) $(LIBFLAGS) -o $@ $(OBJ) $(LIBS)
	$(RULE_DONE)

run: $(EXE)
	$(Q)$(EXE)
else
# --- Host build (prints warning and exits) ---
$(OBJ): $(SRC_FILE)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

$(EXE): $(OBJ)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $(OBJ)
	$(RULE_DONE)

run: $(EXE)
	$(Q)$(EXE)
endif

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
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/make/lib.mk TARGET=device OUT=$(OUT) lib

# Clean up generated files
clean:
	$(Q)rm -rf $(OUT)

# Help target
help:
	@echo "PCM Off-Diagonal Matrix-Vector Product Benchmark (GPU Only)"
	@echo "============================================================"
	@echo ""
	@echo "Parameters:"
	@echo "  ARCH       = CUDA architecture (default: 86)"
	@echo "  NPOINT     = number of surface points (default: 10000)"
	@echo "  THREADS    = threads per block (default: 256)"
	@echo "  NUM_BLOCKS = CUDA blocks (default: 512)"
	@echo "  NUM_ITER   = timing iterations (default: 10)"
	@echo "  OUT        = output directory (default: _out)"
	@echo "  VERBOSE    = 0|1|2|3 (default: 1)"
	@echo "  ACCURACY     = ffloat accuracy: def (default), low, mid, high [alias: ACC]"
	@echo "  LINKAGE    = inline|static|lto (default: inline)"
	@echo "  FMAD_FLAG  = fmad flag (default: true)"
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
	@echo "Usage:"
	@echo "  make rerun                     # default (npoint=10000)"
	@echo "  make NPOINT=8000 rerun         # 8000 points"
	@echo "  make NPOINT=20000 rerun        # 20000 points"
	@echo "  make NPOINT=30000 rerun        # 30000 points"
	@echo "  make ACCURACY=low rerun          # ffloat low accuracy"
	@echo "  make ACCURACY=high rerun         # ffloat high accuracy"
	@echo "  make ARCH=90 rerun             # Hopper GPU"

.PHONY: all build run rebuild rerun clean help

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
