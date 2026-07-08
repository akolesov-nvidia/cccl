###########################################################################################
#
#  File:    Makefile
#  Purpose: Build FFT benchmark
#  Author:  Andrei Kolesov
#  Date:    2025-01-15
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    Note: this benchmark is GPU-only (CUDA/nvcc).
#    `OUT`        = output directory (default is `_out`)
#    `LINKAGE`    = use library or inline emulation cores (`inline` (default), `static`, `lto`)
#    `VERBOSE`    = verbose mode (`0` (silent), `1` (minimal), `2` (full), `3` (very verbose)); aliases: `V`, `VERB`
#    `SIZE`       = FFT size (default: 1024)
#    `BATCH`      = batch size (default: 8)
#    `ITERATIONS` = number of iterations (default: 100)
#    `ARCH`       = CUDA architecture (default: 86)
#    `EMULATION`  = set to disable emulation (adds `-DNO_EMULATION`)
#    `CALLBACKS`, `CALLBACK_ROUNDING_MODE_MANTISSA`, `CALLBACK_ROUNDING_MODE_EXPONENT` = optional callback configuration
#
#  Build targets:
#  --------------
#    `all` (default) - build benchmark
#    `run`           - run benchmark
#    `rerun`         - rebuild and run benchmark
#    `clean`         - delete all generated files and folders
#
#  Usage examples:
#  ---------------
#    `make`                       - build and run benchmark
#    `make rerun`                 - rebuild and run benchmark
#    `make SIZE=2048 rerun`       - run with 2048-point FFT
#    `make BATCH=4 rerun`         - run with batch size 4
#    `make clean`                 - delete all generated files and folders
#
###########################################################################################

.NOTPARALLEL:

# current directory
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

# Output directory
OUT  ?= $(THIS_DIR)/_out

# Default parameters
SIZE ?= 1024
BATCH ?= 8
ITERATIONS ?= 100

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

# Build directory
BUILD_DIR := $(OUT)

# Source files
SRC := $(THIS_DIR)/fft.cpp

# Object files
OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cpp=.o)))

# Debug: print paths (uncomment to debug)
# $(info SRC=$(SRC))
# $(info BUILD_DIR=$(BUILD_DIR))
# $(info OBJ=$(OBJ))

# Executable name
EXEC := fft.exe
EFP_DIR       = $(THIS_DIR)/../..
INCLUDE_DIR   = $(THIS_DIR)/../../../../include
## (unused) SRC_DIR / EXAMPLES_DIR removed (was unused in this Makefile)

# Default target (always device)
TARGET := device

# Linkage mode
LINKAGE ?= inline

# verbose mode
VERBOSE_MODE = 0
ifneq (,$(filter $(V) $(VERBOSE) $(VERB), 1))
    VERBOSE_MODE = 1
else ifneq (,$(filter $(V) $(VERBOSE) $(VERB), 2))
    VERBOSE_MODE = 2
else ifneq (,$(filter $(V) $(VERBOSE) $(VERB), 3 y Y yes Yes full Full f F))
    VERBOSE_MODE = 3
endif

ifeq ($(VERBOSE_MODE), 1)
    RULE_DONE = -@echo "fft: $(notdir $@) built"
else
    RULE_DONE = 
endif

# Compiler and flags (always nvcc for GPU)
CXX := nvcc
CXXFLAGS := -std=c++17 -O2 -arch=sm_$(ARCH) -x cu -Xcompiler -Wall,-Wextra
LIBS := 
LINKLIBS := $(QUADMATH_LIB)

INCLUDES := -I$(INCLUDE_DIR)
# Define macros
DEFINES := -DSIZE=$(SIZE) -DBATCH=$(BATCH) -DITERATIONS=$(ITERATIONS)

ifneq (,$(filter $(EMULATION), n NO No no 0 false False FALSE))
	DEFINES += -DNO_EMULATION
endif

ifneq (,$(CALLBACKS))
	DEFINES += -D__ACCURACY__=native -D__EFP64_CALLBACKS__
	ifneq (, $(filter $(CALLBACK_ROUNDING_MODE_MANTISSA), TRUNCATION ROUND_TO_NEAREST IEEE_ROUNDING))
		DEFINES += -D$(CALLBACK_ROUNDING_MODE_MANTISSA) -D__EFP64_CALLBACKS_MANTISSA_BITS__=$(CALLBACKS)
	endif
	ifneq (,$(CALLBACK_ROUNDING_MODE_EXPONENT))
		DEFINES += -DEXPONENT -D__EFP64_CALLBACKS_EXPONENT_BITS__=$(CALLBACKS)
	endif
endif

ifneq (,$(filter $(LINKAGE), static lto))
    LIB_BASE ?= cufp
    LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
	CXXFLAGS += -DCCCL_FPMP_LIB=1 -DCCCL_FPEMU_LIB=1
	ifneq (,$(filter $(LINKAGE), lto))
	    LIBFLAGS += -dlto
		CXXFLAGS += -dlto
	else
		# Static linkage without LTO requires relocatable device code
		CXXFLAGS += -rdc=true
		LIBFLAGS += -rdc=true
	endif
else
    LIB_NAME =
endif

CXXFLAGS  += $(EXTRA_FLAGS) $(ABI_FLAGS)
LIBFLAGS  += $(EXTRA_FLAGS)

# maximum verbose mode with -v flag
ifeq ($(VERBOSE_MODE), 3)
	CXXFLAGS += -v -Xptxas="-v" --keep --keep-dir $(BUILD_DIR)
	LIBFLAGS += -v -Xptxas="-v" --keep --keep-dir $(BUILD_DIR)
endif

.PHONY: all run rerun clean

all: $(BUILD_DIR)/$(EXEC)

ifeq (,$(filter $(T) $(TARGET), host))
run: $(BUILD_DIR)/$(EXEC)
	$(BUILD_DIR)/$(EXEC)
else
run:
endif

rerun:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile run


clean:
	rm -rf $(BUILD_DIR)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

ifeq (,$(filter $(T) $(TARGET), host))
# Compile object files first
$(BUILD_DIR)/%.o: $(THIS_DIR)/%.cpp $(LIB_NAME) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -dc $< -o $@
	$(RULE_DONE)

# Link executable from object files
$(BUILD_DIR)/$(EXEC): $(OBJ) $(LIB_NAME) | $(BUILD_DIR)
	$(CXX) -arch=sm_$(ARCH)  $(OBJ) $(LIBFLAGS) -o $@ $(LIB_NAME) $(LINKLIBS)
	$(RULE_DONE)
else
$(BUILD_DIR)/$(EXEC): 
	@echo "FFT benchmark is not supported on host"
endif

$(LIB_NAME):
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/make/lib.mk TARGET=$(TARGET) OUT=$(OUT) lib

# Help target
help:
	@echo "FFT Benchmark (GPU Only)"
	@echo "======================="
	@echo ""
	@echo "Parameters:"
	@echo "  OUT         = output directory (default: _out)"
	@echo "  LINKAGE     = linkage mode (inline|static|lto, default: inline)"
	@echo "  VERBOSE     = verbose mode (0|1|2|3, default: 0); aliases: V, VERB"
	@echo "  SIZE        = FFT size (default: 1024)"
	@echo "  BATCH       = batch size (default: 8)"
	@echo "  ITERATIONS  = number of iterations (default: 100)"
	@echo "  ARCH        = CUDA architecture (default: 86)"
	@echo "  EMULATION   = set to disable emulation (adds -DNO_EMULATION)"
	@echo "  CALLBACKS   = optional callback bits (see Makefile for details)"
	@echo ""
	@echo "Configuration:"
	@echo "  - Target: GPU Device (CUDA) only"
	@echo "  - FFT Size: 1024"
	@echo "  - Batch Size: 8"
	@echo "  - Iterations: 100"
	@echo ""
	@echo "Timing Methodology:"
	@echo "  - Multiple timing runs (5) for statistical stability"
	@echo "  - Warm-up iterations (100) to stabilize GPU"
	@echo "  - Outlier exclusion for reliable results"
	@echo "  - Longer run times for accurate measurements"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make rerun              # Run benchmark"
	@echo "  make SIZE=2048 rerun    # Smaller FFT size"
	@echo "  make BATCH=16 rerun     # Larger batch size"
	@echo "  make ARCH=75 rerun      # Different CUDA architecture"
	@echo ""
	@echo "Targets:"
	@echo "  all         - build benchmark"
	@echo "  run         - run benchmark"
	@echo "  rerun       - rebuild and run benchmark"
	@echo "  clean       - clean build files"
	@echo "  help        - show this help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  When FP128 support is enabled (_CCCL_FPMP_FP128_ENABLE=1),"
	@echo "  libquadmath is required for linking (provides expq, sinq, etc.)."
	@echo "  libquadmath is auto-detected (x86 only). Override with HAS_QUADMATH=1/0."
	@echo "  Current detection: HOST_ARCH=$(HOST_ARCH), HAS_QUADMATH=$(HAS_QUADMATH)"

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
