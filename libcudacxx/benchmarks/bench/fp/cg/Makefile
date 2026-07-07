###########################################################################################
#
#  File:    Makefile
#  Purpose: Build CG (Conjugate Gradient) benchmark
#  Author:  Andrei Kolesov
#  Date:    2025-01-15
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    Note: this benchmark is GPU-only (CUDA/nvcc).
#    `OUT`       = output directory (default is `_out`)
#    `LINKAGE`   = use library or inline emulation cores (`inline` (default), `static`, `lto`)
#    `VERBOSE`   = verbose mode (`0` (silent), `1` (minimal), `2` (full), `3` (very verbose)); aliases: `V`, `VERB`
#    `SIZE`      = grid size N (default: 4096)
#    `MAX_ITER`  = maximum iterations (default: 3000)
#    `TOLERANCE` = convergence tolerance (default: 1e-4)
#    `ARCH`      = CUDA architecture (default: 86)
#    `FMAD_FLAG` = -fmad= flag (default: false)
#    `EMULATION` = set to disable emulation (adds `-DNO_EMULATION`)
#    `ACCURACY`    = computation accuracy (`def` (default), `low`, `mid`, `high`)
#    `THREADS`   = CUDA threads per block (default: 256)
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
#    `make SIZE=2048 rerun`       - run with 2048x2048 grid
#    `make MAX_ITER=5000 rerun`   - run with 5000 max iterations
#    `make clean`                 - delete all generated files and folders
#
###########################################################################################

.NOTPARALLEL:

# current directory
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

# Output directory
OUT  ?= $(THIS_DIR)/_out

# Default parameters
SIZE ?= 4096
MAX_ITER ?= 3000
TOLERANCE ?= 1e-4

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
SRC := $(THIS_DIR)/cg.cu

# Object files
OBJ := $(addprefix $(BUILD_DIR)/,$(notdir $(SRC:.cu=.o)))

# Executable name
EXEC := cg_fp64_bench.exe
EFP_DIR       = $(THIS_DIR)/../..
INCLUDE_DIR   = $(THIS_DIR)/../../../../include

# Default target (always device)
TARGET := device

# Linkage mode
LINKAGE ?= inline

# -fmad= flag
FMAD_FLAG ?= true

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
    RULE_DONE = -@echo "cg: $(notdir $@) built"
else
    RULE_DONE = 
endif

# Compiler and flags (always nvcc for GPU)
CXX := nvcc
CXXFLAGS := -std=c++17 -O3 -arch=sm_$(ARCH) -fmad=$(FMAD_FLAG) -x cu -Xcompiler -Wall,-Wextra
LIBS :=

INCLUDES := -I$(THIS_DIR) \
           -I$(INCLUDE_DIR)

# Define macros
DEFINES := -DSIZE=$(SIZE) -DMAX_ITER=$(MAX_ITER) -DTOLERANCE=$(TOLERANCE)

ifneq (,$(filter $(EMULATION), n NO No no 0 false False FALSE))
	DEFINES += -DNO_EMULATION
endif

# ACC: short alias for ACCURACY (e.g. ACC=mid); ACCURACY takes precedence
ifdef ACC
ACCURACY ?= $(ACC)
endif

# Accuracy parameter: def, low, mid, high
ifneq (,$(ACCURACY))
	ifneq (,$(filter $(ACCURACY), def low mid high))
		DEFINES += -D__METHOD__=$(ACCURACY)
	endif
else
	# Default to def if no accuracy specified
	DEFINES += -D__METHOD__=def
endif

ifneq (,$(THREADS))
	DEFINES += -DDEFAULT_THREADS_PER_BLOCK=$(THREADS)
endif

ifneq (,$(filter $(LINKAGE), static lto))
    LIB_BASE ?= cufp
    LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
	CXXFLAGS += -DCCCL_FPMP_LIB=1 -DCCCL_FPEMU_LIB=1
	LIBS += $(LIB_NAME)
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

.PHONY: all run rerun clean debug help

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
$(BUILD_DIR)/%.o: $(THIS_DIR)/%.cu $(LIB_NAME) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -dc $< -o $@
	$(RULE_DONE)

# Link executable from object files
$(BUILD_DIR)/$(EXEC): $(OBJ) $(LIB_NAME) | $(BUILD_DIR)
	$(CXX) -arch=sm_$(ARCH) $(OBJ) $(LIBFLAGS) -o $@ $(LIBS)
	$(RULE_DONE)
else
$(BUILD_DIR)/$(EXEC): 
	@echo "CG benchmark is not supported on host"
endif

$(LIB_NAME):
	$(MAKE) --no-print-directory -f $(EFP_DIR)/make/lib.mk TARGET=$(TARGET) OUT=$(OUT) lib

# Debug target to show what files are being generated
debug:
	@echo "Source files: $(SRC)"
	@echo "Object files: $(OBJ)"
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Library name: $(LIB_NAME)"

# Help target
help:
	@echo "CG (Conjugate Gradient) Benchmark (GPU Only)"
	@echo "============================================"
	@echo ""
	@echo "Parameters:"
	@echo "  OUT         = output directory (default: _out)"
	@echo "  LINKAGE     = linkage mode (inline|static|lto, default: inline)"
	@echo "  VERBOSE     = verbose mode (0|1|2|3, default: 0); aliases: V, VERB"
	@echo "  SIZE        = grid size N (default: 4096)"
	@echo "  MAX_ITER    = maximum iterations (default: 3000)"
	@echo "  TOLERANCE   = convergence tolerance (default: 1e-4)"
	@echo "  ARCH        = CUDA architecture (default: 86)"
	@echo "  FMAD_FLAG   = -fmad= flag (default: false)"
	@echo "  EMULATION   = set to disable emulation (adds -DNO_EMULATION)"
	@echo "  ACCURACY      = computation accuracy (def (default), low, mid, high) [alias: ACC]"
	@echo "  THREADS     = CUDA threads per block (default: 256)"
	@echo ""
	@echo "Configuration:"
	@echo "  - Target: GPU Device (CUDA) only"
	@echo "  - Grid Size: N x N (default: 4096 x 4096)"
	@echo "  - Max Iterations: 3000"
	@echo "  - Convergence Tolerance: 1e-4"
	@echo "  - Solves 2D Laplacian system A*x = b"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make rerun                    # Run benchmark with default parameters"
	@echo "  make SIZE=2048 rerun          # Smaller grid size (2048x2048)"
	@echo "  make MAX_ITER=5000 rerun      # More iterations"
	@echo "  make TOLERANCE=1e-6 rerun     # Stricter tolerance"
	@echo "  make ACCURACY=high rerun        # Use high accuracy"
	@echo "  make ARCH=75 rerun            # Different CUDA architecture"
	@echo ""
	@echo "Targets:"
	@echo "  all         - build benchmark"
	@echo "  run         - run benchmark"
	@echo "  rerun       - rebuild and run benchmark"
	@echo "  clean       - clean build files"
	@echo "  debug       - show build configuration"
	@echo "  help        - show this help" 

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
