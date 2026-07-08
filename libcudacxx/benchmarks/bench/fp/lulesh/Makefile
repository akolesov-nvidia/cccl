###########################################################################################
#
#  File:    Makefile
#  Purpose: Build LULESH benchmark
#  Author:  Andrei Kolesov
#  Date:    2025-01-15
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `TARGET`  = target device (`device` (default), `host`)
#    `OUT`     = output directory (default is `_out`)
#    `LINKAGE` = use library or inline emulation cores (`inline` (default), `static`, `lto`)
#    `VERBOSE` = verbose mode (`0` (silent), `1` (minimal), `2` (full))
#    `SIZE`    = problem size (default: 16)
#    `EMULATION` = emulation  (y (default), n)
#    `ACCURACY` = computation accuracy (def (default), accurate, fast)
#    Legacy: `ACC` = accuracy and `RANGE` = range still supported for backward compatibility
#    `ITERATIONS` = number of iterations (default: 500)
#    `EXTRA_FLAGS` = extra compiler/linker flags
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
#    `make SIZE=100 rerun`        - run with problem size 100
#    `make clean`                 - delete all generated files and folders
#
###########################################################################################

.NOTPARALLEL:

# current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Dev-only library build (lib/make/lib.mk + lib/src/ + lib/include/) lives in the sibling cccl-fpdev
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

# Output directory
OUT  ?= $(THIS_DIR)/_out

# Build directory
BUILD_DIR := $(OUT)

# Default parameters
SIZE ?= 16

# CUDA sm_xx: use ARCH if set; else first GPU (nvidia-smi); else 86.
ifndef ARCH
    _fpmp_cuda_arch_cap := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
    ifeq ($(strip $(_fpmp_cuda_arch_cap)),)
        ARCH := 86
    else
        ARCH := $(_fpmp_cuda_arch_cap)
    endif
endif

ITERATIONS ?= 500

# Source files
SRC := $(wildcard $(THIS_DIR)/*.cu)
SRC_OBJ := $(patsubst $(THIS_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC))

# Executable name
EXEC := lulesh.out
MFP_DIR       = $(THIS_DIR)/../..
INCLUDE_DIR   = $(MFP_DIR)/include
SRC_DIR       = $(MFP_DIR)/src

# Default target (always device)
TARGET := device

# Linkage mode
LINKAGE ?= inline

# -fmad= flag
FMAD_FLAG ?= false

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
    RULE_DONE = -@echo "lulesh: $(notdir $@) built"
else
    RULE_DONE = 
endif

# Compiler and flags (always nvcc for GPU)
CXX := nvcc
CXXFLAGS         := -O3 --disable-warnings -arch=sm_$(ARCH) -fmad=$(FMAD_FLAG) -x cu
LIBS := 
LIBFLAGS := -arch=sm_$(ARCH) -fmad=$(FMAD_FLAG)

INCLUDES := -I$(THIS_DIR) \
           -I$(INCLUDE_DIR)

# Define macros
DEFINES := 

# Data type selection: DATATYPE = fpemu (default), fpmp, double, float
DATATYPE ?= fpemu
ifeq ($(DATATYPE),double)
	DEFINES += -DNO_EMULATION
else ifeq ($(DATATYPE),float)
	DEFINES += -DUSE_FLOAT
else ifeq ($(DATATYPE),fpmp)
	DEFINES += -DUSE_FPMP
else ifneq (,$(filter $(EMULATION), n NO No no 0 false False FALSE))
	DEFINES += -DNO_EMULATION
endif

# Support both old ACC/RANGE and new ACCURACY parameters
# Map old combinations to new accuracy parameter
ifneq (,$(ACCURACY))
	# New ACCURACY parameter: accurate, def, fast
	ifneq (,$(filter $(ACCURACY), accurate def fast))
		DEFINES += -D__METHOD__=$(ACCURACY)
	endif
else
	# Legacy support: map ACC/RANGE to ACCURACY
	ifeq ($(ACC),cr)
		DEFINES += -D__METHOD__=accurate
	else ifeq ($(ACC),ha)
		DEFINES += -D__METHOD__=def
	else ifeq ($(ACC),la)
		DEFINES += -D__METHOD__=fast
	else ifeq ($(ACC),def)
		DEFINES += -D__METHOD__=def
	else ifeq ($(ACC),accurate)
		DEFINES += -D__METHOD__=accurate
	else ifeq ($(ACC),fast)
		DEFINES += -D__METHOD__=fast
	else
		# Default to def if no accuracy specified
		DEFINES += -D__METHOD__=def
	endif
endif

ifeq (,$(filter $(ITERATIONS), auto Auto AUTO n N No NO 0))
	DEFINES += -DITERATION_COUNT=$(ITERATIONS)
endif

# Only link the emulation library for types that need it (fpemu, fpmp)
NEEDS_LIB := $(filter fpemu fpmp,$(DATATYPE))
ifeq ($(DATATYPE),)
    NEEDS_LIB := yes
endif

ifneq (,$(NEEDS_LIB))
  ifneq (,$(filter $(LINKAGE), static lto))
    LIB_BASE ?= cufp
    LIB_NAME = $(OUT)/lib/lib$(LIB_BASE).a
	CXXFLAGS += -DCCCL_FPEMU_LIB=1 -DCCCL_FPMP_LIB=1
	LIBS += $(LIB_NAME)
	ifneq (,$(filter $(LINKAGE), lto))
	  CXXFLAGS += -dlto
	  LIBFLAGS += -dlto	
	else ifneq (,$(filter $(LINKAGE), static))
	  CXXFLAGS += -rdc=true
	  LIBFLAGS += -rdc=true
	endif
  else
    LIB_NAME =
  endif
else
    LIB_NAME =
endif

CXXFLAGS += $(EXTRA_FLAGS) $(ABI_FLAGS)
LIBFLAGS += $(EXTRA_FLAGS)

# maximum verbose mode with -v flag
ifeq ($(VERBOSE_MODE), 3)
	CXXFLAGS += -v
	LIBFLAGS += -v
endif

.PHONY: all run rerun clean debug help build _build compare

all: run

build:
	@START=$$(date +%s%N) ; \
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile _build ; \
	END=$$(date +%s%N) ; \
	ELAPSED=$$(( (END - START) / 1000000 )) ; \
	echo "----------------------------------------" ; \
	echo "Total build: $${ELAPSED} ms ($$(( ELAPSED / 1000 )).$$(( ELAPSED % 1000 / 100 ))s)"

_build: $(BUILD_DIR)/$(EXEC)

ifeq (,$(filter $(T) $(TARGET), host))
run: $(BUILD_DIR)/$(EXEC)
	@echo "Running LULESH benchmark..."
	$(BUILD_DIR)/$(EXEC) -s $(SIZE)
else
run:
endif

rerun:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile compare

clean:
	rm -rf $(BUILD_DIR)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

ifeq (,$(filter $(T) $(TARGET), host))
# Compile source files to object files first
$(BUILD_DIR)/%.o: $(THIS_DIR)/%.cu $(LIB_NAME) | $(BUILD_DIR)
	@START=$$(date +%s%N) ; \
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFINES) -dc $< -o $@ ; \
	END=$$(date +%s%N) ; \
	ELAPSED=$$(( (END - START) / 1000000 )) ; \
	echo "Compile $(notdir $<): $${ELAPSED} ms ($$(( ELAPSED / 1000 )).$$(( ELAPSED % 1000 / 100 ))s)"

# Then link object files to create executable
$(BUILD_DIR)/$(EXEC): $(SRC_OBJ) $(LIB_NAME) | $(BUILD_DIR)
	@START=$$(date +%s%N) ; \
	$(CXX) $(SRC_OBJ) $(LIBFLAGS) -o $@ $(LIBS) ; \
	END=$$(date +%s%N) ; \
	ELAPSED=$$(( (END - START) / 1000000 )) ; \
	echo "Link    $(notdir $@): $${ELAPSED} ms ($$(( ELAPSED / 1000 )).$$(( ELAPSED % 1000 / 100 ))s)"
else
$(BUILD_DIR)/$(EXEC): 
	@echo "LULESH benchmark is not supported on host"
endif

$(LIB_NAME):
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/lib/make/lib.mk TARGET=$(TARGET) OUT=$(OUT) EXTRA_FLAGS="-fmad=$(FMAD_FLAG) $(EXTRA_FLAGS)" lib

# Help target
help:
	@echo "LULESH Benchmark (GPU Only)"
	@echo "==========================="
	@echo ""
	@echo "Parameters:"
	@echo "  DATATYPE    = precision type (fpemu (default), fpmp, double, float)"
	@echo "  OUT         = output directory (default: _out)"
	@echo "  LINKAGE     = linkage mode (inline|static|lto, default: inline)"
	@echo "  VERBOSE     = verbose mode (0|1|2, default: 0)"
	@echo "  SIZE        = problem size (default: 16)"
	@echo "  ARCH        = CUDA architecture (default: 86)"
	@echo "  ABI         = custom ABI: ABI=x,y (forces LINKAGE=static)"
	@echo "  FMAD_FLAG   = -fmad= flag (default: false)"
	@echo "  ACCURACY      = computation accuracy (def (default), accurate, fast) [alias: ACC]"
	@echo "                Legacy: ACC=ha/la/cr still supported"
	@echo "  ITERATIONS  = number of iterations (default: 500)"
	@echo "  EXTRA_FLAGS = extra compiler flags"
	@echo ""
	@echo "Targets:"
	@echo "  all         - build and run benchmark"
	@echo "  build       - build benchmark"
	@echo "  run         - run benchmark"
	@echo "  rerun       - rebuild and run benchmark"
	@echo "  compare     - build & run all 4 types (double/float/fpemu/fpmp) and compare"
	@echo "  clean       - clean build files"
	@echo "  help        - show this help" 

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif

# Compare all data types: double, float, fpemu, fpmp
compare:
	@echo "=========================================="
	@echo " LULESH Precision Comparison"
	@echo " SIZE=$(SIZE)  ACCURACY=$(or $(ACCURACY),def)  LINKAGE=$(LINKAGE)"
	@echo "=========================================="
	@COMPARE_OUT=$(OUT)/compare; \
	mkdir -p $$COMPARE_OUT; \
	for dtype in double float fpemu fpmp; do \
		echo ""; \
		echo "---------- $$dtype ----------"; \
		$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile \
			OUT=$$COMPARE_OUT/$$dtype DATATYPE=$$dtype \
			SIZE=$(SIZE) ACCURACY=$(or $(ACCURACY),def) LINKAGE=$(LINKAGE) \
			clean 2>&1; \
		$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile \
			OUT=$$COMPARE_OUT/$$dtype DATATYPE=$$dtype \
			SIZE=$(SIZE) ACCURACY=$(or $(ACCURACY),def) LINKAGE=$(LINKAGE) \
			build 2>&1 | tee $$COMPARE_OUT/$$dtype.log; \
		$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile \
			OUT=$$COMPARE_OUT/$$dtype DATATYPE=$$dtype \
			SIZE=$(SIZE) ACCURACY=$(or $(ACCURACY),def) LINKAGE=$(LINKAGE) \
			run 2>&1 | tee -a $$COMPARE_OUT/$$dtype.log; \
	done; \
	echo ""; \
	echo "=========================================="; \
	echo " Summary"; \
	echo "=========================================="; \
	printf "%-10s %12s %10s %12s %14s %12s\n" "Type" "Compile(s)" "Binary" "Run(s)" "FOM(z/s)" "MaxRelDiff"; \
	echo "----------+------------+----------+------------+--------------+------------"; \
	for dtype in double float fpemu fpmp; do \
		LOG=$$COMPARE_OUT/$$dtype.log; \
		COMPILE=$$(grep "Total build:" $$LOG 2>/dev/null | sed 's/.*(\([0-9.]*\)s)/\1/'); \
		ELAPSED=$$(grep "Elapsed time" $$LOG 2>/dev/null | sed 's/[^0-9.]//g' | sed 's/^[[:space:]]*//'); \
		FOM=$$(grep "FOM " $$LOG 2>/dev/null | sed 's/.*=  *\([0-9.]*\).*/\1/'); \
		RELDIFF=$$(grep "MaxRelDiff" $$LOG 2>/dev/null | sed 's/.*=  *\([0-9eE.+-]*\).*/\1/'); \
		BINFILE=$$COMPARE_OUT/$$dtype/$(EXEC); \
		if [ -f "$$BINFILE" ]; then \
			BINSIZE=$$(du -h "$$BINFILE" | cut -f1); \
		else \
			BINSIZE="-"; \
		fi; \
		printf "%-10s %12s %10s %12s %14s %12s\n" "$$dtype" "$${COMPILE:--}" "$$BINSIZE" "$${ELAPSED:--}" "$${FOM:--}" "$${RELDIFF:--}"; \
	done; \
	echo "=========================================="; \
	echo ""; \
	echo "Notes:"; \
	echo "  double = fp64 double,  float = fp32 float"; \
	echo "  fpemu  = emulated double (full double range, near-double precision)"; \
	echo "  fpmp   = float-float (fpmp2<float>, same exponent range as float, ~2x mantissa)"; \
	echo ""; \
	HAVE_FAIL=0; \
	for dtype in double fpemu fpmp; do \
		LOG=$$COMPARE_OUT/$$dtype.log; \
		if grep -q "(NOT PASSED)" $$LOG 2>/dev/null; then \
			echo "ERROR: $$dtype precision did not pass accuracy check"; \
			HAVE_FAIL=1; \
		fi; \
	done; \
	if [ $$HAVE_FAIL -eq 0 ]; then \
		echo "Result: OK (float excluded from accuracy check)"; \
	else \
		echo "Result: NOT PASSED"; \
		exit 1; \
	fi

# Debug target to show what files are being generated
debug:
	@echo "Source files: $(SRC)"
	@echo "Object files: $(SRC_OBJ)"
	@echo "Build directory: $(BUILD_DIR)"
	@echo "Library name: $(LIB_NAME)"
