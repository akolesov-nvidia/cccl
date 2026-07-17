###########################################################################################
#
#  File:    Makefile
#  Purpose: Build and run FP64 Tool Runtime Multi-Unit Test
#  
#  This benchmark tests that fp64_tool.h can be safely included in multiple
#  compilation units without causing linker errors or symbol conflicts.
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `TARGET`      = target device (`device` (default), `host`)
#    `ARCH`        = CUDA architecture (default: 86 for RTX 30xx/A100)
#    `OUT`         = output directory (default: _out)
#    `VERBOSE`     = verbose mode (`0` (silent), `1` (minimal), `2` (full))
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
#    `make`                    - build and run on GPU
#    `make TARGET=host`        - build and run on CPU
#    `make ARCH=75 rebuild`    - rebuild for Turing architecture
#    `make clean`              - clean up generated files
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
# CCCL in-tree headers: bench/fp/<name> -> libcudacxx/include is four levels up.
INCLUDE_DIR = $(THIS_DIR)/../../../../include
SRC_MAIN    = $(THIS_DIR)/runtime_multi_unit.cpp
SRC_AUX     = $(THIS_DIR)/runtime_multi_unit_aux.cpp
MFP_DIR     = $(THIS_DIR)/../..

# Output directory
OUT ?= $(THIS_DIR)/_out

# Executable and object files
EXE = $(OUT)/runtime_multi_unit.exe
OBJ_MAIN = $(OUT)/runtime_multi_unit.o
OBJ_AUX = $(OUT)/runtime_multi_unit_aux.o

# Compiler settings
ifneq (,$(filter $(T) $(TARGET), host))
    # Host (CPU) settings — separate compile+link (multi-unit test)
    CXX = g++
    COMPILEFLAGS = -std=c++17 -O2 -Wall -Wextra -c
    LINKFLAGS = -std=c++17 -O2 -Wall -Wextra
    DEFINES  =
    LINKLIBS = 
else
    # CUDA (GPU) settings - use relocatable device code for multi-unit linking
    CXX = nvcc
    COMPILEFLAGS = -std=c++17 -arch=sm_$(ARCH) -O2 -x cu -rdc=true -c
    LINKFLAGS = -std=c++17 -arch=sm_$(ARCH) -O2 -rdc=true
    DEFINES  =
    LINKLIBS = 
endif

COMPILEFLAGS += $(EXTRA_FLAGS) $(ABI_FLAGS)

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
LINKFLAGS    += $(EXTRA_FLAGS)

# Include directories
INCLUDES = -I$(INCLUDE_DIR)

# Verbose mode
VERBOSE_MODE = 1
ifneq (,$(filter 0 n N no No,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 0
else ifneq (,$(filter 1 y Y yes Yes,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 1
else ifneq (,$(filter 2,$(V) $(VERBOSE) $(VERB)))
    VERBOSE_MODE = 2
endif

# Verbose mode
ifeq ($(VERBOSE_MODE), 1)
    RULE_DONE = -@echo "benchmark: $(notdir $@) built"
else
    RULE_DONE = 
endif

ifeq ($(VERBOSE_MODE), 0)
	Q = @
endif

# Default target
all: run

# Build the executable
build: $(EXE)

ifneq (,$(filter $(T) $(TARGET), host))
# Host build: compile both units separately, then link
$(OBJ_MAIN): $(SRC_MAIN)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

$(OBJ_AUX): $(SRC_AUX)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

$(EXE): $(OBJ_MAIN) $(OBJ_AUX)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $^ $(LINKLIBS)
	$(RULE_DONE)
else
# CUDA build: compile both units separately with -dc, then link
$(OBJ_MAIN): $(SRC_MAIN)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

$(OBJ_AUX): $(SRC_AUX)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

# Link with relocatable device code
$(EXE): $(OBJ_MAIN) $(OBJ_AUX)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $^ $(LINKLIBS)
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
	@echo "FP64 Tool Runtime Multi-Unit Test Makefile"
	@echo "==========================================="
	@echo ""
	@echo "Parameters:"
	@echo "  TARGET      = device|host (default: device)"
	@echo "  ARCH        = CUDA architecture (default: 86)"
	@echo "  OUT         = output directory (default: _out)"
	@echo "  VERBOSE     = 0|1|2 (default: 1); aliases: V, VERB"
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

.PHONY: all build run rebuild rerun clean help

# Silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
