###########################################################################################
#
#  File:    Makefile
#  Purpose: Build and run double/fp32mp2 conversion throughput benchmark
#
#                       USER GUIDE
#                       ==========
#
#  Benchmarks both conversion directions:
#    double -> fp32mp2_t  (CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP)
#    fp32mp2_t -> double  (CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE)
#
#  For each direction, standard (FP64) and optimized (INT32 bit ops) variants
#  are compiled into the same binary via separate compilation and compared
#  in a single run.
#
#  Makefile parameters:
#  --------------------
#    `TARGET`      = target device (`device` (default), `host`)
#    `ARCH`        = CUDA architecture (default: 86 for RTX 30xx/A100)
#    `REPS`        = conversions per thread (default: 4096 GPU / 1024 CPU)
#    `DEPTH`       = parallel conversion chains per thread (default: 8)
#                    Higher values increase register pressure to test
#                    whether optimized path causes spills.
#    `UNROLL`      = loop unroll factor (default: 8)
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
#    `help`          - show help message
#
#  Usage examples:
#  ---------------
#    `make`                              - build and run comparison
#    `make rerun`                       - clean, rebuild, and run
#    `make T=host rerun`               - compare on CPU
#    `make DEPTH=8 rerun`              - test with high register pressure
#    `make REPS=8192 rerun`            - compare with more work per thread
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
UNROLL      ?= 8

ifneq (,$(filter $(T) $(TARGET), host))
    REPS       ?= 1024
    NUM_BLOCKS ?= 32
	DEPTH      ?= 1
else
    REPS       ?= 4096
    NUM_BLOCKS ?= 2048
    DEPTH      ?= 8
endif

# Current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Dev-only library build (lib/make/lib.mk + lib/src/ + lib/include/) lives in the sibling cccl-fpdev
# repo; overridable (the cccl-fpdev root Makefile passes it explicitly).
FPDEV_DIR ?= $(abspath $(THIS_DIR)/../../../../../../cccl-fpdev)

# libquadmath availability — only x86 GCC distributions ship it. d2ff itself
# never touches __float128, but linking the precompiled libcufp (LINKAGE=static|
# lto) pulls in its FP128 host fallback (expq/logq/powq/...), so the quad lib is
# required on host when linking the library. Override with HAS_QUADMATH=1/0.
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
MFP_DIR     = $(THIS_DIR)/../..

LINKAGE ?= inline

# Output directory
OUT ?= $(THIS_DIR)/_out

# Source files — double -> fp32mp2 (d2ff)
DRV_SRC = $(THIS_DIR)/d2ff.cpp
STD_SRC = $(THIS_DIR)/d2ff_std.cpp
OPT_SRC = $(THIS_DIR)/d2ff_opt.cpp
KRN_HDR = $(THIS_DIR)/d2ff_kernel.hpp
CMN_HDR = $(THIS_DIR)/d2ff_common.hpp

# Source files — fp32mp2 -> double (ff2d)
FF2D_STD_SRC = $(THIS_DIR)/ff2d_std.cpp
FF2D_OPT_SRC = $(THIS_DIR)/ff2d_opt.cpp
FF2D_KRN_HDR = $(THIS_DIR)/ff2d_kernel.hpp

# Object files
DRV_OBJ = $(OUT)/d2ff.o
STD_OBJ = $(OUT)/d2ff_std.o
OPT_OBJ = $(OUT)/d2ff_opt.o
FF2D_STD_OBJ = $(OUT)/ff2d_std.o
FF2D_OPT_OBJ = $(OUT)/ff2d_opt.o
ALL_OBJS = $(DRV_OBJ) $(STD_OBJ) $(OPT_OBJ) $(FF2D_STD_OBJ) $(FF2D_OPT_OBJ)

# Executable name
EXE = $(OUT)/d2ff.exe

# Compiler settings
ifneq (,$(filter $(T) $(TARGET), host))
    TARGET_SUFFIX = $(HOST_SUFFIX)
    CXX = g++
    COMPILEFLAGS = -std=c++20 -O3 -Wall -Wextra -Wno-unknown-pragmas -c
    LINKFLAGS = -std=c++20 -O3 -Wall -Wextra
    DEFINES  =
    LINKLIBS =
else
    TARGET_SUFFIX = $(DEVICE_SUFFIX)
    CXX = nvcc
    COMPILEFLAGS = -std=c++20 -arch=sm_$(ARCH) -O3 -x cu --extended-lambda -c
    LINKFLAGS = -std=c++20 -arch=sm_$(ARCH) -O3
    DEFINES  =
    LINKLIBS =
endif

ifneq (,$(filter $(LINKAGE), static lto))
    LIB_BASE ?= cufp
    LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
    COMPILEFLAGS += -DCCCL_FPMP_LIB=1
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
    # The host library carries the FP128 quad fallback; link libquadmath.
    ifneq (,$(filter $(T) $(TARGET), host))
        LINKLIBS += $(QUADMATH_LIB)
    endif
else
    LIB_NAME =
endif

# d2ff measures both the standard (FP64-cast) and optimized (integer-only)
# conversion paths in the same binary by hardcoding CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP
# / CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE in the per-TU sources (d2ff_*/ff2d_*.cpp).
# An external -D from EXTRA_FLAGS (e.g. propagated from the root Makefile via
# FP2MP / MP2FP) would either silently swap the std TU to the opt path, or
# clash with the opt TU's own #define. Strip those flags here to keep the
# benchmark's A/B comparison meaningful regardless of root-level settings.
EXTRA_FLAGS_FILTERED := $(filter-out -DCCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=% \
                                     -DCCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE=%, \
                                     $(EXTRA_FLAGS))
COMPILEFLAGS += $(EXTRA_FLAGS_FILTERED) $(ABI_FLAGS)
LINKFLAGS    += $(EXTRA_FLAGS_FILTERED)

# Add compile-time definitions
DEFINES += -DREPS=$(REPS)
DEFINES += -DUNROLL=$(UNROLL)
DEFINES += -DNUM_ITERATIONS=$(NUM_ITER)
DEFINES += -DTHREADS_PER_BLOCK=$(THREADS)
DEFINES += -DNUM_BLOCKS=$(NUM_BLOCKS)
DEFINES += -DDEPTH=$(DEPTH)

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
    RULE_DONE = -@echo "d2ff: $(notdir $@) built"
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

# Compile driver
$(DRV_OBJ): $(DRV_SRC) $(CMN_HDR)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

# Compile standard conversion kernel
$(STD_OBJ): $(STD_SRC) $(KRN_HDR) $(CMN_HDR)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

# Compile optimized conversion kernel (d2ff)
$(OPT_OBJ): $(OPT_SRC) $(KRN_HDR) $(CMN_HDR)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

# Compile standard reverse conversion kernel (ff2d)
$(FF2D_STD_OBJ): $(FF2D_STD_SRC) $(FF2D_KRN_HDR) $(CMN_HDR)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

# Compile optimized reverse conversion kernel (ff2d)
$(FF2D_OPT_OBJ): $(FF2D_OPT_SRC) $(FF2D_KRN_HDR) $(CMN_HDR)
	$(Q)mkdir -p $(OUT)
	$(Q)$(CXX) $(COMPILEFLAGS) $(DEFINES) $(INCLUDES) -o $@ $<
	$(RULE_DONE)

# Link all objects
$(EXE): $(ALL_OBJS) $(LIB_NAME)
	$(Q)$(CXX) $(LINKFLAGS) -o $@ $(ALL_OBJS) $(LIB_NAME) $(LINKLIBS)
	$(RULE_DONE)

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
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/lib/make/lib.mk TARGET=$(TARGET) OUT=$(OUT) lib

# Show kernel register usage and object sizes (GPU only, recompiles with --resource-usage)
info:
	@echo "=== Standard conversion kernel (d2ff, depth=$(DEPTH)) ==="
	@start=$$(date +%s%N) ; \
	 $(CXX) $(COMPILEFLAGS) --resource-usage $(DEFINES) $(INCLUDES) -o $(OUT)/d2ff_std_info.o $(STD_SRC) 2>&1 | c++filt | grep -E "registers|spill|cmem|smem|Function" ; \
	 end=$$(date +%s%N) ; \
	 ms=$$(( (end - start) / 1000000 )) ; \
	 ls -la $(OUT)/d2ff_std_info.o | awk '{printf "  Object size: %s bytes\n", $$5}' ; \
	 echo "  Compile time: $${ms} ms"
	@echo ""
	@echo "=== Optimized conversion kernel (d2ff, depth=$(DEPTH)) ==="
	@start=$$(date +%s%N) ; \
	 $(CXX) $(COMPILEFLAGS) --resource-usage $(DEFINES) $(INCLUDES) -o $(OUT)/d2ff_opt_info.o $(OPT_SRC) 2>&1 | c++filt | grep -E "registers|spill|cmem|smem|Function" ; \
	 end=$$(date +%s%N) ; \
	 ms=$$(( (end - start) / 1000000 )) ; \
	 ls -la $(OUT)/d2ff_opt_info.o | awk '{printf "  Object size: %s bytes\n", $$5}' ; \
	 echo "  Compile time: $${ms} ms"
	@echo ""
	@echo "=== Standard reverse kernel (ff2d, depth=$(DEPTH)) ==="
	@start=$$(date +%s%N) ; \
	 $(CXX) $(COMPILEFLAGS) --resource-usage $(DEFINES) $(INCLUDES) -o $(OUT)/ff2d_std_info.o $(FF2D_STD_SRC) 2>&1 | c++filt | grep -E "registers|spill|cmem|smem|Function" ; \
	 end=$$(date +%s%N) ; \
	 ms=$$(( (end - start) / 1000000 )) ; \
	 ls -la $(OUT)/ff2d_std_info.o | awk '{printf "  Object size: %s bytes\n", $$5}' ; \
	 echo "  Compile time: $${ms} ms"
	@echo ""
	@echo "=== Optimized reverse kernel (ff2d, depth=$(DEPTH)) ==="
	@start=$$(date +%s%N) ; \
	 $(CXX) $(COMPILEFLAGS) --resource-usage $(DEFINES) $(INCLUDES) -o $(OUT)/ff2d_opt_info.o $(FF2D_OPT_SRC) 2>&1 | c++filt | grep -E "registers|spill|cmem|smem|Function" ; \
	 end=$$(date +%s%N) ; \
	 ms=$$(( (end - start) / 1000000 )) ; \
	 ls -la $(OUT)/ff2d_opt_info.o | awk '{printf "  Object size: %s bytes\n", $$5}' ; \
	 echo "  Compile time: $${ms} ms"
	@rm -f $(OUT)/d2ff_std_info.o $(OUT)/d2ff_opt_info.o $(OUT)/ff2d_std_info.o $(OUT)/ff2d_opt_info.o

# Clean up generated files
clean:
	$(Q)rm -rf $(OUT)

# Help target
help:
	@echo "Double-to-fp32mp2 Conversion Benchmark"
	@echo "======================================="
	@echo ""
	@echo "Both conversion variants (standard FP64 cast and optimized"
	@echo "INT32 bit ops) are compiled into the same binary and compared"
	@echo "in a single run."
	@echo ""
	@echo "Parameters:"
	@echo "  TARGET      = device|host (default: device)"
	@echo "  ARCH        = CUDA architecture (default: 86)"
	@echo "  REPS        = conversions per thread (default: 4096 GPU / 1024 CPU)"
	@echo "  DEPTH       = parallel chains per thread (default: 8, try 1/2/4/8/16/32)"
	@echo "  UNROLL      = loop unroll factor (default: 8)"
	@echo "  NUM_ITER    = timing iterations (default: 10)"
	@echo "  THREADS     = threads per block (default: 256)"
	@echo "  NUM_BLOCKS  = CUDA blocks (default: 2048)"
	@echo "  LINKAGE     = inline|static|lto (default: inline)"
	@echo ""
	@echo "Targets:"
	@echo "  all      - build and run (default)"
	@echo "  build    - build only"
	@echo "  run      - run (builds if needed)"
	@echo "  rebuild  - clean and build"
	@echo "  rerun    - clean, build, and run"
	@echo "  info     - show kernel register usage (cuobjdump)"
	@echo "  clean    - remove generated files"
	@echo "  help     - show this message"

.PHONY: all build run rebuild rerun info clean help

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
