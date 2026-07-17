###########################################################################################
#
#  File:    Makefile
#  Purpose: Convenience Makefile for building and running the ts test suite
#  Author:  Andrei Kolesov
#  Date:    2025-04-11
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `TARGET`   = target device (`device` (default), `host`)
#    `FUNC`   = function to evaluate (`fma`, `add`, `mul`, `sub`, `div`, `mad`, ...)
#    `API`    = emulated API to exercise; list allowed (`packed`, `unpacked`) (default is `packed`)
#    `ROUNDING` = rounding mode (`rn`, `rz`, `ru`, `rd`)
#    `ACCURACY`   = accuracy level (`high`, `mid`, `low`; `def` == high)
#    `ARCH` = CUDA sm_xx (unset: nvidia-smi, else 86)
#    `LINKAGE` = linkage mode (`inline` (default), `lto`, 'static')
#    `THREADS`  = number of threads per block (default is 1024)
#    `REPEATS`  = number of repetitions 
#    `ITERATIONS`    = number of iterations 
#    `ACCURACY_LEN`      = length of the accuracy array 
#    `LATENCY_LEN`      = length of the latency timing array 
#    `THROUGHPUT_LEN`      = length of the throughput timing array 
#    `NAN_PAYLOAD` = use nan payload (default is 0)
#    `MEAN`        = mean (default is 0)
#    `STDDEV`      = standard deviation (default is 1)
#    `USE_BUILTINS` = use builtins for compound functions (default is 1)
#    `PRINT_LIMIT` = print limit (default is 10)
#    `PRINT`       = choose what to print into log file ('ok', 'fail', 'warn', 'all', 'none', default is 'fail')
#    `DEBUG`       = debug mode, custom printouts to console (default is 0)
#    `CONSOLE`     = console type ('stdout', 'stderr', 'file', 'null', default is 'stdout')
#    `TIMING`      = run timing (default is 0)
#    `PACKED_VIA_UNPACKED` = route packed API via unpacked cores for testing (default is 0)
#    `DATASET`     = dataset type for total timing report (default is gauss)
#    `TOTAL_TIMING` = dataset type for total timing report (default is normal)
#    `A1`, `A2`, `A3` = first, second and third input arguments (no by default)
#    `OUT`     = output directory (default is `_out`)
#    `VERBOSE` = verbose mode (`0` (silent), `1` (minimal), `2` (full))
#
#  Build targets:
#  --------------
#    `all` (default) - build object files and run timing
#    `run`           - build object files and run timing
#    `rerun`         - rebuild object files and run timing
#    `clean`         - delete all generated files and folders
#
#  Usage examples:
#  ---------------
#    `make`                       - build object for exp function and double type and run timing on device
#    `make rerun`                 - rebuild object for exp function and double type and run timing on device
#    `make FUNC=dadd rerun`     - rebuild object for dadd function and double type and run timing on device
#    `make TARGET=host rerun`     - rebuild object for exp function and double type and run timing on host
#    `make clean`                 - delete all generated files and folders
#
###########################################################################################

# Target device
TARGET  ?= device
# CUDA sm_xx: use ARCH if set; else first GPU (nvidia-smi); else 86.
ifndef ARCH
    _fpmp_cuda_arch_cap := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ')
    ifeq ($(strip $(_fpmp_cuda_arch_cap)),)
        ARCH := 86
    else
        ARCH := $(_fpmp_cuda_arch_cap)
    endif
endif
# do not check nan payload by default
NAN_PAYLOAD ?= n
# do not run timing by default
TIMING ?= n
# Linkage mode
LINKAGE ?= inline
# Generate assembly
ASM ?= no
# target suffix: distinguishes host vs device output filenames
DEVICE_SUFFIX ?=
HOST_SUFFIX   ?= _host

# current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# Dev-only library build (lib/make/lib.mk + lib/src/ + lib/include/) lives in the sibling cccl-fpdev
# repo; overridable (the cccl-fpdev root Makefile passes it explicitly).
FPDEV_DIR ?= $(abspath $(THIS_DIR)/../../../../../../../../cccl-fpdev)
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

# verbose mode
VERBOSE_MODE = 0
ifneq (,$(filter $(V) $(VERBOSE) $(VERB), 1))
    VERBOSE_MODE = 1
else ifneq (,$(filter $(V) $(VERBOSE) $(VERB), 2))
    VERBOSE_MODE = 2
else ifneq (,$(filter $(V) $(VERBOSE) $(VERB), 3 y Y yes Yes full Full f F))
    VERBOSE_MODE = 3
endif

# utility functions
BASENAME = $(basename $(notdir $@))
W1 = $(word 1, $(subst -, , $(BASENAME)))
W2 = $(word 2, $(subst -, , $(BASENAME)))
W3 = $(word 3, $(subst -, , $(BASENAME)))
W4 = $(word 4, $(subst -, , $(BASENAME)))
W5 = $(word 5, $(subst -, , $(BASENAME)))
# convert to upper case
define ToUpper
$(shell echo $1 | tr '[:lower:]' '[:upper:]')
endef
# convert to lower case 
define ToLower
$(shell echo $1 | tr '[:upper:]' '[:lower:]')
endef

ifeq ($(VERBOSE_MODE), 1)
    RULE_DONE = -@echo "ts: $(notdir $@) built"
else
    RULE_DONE = 
endif

# all available operations, rounding modes and accuracies
ALL_METHODS   = mid high
EXT_METHODS   = $(ALL_METHODS) low
ALL_DATASETS  = zero denorm key fixed hard gauss uniform mixed normal finite inf nan
ALL_PRINTS    = ok fail warn all none

ifneq (,$(filter $(T) $(TARGET), host))
  ALL_ROUNDINGS   = rn
  DEFAULT_FUNCS   = fma add sub mul div sqrt
  ALL_FUNCS       = $(DEFAULT_FUNCS)
  EXT_FUNCS       = $(ALL_FUNCS) mad exp poly4 polyf4 poly8 polyf8 polyu8 dot cmul
else
  ALL_ROUNDINGS   = rn rz ru rd
  DEFAULT_FUNCS   = fma add sub mul div sqrt
  ALL_FUNCS       = $(DEFAULT_FUNCS) dfma dadd dmul ddiv dsub dsqrt
  EXT_FUNCS       = $(ALL_FUNCS) mad rsqrt exp poly4 polyf4 poly8 polyf8 polyu8 dot cmul
endif

# default operations, rounding modes and accuracies
DEFAULT_ROUNDINGS = rn rz ru rd
DEFAULT_METHODS   = high

OPT_FUNCS         = fma add sub mul
OPT_ROUNDINGS     = rn
OPT_METHODS       = mid high

UNPACKED_FUNCS     = dmul dadd dsub
UNPACKED_ROUNDINGS = rn
UNPACKED_METHODS   = mid low

# API selects which emulated API the harness exercises. It may be a LIST so one
# invocation evaluates both side by side (distinct target names and log rows):
#   packed   -> fpemu           (packed API: legacy kernels, or the
#                                          unpack->core->pack path when PACKED_VIA_UNPACKED=y)
#   unpacked -> fpemu_unpacked  (unpacked API directly: operands are stored
#                                          already unpacked, so the TIMED kernel runs the
#                                          pure unpacked cores -- unpack/pack happen only in
#                                          dataset fill and accuracy compare, OUTSIDE the
#                                          measured region; ideal for unpacked-kernel perf)
# The emulated C++ type is selected PER TARGET from the api token (see get_type
# below), so a single build can carry both packed and unpacked executables.
ALL_APIS = packed unpacked

API  ?= packed
ifeq ($(API),all)
	override API = $(ALL_APIS)
endif
APIS  = $(filter $(API), $(ALL_APIS))
ifeq ($(APIS),)
    $(error Empty API list! API must be one or more of: $(ALL_APIS))
endif

# all operations, rounding modes and accuracies
ifeq ($(FUNC),all)
	override FUNC = $(ALL_FUNCS)
endif
ifeq ($(FUNC),ext)
	override FUNC = $(EXT_FUNCS)
endif

ifeq ($(ROUNDING),all)
	override ROUNDING = $(ALL_ROUNDINGS)
endif

# ACC: short alias for ACCURACY (e.g. ACC=mid); ACCURACY takes precedence
ifdef ACC
ACCURACY ?= $(ACC)
endif

ifeq ($(ACCURACY),all)
	override ACCURACY = $(ALL_METHODS)
endif
ifeq ($(ACCURACY),ext)
	override ACCURACY = $(EXT_METHODS)
endif

# default operations, rounding modes and accuracies
ifeq ($(FUNC),def)
	override FUNC = $(DEFAULT_FUNCS)
endif
ifeq ($(ROUNDING),def)
	override ROUNDING = $(DEFAULT_ROUNDINGS)
endif

# optimized operations, rounding modes and accuracies
ifeq ($(FUNC),opt)
	override FUNC = $(DEFAULT_FUNCS)
endif
ifeq ($(ROUNDING),opt)
	override ROUNDING = $(DEFAULT_ROUNDINGS)
endif
ifeq ($(ACCURACY),opt)
	override ACCURACY = $(DEFAULT_METHODS)
endif

# default operations, rounding modes and accuracies
ROUNDING ?= rn
FUNC     ?= $(DEFAULT_FUNCS)
# ACCURACY is the external parameter - use it directly for test names
# Default ACCURACY if not specified
ACCURACY ?= def

# selected operations, rounding modes, and accuracies
ifeq ($(SET),all)
	override FUNC     = $(ALL_FUNCS)
	override ROUNDING = $(ALL_ROUNDINGS)
	override ACCURACY = $(ALL_METHODS)
	override API      = $(ALL_APIS)
endif
ifeq ($(SET),ext)
	override FUNC     = $(EXT_FUNCS)
	override ROUNDING = $(ALL_ROUNDINGS)
	override ACCURACY = $(EXT_METHODS)
	override API      = $(ALL_APIS)
endif
ifeq ($(SET),def)
	override FUNC         = $(DEFAULT_FUNCS)
	override ROUNDING     = $(DEFAULT_ROUNDINGS)
	override API          = $(ALL_APIS)
	override ACCURACY     = $(ALL_METHODS)
	override DATASET      = mixed
	override ACCURACY_LEN = 1000000
endif
ifeq ($(SET),ci)
	override FUNC     = $(ALL_FUNCS)
	override ROUNDING = $(ALL_ROUNDINGS)
	override API      = $(ALL_APIS)
	override ACCURACY = $(ALL_METHODS)
	override DATASET  = mixed
endif

# selected operations, rounding modes, and accuracies
FUNCS     = $(filter $(FUNC),     $(ALL_FUNCS))
ROUNDINGS = $(filter $(ROUNDING), $(ALL_ROUNDINGS))
METHODS   = $(filter $(ACCURACY),   mid high low def)

CSV_FILE ?= $(OUT)/$(shell date +%Y-%m-%d).csv

# all tests - target name is func-rounding-accuracy-api (api selects the emulated type)
TESTS_ALL = $(foreach function, $(FUNCS), $(foreach rounding, $(ROUNDINGS), $(foreach accuracy, $(METHODS), $(foreach api, $(APIS), $(function)-$(rounding)-$(accuracy)-$(api)))))

# Directed rounding (rz/ru/rd) is exercised ONLY through the d-prefixed CUDA-builtin
# overloads (dadd/dsub/dmul/ddiv/dsqrt/dfma). The non-prefixed operator-style
# functions (add/sub/mul/div/sqrt and the compound mad/dot/cmul/rsqrt) apply the
# plain C++ operator / _rn intrinsic to BOTH the emulated and the native operand,
# so their rz/ru/rd variants merely re-test rn on both sides (vacuous). fma is the
# same case: fma_function and dfma_function are byte-identical, so fma's directed
# variants just duplicate dfma. Excluding all of these on every target keeps the
# device report consistent with host (where these run rn-only) and avoids
# presenting users with meaningless "<rz/ru/rd>" rows.
OPFAMILY_DIRECTED_EXCLUDE = \
    fma-rz-% fma-ru-% fma-rd-% \
    add-rz-% add-ru-% add-rd-% \
    sub-rz-% sub-ru-% sub-rd-% \
    mul-rz-% mul-ru-% mul-rd-% \
    div-rz-% div-ru-% div-rd-% \
    sqrt-rz-% sqrt-ru-% sqrt-rd-% \
    mad-rz-% mad-ru-% mad-rd-% \
    dot-rz-% dot-ru-% dot-rd-% \
    cmul-rz-% cmul-ru-% cmul-rd-% \
    rsqrt-rz-% rsqrt-ru-% rsqrt-rd-%

TESTS_EXCLUDE = $(OPFAMILY_DIRECTED_EXCLUDE)
TESTS         = $(filter-out $(TESTS_EXCLUDE), $(TESTS_ALL))

ifeq ($(TESTS),)
    $(error Empty test list! Check your FUNC, ROUNDING, and ACCURACY variables.)
endif

# all object files
OBJ_FILES = $(foreach test, $(TESTS), $(OUT)/$(test).o)
# all executable files
EXE_FILES = $(foreach test, $(TESTS), $(OUT)/$(test).exe)
# all log files
LOG_FILES = $(foreach test, $(TESTS), $(OUT)/$(test).log)

ifeq (,$(filter $(LINKAGE), static lto))
# all ptx files
PTX_FILES = $(foreach test, $(TESTS), $(OUT)/$(test).ptx)
# all sass files
SASS_FILES = $(foreach test, $(TESTS), $(OUT)/$(test).sass)
endif

# Treat warnings as errors when WERROR is enabled (default 0 for ts). When on,
# host adds -Wall -Wextra -Werror and nvcc adds -Werror via -Xcompiler (device
# already carries -Wall,-Wextra below). Enable with WERROR=1.
WERROR ?= 0
ifeq (,$(filter 0 n N no No false False,$(WERROR)))
    WERROR_HOST := -Wall -Wextra -Wno-strict-aliasing -Werror
    WERROR_NVCC := -Xcompiler -Werror
else
    WERROR_HOST :=
    WERROR_NVCC :=
endif

# Compiler settings
ifneq (,$(filter $(T) $(TARGET), host))
    TARGET_SUFFIX = $(HOST_SUFFIX)
    # Host settings
    CXX = g++ -ffp-contract=off 
	COMMONFLAGS += -O2 -std=c++20 $(WERROR_HOST)
    OBJFLAGS  =  -c 
	EXEFLAGS  = 
	PTXFLAGS  = 
	SASSFLAGS = 
	ASM_EXT  = s
	ASM_FLAG = -S
	ASM_NAME = asm
else
    TARGET_SUFFIX = $(DEVICE_SUFFIX)
    # CUDA settings
    CXX = nvcc
	COMMONFLAGS += -O2 -std=c++20 -rdc=true -Xcompiler -ffp-contract=off
    OBJFLAGS    = -arch=sm_$(ARCH)  -x cu -dc -Xcompiler -Wall,-Wextra,-Wno-strict-aliasing $(WERROR_NVCC)
	EXEFLAGS    = -arch=sm_$(ARCH) 
	SASSFLAGS   = -arch=sm_$(ARCH)
	PTXFLAGS    = -arch=compute_$(ARCH) 
	ASM_EXT  = ptx
	ASM_FLAG = -ptx
	ASM_NAME = ptx
endif

COMMONFLAGS += $(EXTRA_FLAGS) $(ABI_FLAGS)

ifneq (,$(filter $(LINKAGE), static lto))
    LIB_BASE ?= cufp
    LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
	COMMONFLAGS += -DCCCL_FPEMU_LIB=1
else
    LIB_NAME =
	COMMONFLAGS += -D_CCCL_FPEMU_INLINE
endif

ifneq (,$(filter $(LINKAGE), lto))
	OBJFLAGS  += -dlto
	EXEFLAGS  += -dlto
	SASSFLAGS += -dlto
else

endif

# Pass ACCURACY parameter to compilation - W3 is now ACCURACY (high, mid, low)
# Helper functions to extract parts from target name (e.g., add-rn-low.o -> add, rn, low)
get_basename = $(basename $(notdir $(1)))
get_w1 = $(word 1,$(subst -, ,$(call get_basename,$(1))))
get_w2 = $(word 2,$(subst -, ,$(call get_basename,$(1))))
get_w3 = $(word 3,$(subst -, ,$(call get_basename,$(1))))
get_w4 = $(word 4,$(subst -, ,$(call get_basename,$(1))))
# Map W3 to ACCURACY
get_method = $(if $(filter high,$(1)),high,$(if $(filter mid,$(1)),mid,$(if $(filter low,$(1)),low,$(if $(filter cr,$(1)),high,$(if $(filter ha,$(1)),mid,$(if $(filter la,$(1)),low,$(if $(filter def,$(1)),def,mid)))))))
# Map W4 (api) to the emulated C++ type fed to the harness (default packed)
get_type = $(if $(filter unpacked,$(1)),fpemu_unpacked,fpemu)
# Build the full flags for a target
get_target_flags = $(INCLUDES) -D__FUNC__=$(call get_w1,$(1)) -D__ROUNDING__=$(call get_w2,$(1)) -D__METHOD__=$(call get_w3,$(1)) -D__TYPE__=$(call get_type,$(call get_w4,$(1)))
# THREADS: number of threads per block
ifdef THREADS
	COMMONFLAGS += -D__THREADS_PER_BLOCK__=$(THREADS)
endif

# REPEATS: number of repetitions of each function
ifdef REPEATS
	COMMONFLAGS += -D__FUNC_REPEATS__=$(REPEATS)
endif

# ITERATIONS: number of timing iterations
ifdef ITERATIONS  
	COMMONFLAGS += -D__TIMING_ITERATIONS__=$(ITERATIONS)
endif

# ACCURACY_LEN: length of the accuracy test array
ifdef ACCURACY_LEN
	COMMONFLAGS += -D__ACCURACY_LEN__=$(ACCURACY_LEN)
endif

# LATENCY: latency length
ifdef LATENCY_LEN
	COMMONFLAGS += -D__LATENCY_LEN__=$(LATENCY_LEN)
endif

# THROUGHPUT: throughput length
ifdef THROUGHPUT_LEN
	COMMONFLAGS += -D__THROUGHPUT_LEN__=$(THROUGHPUT_LEN)
endif

# MEAN: mean
ifdef MEAN
	COMMONFLAGS += -D__MEAN__=$(MEAN)
endif

# STDDEV: standard deviation
ifdef STDDEV
	COMMONFLAGS += -D__STDDEV__=$(STDDEV)
endif

# USE_BUILTINS: use builtins
ifneq (,$(filter $(USE_BUILTINS), 1 y Y yes Yes true True))
	COMMONFLAGS += -D__USE_BUILTINS__=1
endif

# NAN_PAYLOAD: use nan payload
ifneq (,$(filter $(NAN_PAYLOAD), 1 y Y yes Yes true True))
	COMMONFLAGS += -D__NAN_PAYLOAD__=1
endif

# PRINT_LIMIT: print limit
ifdef PRINT_LIMIT
	COMMONFLAGS += -D__PRINT_LIMIT__=$(PRINT_LIMIT)
endif

ifneq (,$(filter $(PACKED_VIA_UNPACKED), 1 y Y yes Yes true True))
	COMMONFLAGS += -D_CCCL_FPEMU_PACKED_VIA_UNPACKED=1
endif

# TIMING: run timing (default is 0)
ifneq (,$(filter $(TIMING), 1 y Y yes Yes true True))
	COMMONFLAGS += -D__RUN_TIMING__=1
endif

# DATASET: datasets to use
ifdef DATASET
	DATASETS    = $(filter $(DATASET),  $(ALL_DATASETS))
    DATASETS_U := $(call ToUpper, $(DATASETS))
	COMMONFLAGS   += $(foreach dataset, $(DATASETS_U), -D__DATASET_$(dataset)__)
endif

# PRINT: prints to use
ifdef PRINT
	PRINTS     = $(filter $(PRINT),  $(ALL_PRINTS))
    PRINTS_U  := $(call ToUpper, $(PRINTS))
	COMMONFLAGS += $(foreach print, $(PRINTS_U), -D__PRINT_$(print)__)
endif

# Fixed input arguments	
ifdef A1
	COMMONFLAGS += -D__A1__=$(A1) 
endif
ifdef A2
	COMMONFLAGS += -D__A2__=$(A2)
endif
ifdef A3
	COMMONFLAGS += -D__A3__=$(A3)
endif

ifdef CONSOLE
	COMMONFLAGS += -D__CONSOLE__=$(CONSOLE)
endif

ifneq (,$(filter $(DEBUG), 1 y Y yes Yes true True))
	COMMONFLAGS += -D__FPEMU_DEBUG__
	ifndef $(CONSOLE)
		COMMONFLAGS += -D__CONSOLE__=null
	endif
endif

# Source directories
FPEMU_DIR      = $(THIS_DIR)/../..
# CCCL libcudacxx headers: <cuda/fpemu>, <cuda/std/...>.
# ts/fpemu lives at libcudacxx/test/libcudacxx/cuda/fp/ts/fpemu -> 6 levels up
# reaches libcudacxx/include.
INCLUDE_DIR    = $(THIS_DIR)/../../../../../../include

# The benchmark source file
SRC      = $(THIS_DIR)/ts.cpp

# Include directories
INCLUDES = -I$(THIS_DIR) \
		   -I$(INCLUDE_DIR)

# maximum verbose mode with -v flag
ifeq ($(VERBOSE_MODE), 3)
	COMMONFLAGS += -v
endif

# Default target
all: run

MAKEFLAGS += --no-print-directory

run: 
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build
	MAKEFLAGS= $(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile logs -j1

rerun:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build
	MAKEFLAGS= $(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile logs -j1

rebuild:
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile clean
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile build

# Output directory
OUT ?= $(THIS_DIR)/_out

# Build the object files
$(OBJ_FILES) : $(SRC) $(LIB_NAME) | $(OUT)
	$(CXX) $(COMMONFLAGS) $(call get_target_flags,$@) $(OBJFLAGS) -o $@ $<
	$(RULE_DONE)

# Build the executable files
$(EXE_FILES) :$(OUT)/%.exe: $(OUT)/%.o $(OBJ_FILES) $(LIB_NAME) | $(OUT)
	$(CXX) $(COMMONFLAGS) $(call get_target_flags,$@) $(EXEFLAGS) -o $@ $< $(LIB_NAME)
	$(RULE_DONE)

# Build the ptx files and count the lines
$(PTX_FILES) : $(SRC)  $(LIB_NAME) | $(OUT)
	$(CXX) $(COMMONFLAGS) $(call get_target_flags,$@) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.emu
	$(CXX) $(COMMONFLAGS) $(call get_target_flags,$@) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -D__NATIVE_IMPL__ -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.native
	awk '/\.visible/ {in_func=1; print; next} in_func {print} /ret/ {if (in_func) {print; in_func=0; exit}}' $@.emu > $@.emu.tmp
	awk '/\.visible/ {in_func=1; print; next} in_func {print} /ret/ {if (in_func) {print; in_func=0; exit}}' $@.native > $@.native.tmp
	@echo "ts: $(notdir $@.emu)    : $$(($$(wc -l < $@.emu.tmp) - 5)) lines"
	@echo "ts: $(notdir $@.native) : $$(($$(wc -l < $@.native.tmp) - 5)) lines"
	rm -f $@.emu.tmp $@.native.tmp

# Build the sass files and count the lines + ptxas register usage.
#
# Two-pass compile path keeps the SASS line count wrapper-free while still
# capturing physical register allocation.  See ts/fpmp/Makefile for the
# detailed rationale; the structure here is the same with .emu/.native
# tags replacing fpmp's .mp/.hi tags and `$(call get_target_flags,$@)`
# threading per-target FUNC/ROUNDING/ACCURACY defines.
#
#   Pass 1 (FUNC):   ptxas --compile-only on PTX without the wrapper —
#                    cubin is dumped to get the pure FUNC_IMPL SASS.
#   Pass 2 (KERN):   ptxas final-codegen on PTX with -D__TS_SASS_KERNEL__=1
#                    so a launchable `__global__ __ts_sass_kernel` exists;
#                    its ptxas log carries the `Used N registers` count.
$(SASS_FILES) : $(SRC)  $(LIB_NAME) | $(OUT)
	@$(CXX) $(COMMONFLAGS) $(call get_target_flags,$@) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.emu.ptx
	@$(CXX) $(COMMONFLAGS) $(call get_target_flags,$@) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -D__NATIVE_IMPL__ -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.native.ptx
	@ptxas --compile-only -arch=sm_$(ARCH) $@.emu.ptx -o $@.emu.cubin > $@.emu.log 2>&1 \
	  || { cat $@.emu.log >&2; rm -f $@.emu.log $@.emu.ptx; exit 1; }
	@ptxas --compile-only -arch=sm_$(ARCH) $@.native.ptx -o $@.native.cubin > $@.native.log 2>&1 \
	  || { cat $@.native.log >&2; rm -f $@.native.log $@.native.ptx; exit 1; }
	cuobjdump --dump-sass $@.emu.cubin > $@.emu
	cuobjdump --dump-sass $@.native.cubin > $@.native
	@$(CXX) $(COMMONFLAGS) $(call get_target_flags,$@) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -D__TS_SASS_KERNEL__=1 -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.emu.kern.ptx
	@$(CXX) $(COMMONFLAGS) $(call get_target_flags,$@) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -D__TS_SASS_KERNEL__=1 -D__NATIVE_IMPL__ -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.native.kern.ptx
	@ptxas -v -arch=sm_$(ARCH) $@.emu.kern.ptx -o $@.emu.kern.cubin > $@.emu.kern.log 2>&1 \
	  || { cat $@.emu.kern.log >&2; rm -f $@.emu.kern.log $@.emu.kern.ptx; exit 1; }
	@ptxas -v -arch=sm_$(ARCH) $@.native.kern.ptx -o $@.native.kern.cubin > $@.native.kern.log 2>&1 \
	  || { cat $@.native.kern.log >&2; rm -f $@.native.kern.log $@.native.kern.ptx; exit 1; }
	rm -f $@.emu.cubin $@.native.cubin $@.emu.ptx $@.native.ptx \
	      $@.emu.kern.cubin $@.native.kern.cubin $@.emu.kern.ptx $@.native.kern.ptx
	awk '/Function :.*_device_impl/ {in_func=1; print; next} in_func {print} /NOP;/ {if (in_func) {print; in_func=0; exit}}' $@.emu > $@.emu.tmp
	awk '/Function :.*_device_impl/ {in_func=1; print; next} in_func {print} /NOP;/ {if (in_func) {print; in_func=0; exit}}' $@.native > $@.native.tmp
	@base=$$(basename $@); \
	 width=$$(printf '%s\n' "$$base.emu" "$$base.native" | awk '{ if (length($$0) > m) m = length($$0) } END { print m }'); \
	 for tag in emu native; do \
	   sass=$@.$$tag; tmp=$$sass.tmp; kern_log=$$sass.kern.log; \
	   lines=$$((($$(wc -l < $$tmp) - 4) / 2)); \
	   regs=$$(grep -oE 'Used [0-9]+ registers' $$kern_log | grep -oE '[0-9]+' | sort -n | tail -1); \
	   printf 'ts: %-*s: %d lines, %s regs\n' "$$width" "$$base.$$tag" "$$lines" "$${regs:-?}"; \
	   if [ -z "$$regs" ]; then \
	     printf '  [kernel-pass ptxas log size: %s bytes; first 12 lines below]\n' "$$(wc -c < $$kern_log)" >&2; \
	     head -12 $$kern_log | sed "s/^/    | /" >&2; \
	   fi; \
	 done
	@rm -f $@.emu.tmp $@.native.tmp $@.emu.log $@.native.log $@.emu.kern.log $@.native.kern.log

# Generate the log files with running the executable files
$(LOG_FILES): $(OUT)/%.log: $(OUT)/%.exe | $(OUT)
	@$< $(CSV_FILE) 2> $@

# Build the executable files
ifneq (,$(filter $(T) $(TARGET), host))
build: $(EXE_FILES)
else
 ifneq (,$(filter $(ASM), 1 y Y yes Yes true True))
build: $(EXE_FILES) $(SASS_FILES) $(PTX_FILES)
 else
  build: $(EXE_FILES)
 endif
endif

$(LIB_NAME):
	$(MAKE) --no-print-directory -f $(FPDEV_DIR)/lib/make/lib.mk TARGET=$(TARGET) OUT=$(OUT) lib

# Generate the log files with running the executable files
logs: 
	rm -f $(CSV_FILE)
	$(MAKE) --no-print-directory -f $(THIS_DIR)/Makefile $(LOG_FILES)

# Clean up
clean:
	rm -fr $(OUT)

# Create directories if they don't exist
$(OUT):
	mkdir -p $(OUT)

help:
	@echo "FPEMU Test Suite (ts/fpemu/) Makefile"
	@echo "====================================="
	@echo ""
	@echo "Build Targets:"
	@echo "  all        - build object files and run timing (default target)"
	@echo "  run        - build object files and run timing"
	@echo "  rerun      - rebuild object files and run timing"
	@echo "  rebuild    - rebuild object files only"
	@echo "  build      - build object files and executables"
	@echo "  logs       - generate log files by running executables"
	@echo "  clean      - delete all generated files and folders"
	@echo "  help       - show this help message"
	@echo ""
	@echo "Core Parameters:"
	@echo "  TARGET     = target device (device (default), host)"
	@echo "  OUT        = output directory (default: _out)"
	@echo "  LINKAGE    = linkage mode (inline (default), static, lto)"
	@echo "  ARCH       = CUDA architecture (default: 86)"
	@echo "  VERBOSE    = verbose mode (0=silent, 1=minimal, 2=full, 3=maximum)"
	@echo "  FMAD_FLAG  = -fmad= flag (false (default), true)"
	@echo ""
	@echo "Test Configuration:"
	@echo "  SET        = test set (def=default functions, opt=optimized functions, all=all functions)"
	@echo "  FUNC       = function to evaluate (fma, mad, add, sub, mul, div, sqrt, exp, poly4, polyf4, poly8, polyf8)"
	@echo "  API        = emulated API to exercise; list allowed (packed (default), unpacked)"
	@echo "  ROUNDING   = rounding mode (rn, rz, ru, rd)"
	@echo "  ACCURACY     = accuracy level (high (default), mid, low) [alias: ACC]"
	@echo ""
	@echo "Library Parameters:"
	@echo "  USE_BUILTINS = use builtins for compound functions (1=yes (default), 0=no)"
	@echo "  PACKED_VIA_UNPACKED = route packed API via unpacked cores for testing (0=no (default), 1=yes)"
	@echo ""
	@echo "Performance Parameters:"
	@echo "  THREADS    = number of threads per block (default: 1024)"
	@echo "  REPEATS    = number of repetitions (default: 100)"
	@echo "  ITERATIONS = number of timing iterations (default: 5)"
	@echo "  TIMING     = run timing (0=no (default), 1=yes)"
	@echo "  ACCURACY_LEN = length of accuracy array (default: 1000000)"
	@echo "  LATENCY_LEN = length of latency timing array (default: 1000000)"
	@echo "  THROUGHPUT_LEN = length of throughput timing array (default: 1000000)"
	@echo ""
	@echo "Data Parameters:"
	@echo "  DATASET    = dataset type (zero, denorm, key, fixed, hard, gauss, uniform, mixed, normal, finite, inf, nan)"
	@echo "  MEAN       = mean value (default: 0)"
	@echo "  STDDEV     = standard deviation (default: 1)"
	@echo "  NAN_PAYLOAD = use nan payload (0=no (default), 1=yes)"
	@echo "  A1, A2, A3 = fixed input arguments (no default)"
	@echo ""
	@echo "Output Parameters:"
	@echo "  PRINT      = print level (ok, fail, warn, all, none, default: fail)"
	@echo "  PRINT_LIMIT = print limit (default: 10)"
	@echo "  DEBUG      = debug mode (0=no (default), 1=yes)"
	@echo "  CONSOLE    = console type (stdout (default), stderr, file, null)"
	@echo "  CSV_FILE   = csv file name (default: $(CSV_FILE))"
	@echo "  ASM        = generate assembly (0=no (default), 1=yes)"
	@echo ""
	@echo "Available Function Sets:"
	@echo "  Host:      fma, mad, add, sub, mul, div, sqrt, exp, poly4, polyf4, poly8, polyf8"
	@echo "  Device:    fma, mad, add, sub, mul, div, dfma, dadd, dmul, ddiv, dsub, sqrt, dsqrt, exp, poly4, polyf4, poly8, polyf8"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make                           - build and run default function tests on device"
	@echo "  make rerun                     - rebuild and run default function tests on device"
	@echo "  make TARGET=host               - build and run tests on host"
	@echo "  make SET=def                   - run default function set"
	@echo "  make SET=opt                   - run optimized function set"
	@echo "  make SET=all                   - run all available functions"
	@echo "  make FUNC=dadd                 - run only dadd function"
	@echo "  make ROUNDING=rn,rz            - run with specific rounding modes"
	@echo "  make ACCURACY=high               - run with high (correctly-rounded) accuracy"
	@echo "  make ACCURACY=all                - run with all computation accuracies"
	@echo "  make TIMING=1                  - enable timing measurements"
	@echo "  make DATASET=gauss             - use gaussian dataset"
	@echo "  make PRINT=all                 - print all results"
	@echo "  make DEBUG=1                   - enable debug mode"
	@echo "  make clean                     - clean all generated files"

.PHONY: all clean $(LOG_FILES) build run rerun rebuild logs help

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
