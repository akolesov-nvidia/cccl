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
#    `SET`         = test set (`def` (default), `all`)
#    `FUNC`        = function(s) to evaluate (space-separated, or `all`)
#    `TYPE`        = type(s) of multi-precision (`fp32mp2`, `fp64mp2`; space-separated, or `all`)
#    `ACCURACY`      = accuracy level(s) (`def`, `low`, `high`; space-separated, or `all`)
#    `TARGET`      = target device (`device` (default), `host`)
#    `ARCH`        = CUDA sm_xx (unset: nvidia-smi, else 86)
#    `LINKAGE`     = linkage mode (`inline` (default), `static`, `lto`)
#    `RIGOR`       = accuracy rigor level (log2 of samples, e.g., 24=16M, 32=4G; passed as `TS_ACCURACY_RIGOR`)
#    `RUN_ACCURACY` = enable accuracy tests (`y` (default), `n` to disable)
#    `CLASSIFY`    = enable accuracy classification (`y` (default), `n` to disable for faster host runs)
#    `TIMING`      = enable timing tests (`y` (default), `n` to disable)
#    `PRINT`       = what to print (`ok`, `fail`, `warn`, `all`, `none`)
#    `DATASET`     = dataset(s) to include in reports (`work`, `normal`, `special`, `pattern`)
#    `A1`, `A2`, `A3`, `A4` = fixed input arguments for debugging (enables __FIXED_INPUTS__ mode)
#                         For multi-precision: use colon separator, e.g., A1=1.5:1e-8
#                         For scalars: just the value, e.g., A1=3.14159
#    `CONSOLE`     = console stream (`stdout`, `stderr`, `file`, `null`)
#    `OUT`         = output directory (default: `_out`)
#    `VERBOSE`     = verbose mode (`0`/`1`/`2`/`3`); aliases: `V`, `VERB`
#    `FP2MP`       = enable optimized double→fpmp2 conversion (`1` to enable CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP)
#    `MP2FP`       = enable optimized fpmp2→double conversion (`1` to enable CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE)
#    `EXTRA_FLAGS` = extra compiler flags (inherited from root Makefile if exported)
#    `FPMP_NVVM_FP128_EXT` = set by the repo root Makefile for CUDA device builds (extended __nv_fp128_*).
#                            Standalone builds query the root Makefile when this is unset.
#
#  Notes (WIP):
#  ------------
#  Some legacy knobs (e.g. NAN_PAYLOAD / TOTAL_TIMING / FMAD_FLAG) may be present in the Makefile but
#  are not currently wired into compilation; this test suite is still evolving.
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
# run timing by default
TIMING ?= y
# run accuracy tests by default
RUN_ACCURACY ?= y
# enable accuracy classification by default
CLASSIFY ?= y
# Linkage mode
LINKAGE ?= inline

ifneq (,$(filter y Y yes Yes true True 1,$(NOFP128)))
    EMU_NO_FP128 := 1
else
    EMU_NO_FP128 :=
endif

# target suffix: distinguishes host vs device output filenames
DEVICE_SUFFIX ?=
HOST_SUFFIX   ?= _host
# fmad flag
FMAD_FLAG ?= false
# generate ptx and sass files by default
ASM ?= no

# current directory
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

ALL_MATH      = exp exp2 exp10 expm1 ldexp scalbn log log2 log10 log1p pow cbrt rcbrt fmod remainder floor ceil round trunc rint nearbyint sin cos tan sincos asin acos atan atan2 sinh cosh tanh asinh acosh atanh erf erfc boys_f0 normcdfinv
# all available operations, rounding modes, accuracies and ranges
ALL_FUNCS     = fma mad add acc sub mul div sqrt rsqrt mp2int int2mp mp2uint uint2mp mp2ll ll2mp mp2ull ull2mp mp2fp fp2mp eq ne lt le gt ge $(ALL_MATH)
ALL_TYPES     = fp32mp2 fp64mp2
ALL_METHODS   = def low high
ALL_DATASETS  = work normal special pattern
ALL_PRINTS    = ok fail warn all none

# Functions that support multiple accuracies (fast, def, accurate)
MULTI_METHOD_FUNCS = add sub mul div acc mad fma
# Functions that support fast and def accuracies only (no accurate variant)
FAST_DEF_FUNCS     = 
# Functions that only support the default accuracy
# Note: sincos is excluded — it returns two values and doesn't fit the single-result test framework
DEF_ONLY_FUNCS     = sqrt rsqrt $(filter-out sincos,$(ALL_MATH)) mp2int int2mp mp2uint uint2mp mp2ll ll2mp mp2ull ull2mp mp2fp fp2mp eq ne lt le gt ge

DEFAULT_FUNCS   = fma mad add acc sub mul div sqrt rsqrt exp exp2 exp10 expm1 ldexp scalbn log log2 log10 log1p pow cbrt rcbrt fmod remainder floor ceil round trunc rint nearbyint sinh cosh tanh asinh acosh atanh erf erfc boys_f0 sin cos tan asin acos atan atan2 normcdfinv mp2int int2mp mp2uint uint2mp mp2ll ll2mp mp2ull ull2mp mp2fp fp2mp eq ne lt le gt ge
DEFAULT_TYPES   = fp32mp2 fp64mp2
DEFAULT_METHODS = def low high

# all operations, rounding modes, accuracies and ranges
ifeq ($(FUNC),all)
	override FUNC = $(ALL_FUNCS)
endif
ifeq ($(TYPE),all)
	override TYPE = $(ALL_TYPES)
endif

# ACC: short alias for ACCURACY (e.g. ACC=mid); ACCURACY takes precedence
ifdef ACC
ACCURACY ?= $(ACC)
endif

ifeq ($(ACCURACY),all)
	override ACCURACY = $(ALL_METHODS)
endif

# default operations, rounding modes, accuracies and ranges
FUNC     ?= $(DEFAULT_FUNCS)
TYPE     ?= $(DEFAULT_TYPES)
ACCURACY   ?= $(DEFAULT_METHODS)

# selected operations, rounding modes, accuracies and ranges
ifeq ($(SET),all)
	override FUNC     = $(ALL_FUNCS)
	override TYPE     = $(ALL_TYPES)
	override ACCURACY = $(ALL_METHODS)
endif
ifeq ($(SET),def)
	override FUNC     = $(DEFAULT_FUNCS)
	override TYPE     = $(DEFAULT_TYPES)
	override ACCURACY = $(DEFAULT_METHODS)
endif
ifeq ($(SET),ci)
	override FUNC     = $(ALL_FUNCS)
	override TYPE     = $(ALL_TYPES)
	override ACCURACY = $(ALL_METHODS)
	override DATASET  = pattern
	override TIMING   = 0
endif

# selected operations, rounding modes, accuracies and ranges
FUNCS   = $(filter $(FUNC),   $(ALL_FUNCS))
TYPES   = $(filter $(TYPE),   $(ALL_TYPES))
METHODS = $(filter $(ACCURACY), $(ALL_METHODS))

CSV_FILE ?= $(OUT)/$(shell date +%Y-%m-%d).csv

# all tests (raw cross product)
TESTS_ALL = $(foreach function, $(FUNCS), $(foreach type, $(TYPES), $(foreach accuracy, $(METHODS), $(function)-$(type)-$(accuracy))))

# Build list of valid tests based on function/accuracy support:
# - MULTI_METHOD_FUNCS: support def, low, high
# - FAST_DEF_FUNCS: support def, low only (no high)
# - DEF_ONLY_FUNCS: support def only

# Tests for functions that support all accuracies
TESTS_MULTI = $(foreach func, $(filter $(MULTI_METHOD_FUNCS), $(FUNCS)), \
                $(foreach type, $(TYPES), \
                  $(foreach accuracy, $(METHODS), $(func)-$(type)-$(accuracy))))

# Tests for functions that support fast and def only
TESTS_FAST_DEF = $(foreach func, $(filter $(FAST_DEF_FUNCS), $(FUNCS)), \
                   $(foreach type, $(TYPES), \
                     $(foreach accuracy, $(filter-out high, $(METHODS)), $(func)-$(type)-$(accuracy))))

# Tests for functions that only support def
TESTS_DEF_ONLY = $(foreach func, $(filter $(DEF_ONLY_FUNCS), $(FUNCS)), \
                   $(foreach type, $(TYPES), $(func)-$(type)-def))

# Combine valid tests
TESTS = $(TESTS_MULTI) $(TESTS_FAST_DEF) $(TESTS_DEF_ONLY)

# Exclude (function, type) combos that have neither a dedicated library
# implementation nor a usable reference at the required precision. Currently:
#   normcdfinv-fp64mp2 — the fp64mp2 path is a thin wrapper over the system
#       erfcinv (libm has no fp128 erfcinv on most platforms), and the
#       reference itself is double-precision-bound, so the comparison gives
#       spurious failures in CI without telling us anything new about the
#       library. Drop the (normcdfinv, fp64mp2) cell entirely.
TESTS := $(filter-out normcdfinv-fp64mp2-%, $(TESTS))

# Exclude functions without proper host reference implementations
ifneq (,$(filter $(T) $(TARGET), host))
    TESTS := $(filter-out erfc-%, $(TESTS))
endif

ifeq ($(strip $(TESTS)),)
    $(error Empty test list! Check your FUNC, TYPE, ACCURACY variables.)
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

# Standalone builds (no parent make): the NVVM literal is defined ONLY in the repo
# root Makefile (it is intentionally absent from the customer-exposed make/lib.mk).
# Query it from the root so that direct `make -C ts/fpmp ...` invocations still get
# extended __nv_fp128_* on devices that need them.
FPMP_REPO_ROOT := $(abspath $(THIS_DIR)/../..)
ifeq (,$(filter host,$(T) $(TARGET)))
ifeq ($(strip $(FPMP_NVVM_FP128_EXT)),)
    # Internal nvcc switch that enables __float128 (fp128) device codegen on any
    # compiler version and GPU architecture, including the base and extended
    # __nv_fp128_* intrinsics (declared in crt/device_fp128_functions.h) that the
    # double-double accuracy reference relies on. Without it, nvcc rejects
    # __float128 in device code.
    #
    # This is a DEV-ONLY test-suite knob: the switch is internal and must NOT be
    # exposed to external users or appear in examples. Override with an explicit
    # `FPMP_NVVM_FP128_EXT=` (empty) to disable it.
    FPMP_NVVM_FP128_EXT := -nvvm-version=nvvm-latest
endif
endif

# Compiler settings
ifneq (,$(filter $(T) $(TARGET), host))
    TARGET_SUFFIX = $(HOST_SUFFIX)
    # Host settings
    CXX = g++ -ffp-contract=off 
	COMMONFLAGS += -O2 -std=c++20 -fopenmp -march=native
    OBJFLAGS  =  -c 
	EXEFLAGS  = 
	LINKLIBS  = $(QUADMATH_LIB) -lgomp
	PTXFLAGS  = 
	SASSFLAGS = 
	ASM_EXT  = s
	ASM_FLAG = -S
	ASM_NAME = asm
else
    TARGET_SUFFIX = $(DEVICE_SUFFIX)
    # CUDA settings
    CXX = nvcc
	COMMONFLAGS += -O2 -std=c++20 -rdc=true
	# Inherited from repo root Makefile as $(FPMP_NVVM_FP128_EXT); required for extended __nv_fp128_*.
	COMMONFLAGS += $(FPMP_NVVM_FP128_EXT)
    OBJFLAGS    = -arch=sm_$(ARCH)  -x cu -dc -Xcompiler -Wall,-Wextra,-Wno-strict-aliasing
	EXEFLAGS    = -arch=sm_$(ARCH) 
	LINKLIBS    = -lcurand $(QUADMATH_LIB)
	SASSFLAGS   = -arch=sm_$(ARCH)
	PTXFLAGS    = -arch=compute_$(ARCH) 
	ASM_EXT  = ptx
	ASM_FLAG = -ptx
	ASM_NAME = ptx
endif

COMMONFLAGS += $(EXTRA_FLAGS) $(ABI_FLAGS)

# Optimized double → fpmp2 conversion (integer bit manipulation, no FP64 ops)
ifeq ($(FP2MP), 1)
    COMMONFLAGS += -DCCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP=1
endif

# Optimized fpmp2 → double conversion (integer bit manipulation, no FP64 ops)
ifeq ($(MP2FP), 1)
    COMMONFLAGS += -DCCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE=1
endif

# FP128 support for reference value verification.
# The fp128 *type* and the in-test *reference* are always on; the library's
# fp64mp2 *math* fallback (_CCCL_FPMP_FP128_MATH_FALLBACK) is decided below.
COMMONFLAGS += -D_CCCL_FPMP_FP128_ENABLE=1
COMMONFLAGS += -D__TS_FP128_ENABLE__=1

# Library fp64mp2 math fallback selector. Emit EXACTLY ONE -D_CCCL_FPMP_FP128_MATH_FALLBACK
# so no compiler frontend can disagree on its value. nvcc preprocesses the host and
# device passes separately and does NOT reliably honor "last -D wins" for a duplicated
# (or -U'd then redefined) macro the way g++ does -- a stray -D_CCCL_FPMP_FP128_MATH_FALLBACK=1
# left earlier on the line can survive on the device pass while being overridden on the
# host pass, silently keeping native fp128 in the device library (false 106-bit result).
#   - normal                : 1 (library fp64mp2 math uses true binary128)
#   - NOFP128 + inline       : 0 (library math degraded to double; reference stays fp128
#                                 -- this TU holds both, gated by different macros)
#   - NOFP128 + static/lto   : 1 here is irrelevant -- the library lives in a separate .a
#                                 that is built fp128-stripped via LIB_FP128_ARGS, and this
#                                 TU only carries the (fp128) reference.
ifeq ($(EMU_NO_FP128),1)
    ifeq (,$(filter $(LINKAGE), static lto))
        TS_FP128_MATH_FALLBACK := 0
    else
        TS_FP128_MATH_FALLBACK := 1
    endif
else
    TS_FP128_MATH_FALLBACK := 1
endif
COMMONFLAGS += -D_CCCL_FPMP_FP128_MATH_FALLBACK=$(TS_FP128_MATH_FALLBACK)

# Inherit portable fp128 macros from the parent (e.g. tsmp's TSMP_FP128_FLAGS), but
# drop any _CCCL_FPMP_FP128_ENABLE / _CCCL_FPMP_FP128_MATH_FALLBACK it carries: we set those
# explicitly above and must not let a duplicate re-introduce the value ambiguity.
COMMONFLAGS += $(filter-out -D_CCCL_FPMP_FP128_ENABLE=% -D_CCCL_FPMP_FP128_MATH_FALLBACK=%,$(FP128_EXTRA_FLAGS))

# Keep the in-test fp128 reference in lock-step with the library build.
# make/lib.mk couples the extended NVVM flag with -D_CCCL_FPMP_CUDA_FP128_INTRINSICS=1:
# whenever $(FPMP_NVVM_FP128_EXT) is active on device, the native __nv_fp128_*
# (true binary128) math is guaranteed, so the library compiles fp64mp2 math with
# it. The reference (ts.hpp __TS_*Q) is gated on the same macro and MUST define
# it too -- otherwise it silently takes the degraded double-widen path. Under
# inline linkage both sides are in one TU so they degrade together and the bug
# hides (rel err 0, 106 bits); under static/lto the library uses native fp128
# while the reference does not, so the comparison fails (e.g. exp10/floor/ceil/
# rint fp64mp2 ~43 bits). $(FP128_EXTRA_FLAGS) only carries this when the caller
# passes TSMP_FP128_FLAGS (the tsmp target); `make ci` invokes us directly, so
# derive it here from the nvvm flag exactly as lib.mk does. (Duplicate -D with
# the same value is harmless.)
ifeq (,$(filter host,$(T) $(TARGET)))
    ifneq ($(strip $(FPMP_NVVM_FP128_EXT)),)
        COMMONFLAGS += -D_CCCL_FPMP_CUDA_FP128_INTRINSICS=1
    endif
endif

# Suppress long double warnings on device (demoted to double)
ifeq (,$(filter host,$(TARGET)))
    COMMONFLAGS += -Xcudafe --diag_suppress=20208
endif

ifneq (,$(filter $(LINKAGE), static lto))
    LIB_BASE ?= cufp
    LIB_NAME = $(OUT)/lib/lib$(LIB_BASE)$(TARGET_SUFFIX).a
	COMMONFLAGS += -DCCCL_FPMP_LIB=1 -DCCCL_FPEMU_LIB=1
endif

ifneq (,$(filter $(LINKAGE), lto))
	OBJFLAGS  += -dlto
	EXEFLAGS  += -dlto
	SASSFLAGS += -dlto
else ifneq (,$(filter $(LINKAGE), static))
 ifeq (,$(filter $(T) $(TARGET), host))
	# Static linkage without LTO requires relocatable device code
	OBJFLAGS  += -rdc=true
	EXEFLAGS  += -rdc=true
	SASSFLAGS += -rdc=true
 endif
endif

# +includes,  +function and type
COMMONFLAGS += $(INCLUDES) -D__FUNC__=$(W1) -D__TYPE__=$(W2) -D__METHOD__=$(W3)

# RIGOR: accuracy rigor level (log2 of sample count, e.g., 24 = 16M samples)
ifdef RIGOR
	COMMONFLAGS += -DTS_ACCURACY_RIGOR=$(RIGOR)
endif

# TIMING: run timing (default is 0)
ifneq (,$(filter $(TIMING), 1 y Y yes Yes true True))
	COMMONFLAGS += -D__RUN_TIMING__=1
else	
	COMMONFLAGS += -D__RUN_TIMING__=0
endif

# RUN_ACCURACY: run accuracy tests (default is 1)
ifneq (,$(filter $(RUN_ACCURACY), 0 n N no No false False))
	COMMONFLAGS += -D__RUN_ACCURACY__=0
else	
	COMMONFLAGS += -D__RUN_ACCURACY__=1
endif

# CLASSIFY: enable accuracy classification (default is 1)
ifneq (,$(filter $(CLASSIFY), 0 n N no No false False))
	COMMONFLAGS += -D__RUN_CLASSIFY__=0
else	
	COMMONFLAGS += -D__RUN_CLASSIFY__=1
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
# Note: __FIXED_INPUTS__ is automatically enabled in ts.hpp when __A1__/__A2__/__A3__ is defined
ifdef A1
	COMMONFLAGS += -D__A1__=$(A1) 
endif
ifdef A2
	COMMONFLAGS += -D__A2__=$(A2)
endif
ifdef A3
	COMMONFLAGS += -D__A3__=$(A3)
endif
ifdef A4
	COMMONFLAGS += -D__A4__=$(A4)
endif

ifdef CONSOLE
	COMMONFLAGS += -D__CONSOLE__=$(CONSOLE)
endif


# Source directories
FPMP_DIR       = $(THIS_DIR)/../..
# CCCL libcudacxx headers: <cuda/fpmp>, <cuda/fpmp_math>, <cuda/std/...>.
# ts/fpmp lives at libcudacxx/test/libcudacxx/cuda/fp/ts/fpmp -> 6 levels up
# reaches libcudacxx/include.
INCLUDE_DIR   = $(THIS_DIR)/../../../../../../include


# The benchmark source file
SRC      = $(THIS_DIR)/ts.cpp

# Include directories
INCLUDES = -I$(THIS_DIR) \
		   -I$(INCLUDE_DIR)

# maximum verbose mode with -v flag
ifeq ($(VERBOSE_MODE), 3)
	COMMONFLAGS += -v -Xptxas="-v" --keep --keep-dir $(OUT)
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
	$(CXX) $(COMMONFLAGS) $(OBJFLAGS) -o $@ $<
	$(RULE_DONE)

# Build the executable files
$(EXE_FILES) :$(OUT)/%.exe: $(OUT)/%.o $(OBJ_FILES) $(LIB_NAME) | $(OUT)
	$(CXX) $(COMMONFLAGS) $(EXEFLAGS) -o $@ $< $(LIB_NAME) $(LINKLIBS)
	$(RULE_DONE)

# Build the ptx files and count the lines
$(PTX_FILES) : $(SRC)  $(LIB_NAME) | $(OUT)
	$(CXX) $(COMMONFLAGS) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.mp
	$(CXX) $(COMMONFLAGS) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -D__NATIVE_IMPL__ -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.hi
	awk '/\.visible/ {in_func=1; print; next} in_func {print} /ret/ {if (in_func) {print; in_func=0; exit}}' $@.mp > $@.mp.tmp
	awk '/\.visible/ {in_func=1; print; next} in_func {print} /ret/ {if (in_func) {print; in_func=0; exit}}' $@.hi > $@.hi.tmp
	@echo "ts: $(notdir $@.mp) : $$(($$(wc -l < $@.mp.tmp) - 5)) lines"
	@echo "ts: $(notdir $@.hi) : $$(($$(wc -l < $@.hi.tmp) - 5)) lines"
	rm -f $@.mp.tmp $@.hi.tmp

# Build the sass files and count the lines + ptxas register usage.
#
# Two-pass compile path keeps the SASS line count wrapper-free while still
# capturing physical register allocation.
#
#   Pass 1 (FUNC):   nvcc compiles ts_functions.hpp to PTX without the
#                    __TS_SASS_KERNEL__ wrapper — only FUNC_IMPL is present.
#                    ptxas runs with `--compile-only` (no launchable kernel
#                    needed) and cuobjdump dumps the cubin to get the
#                    *pure* FUNC_IMPL SASS for line counting.  Important
#                    for short functions (e.g. MUL) where a wrapper would
#                    add a large relative bias to the line count.
#
#   Pass 2 (KERN):   nvcc compiles again with -D__TS_SASS_KERNEL__=1 so
#                    ts_functions.hpp emits a `__global__ __ts_sass_kernel`
#                    that calls FUNC_IMPL.  ptxas runs *without*
#                    `--compile-only`, which is what triggers physical
#                    register allocation and the `Used N registers` line.
#                    The SASS/cubin from this pass is discarded; only the
#                    ptxas log is parsed for the register count.
$(SASS_FILES) : $(SRC)  $(LIB_NAME) | $(OUT)
	@$(CXX) $(COMMONFLAGS) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.mp.ptx
	@$(CXX) $(COMMONFLAGS) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -D__NATIVE_IMPL__ -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.hi.ptx
	@ptxas --compile-only -arch=sm_$(ARCH) $@.mp.ptx -o $@.mp.cubin > $@.mp.log 2>&1 \
	  || { cat $@.mp.log >&2; rm -f $@.mp.log $@.mp.ptx; exit 1; }
	@ptxas --compile-only -arch=sm_$(ARCH) $@.hi.ptx -o $@.hi.cubin > $@.hi.log 2>&1 \
	  || { cat $@.hi.log >&2; rm -f $@.hi.log $@.hi.ptx; exit 1; }
	cuobjdump --dump-sass $@.mp.cubin > $@.mp
	cuobjdump --dump-sass $@.hi.cubin > $@.hi
	@$(CXX) $(COMMONFLAGS) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -D__TS_SASS_KERNEL__=1 -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.mp.kern.ptx
	@$(CXX) $(COMMONFLAGS) $(PTXFLAGS) -D__DEVICE_FUNC__=1 -D__TS_SASS_KERNEL__=1 -D__NATIVE_IMPL__ -dc -x cu $(THIS_DIR)/ts_functions.hpp -ptx -o $@.hi.kern.ptx
	@ptxas -v -arch=sm_$(ARCH) $@.mp.kern.ptx -o $@.mp.kern.cubin > $@.mp.kern.log 2>&1 \
	  || { cat $@.mp.kern.log >&2; rm -f $@.mp.kern.log $@.mp.kern.ptx; exit 1; }
	@ptxas -v -arch=sm_$(ARCH) $@.hi.kern.ptx -o $@.hi.kern.cubin > $@.hi.kern.log 2>&1 \
	  || { cat $@.hi.kern.log >&2; rm -f $@.hi.kern.log $@.hi.kern.ptx; exit 1; }
	rm -f $@.mp.cubin $@.hi.cubin $@.mp.ptx $@.hi.ptx \
	      $@.mp.kern.cubin $@.hi.kern.cubin $@.mp.kern.ptx $@.hi.kern.ptx
	awk '/Function :.*_device_impl/ {in_func=1; print; next} in_func {print} /NOP;/ {if (in_func) {print; in_func=0; exit}}' $@.mp > $@.mp.tmp
	awk '/Function :.*_device_impl/ {in_func=1; print; next} in_func {print} /NOP;/ {if (in_func) {print; in_func=0; exit}}' $@.hi > $@.hi.tmp
	@for tag in mp hi; do \
	  sass=$@.$$tag; tmp=$$sass.tmp; kern_log=$$sass.kern.log; \
	  lines=$$((($$(wc -l < $$tmp) - 4) / 2)); \
	  regs=$$(grep -oE 'Used [0-9]+ registers' $$kern_log | grep -oE '[0-9]+' | sort -n | tail -1); \
	  printf 'ts: %s : %d lines, %s regs\n' \
	    "$$(basename $$sass)" "$$lines" "$${regs:-?}"; \
	  if [ -z "$$regs" ]; then \
	    printf '  [kernel-pass ptxas log size: %s bytes; first 12 lines below]\n' "$$(wc -c < $$kern_log)" >&2; \
	    head -12 $$kern_log | sed "s/^/    | /" >&2; \
	  fi; \
	done
	@rm -f $@.mp.tmp $@.hi.tmp $@.mp.log $@.hi.log $@.mp.kern.log $@.hi.kern.log

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

# Library build flags.
#   Normal: build the test library WITH fp128 so its fp64mp2 math matches the
#   in-test reference (forward the nvcc-only NVVM switch so make/lib.mk uses the
#   same native-binary128 path; lib.mk applies it device-only so g++ never sees
#   it -- this also covers standalone `make -C ts/fpmp`).
#   NOFP128=1: strip fp128 from the library (double fallback) to emulate a
#   customer without fp128 support, while the test TU keeps fp128 for the
#   reference. The explicit empty FP128_EXTRA_FLAGS / FPMP_NVVM_FP128_EXT override
#   any values propagated from the parent `tsmp` command line.
ifeq ($(EMU_NO_FP128),1)
LIB_FP128_ARGS = FP128=0 _CCCL_FPMP_FP128_ENABLE=0 FP128_EXTRA_FLAGS= FPMP_NVVM_FP128_EXT=
else
LIB_FP128_ARGS = _CCCL_FPMP_FP128_ENABLE=1 FPMP_NVVM_FP128_EXT="$(FPMP_NVVM_FP128_EXT)"
endif

$(LIB_NAME):
	$(MAKE) --no-print-directory -f $(FPMP_DIR)/make/lib.mk TARGET=$(TARGET) OUT=$(OUT) $(LIB_FP128_ARGS) lib

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
	@echo "FPMP test suite (ts/fpmp/) Makefile — work in progress"
	@echo ""
	@echo "Usage: make [run|rerun|rebuild|clean|help] [SET=def|all] [FUNC=...] [TYPE=...] [ACCURACY=...] [TARGET=host|device] [LINKAGE=inline|static|lto] [ARCH=..] [VERBOSE=0|1|2|3]"
	@echo ""
	@echo "Targets:"
	@echo "  run      - build and run selected tests (default)"
	@echo "  rerun    - clean, build, and run"
	@echo "  rebuild  - clean and build"
	@echo "  clean    - remove OUT"
	@echo "  help     - show this message"
	@echo ""
	@echo "Core selectors:"
	@echo "  SET      = def|all"
	@echo "  FUNC     = $(ALL_FUNCS) | all"
	@echo "  TYPE     = $(ALL_TYPES) | all"
	@echo "  ACCURACY   = $(ALL_METHODS) | all [alias: ACC]"
	@echo ""
	@echo "Common options:"
	@echo "  TARGET   = device|host (default: device)"
	@echo "  LINKAGE  = inline|static|lto (default: inline)"
	@echo "  ARCH     = CUDA arch (default: 86)"
	@echo "  OUT      = output directory (default: _out)"
	@echo "  VERBOSE  = 0|1|2|3 (default: 0); aliases: V, VERB"
	@echo "  FP2MP    = 1 to enable optimized double->fpmp2 conversion (CCCL_FPMP_OPTIMIZED_DOUBLE_TO_FPMP)"
	@echo "  MP2FP    = 1 to enable optimized fpmp2->double conversion (CCCL_FPMP_OPTIMIZED_FPMP_TO_DOUBLE)"
	@echo "  EXTRA_FLAGS = extra compiler flags"
	@echo ""
	@echo "Test knobs:"
	@echo "  RIGOR    = accuracy rigor level (log2 of samples, e.g., 24=16M, 32=4G)"
	@echo "  RUN_ACCURACY = enable accuracy tests (y=enable (default), n=disable)"
	@echo "  CLASSIFY = enable accuracy classification (y=enable (default), n=disable for faster host runs)"
	@echo "  TIMING   = enable timing tests (y=enable (default), n=disable)"
	@echo "  PRINT    = what to print (ok|fail|warn|all|none)"
	@echo "  DATASET  = datasets to run (work|normal|special|pattern)"
	@echo "  A1/A2/A3/A4 = fixed input arguments for debugging (1 sample, no special table)"
	@echo "  CONSOLE  = console stream (stdout|stderr|file|null)"
	@echo ""
	@echo "Prerequisites:"
	@echo "  When FP128 support is enabled (_CCCL_FPMP_FP128_ENABLE=1),"
	@echo "  libquadmath is required for linking (provides expq, sinq, etc.)."
	@echo "  libquadmath is auto-detected (x86 only). Override with HAS_QUADMATH=1/0."
	@echo "  Current detection: HOST_ARCH=$(HOST_ARCH), HAS_QUADMATH=$(HAS_QUADMATH)"


.PHONY: all clean $(LOG_FILES) build run rerun rebuild logs help

# silent mode
ifneq (,$(filter $(VERBOSE_MODE), 0 1))
.SILENT:
endif
