###########################################################################################
#
#  File:    Makefile
#  Purpose: Build all benchmarks in the folder
#  Author:  Andrei Kolesov
#  Date:    2025-04-15
#
#                       USER GUIDE
#                       ==========
#
#  Makefile parameters:
#  --------------------
#    `TARGET`    = target device (`device` (default), `host`)
#    `OUT`       = output directory (default is `_out`)
#    `LINKAGE`   = use library or inline emulation cores (`inline` (default), `static`, `lto`)
#    `VERBOSE`   = verbose mode (`0` (silent), `1` (minimal), `2` (full), `3` (very verbose)); aliases: `V`, `VERB`
#    `BENCHMARK` = benchmark(s) to run/build (default: all benchmark subdirectories)
#
#  Note:
#  -----
#  This Makefile is a thin wrapper that invokes each benchmark's own Makefile.
#  Variables such as `TARGET`, `OUT`, `LINKAGE`, `VERBOSE`, etc. are forwarded
#  to sub-makes automatically when passed on the command line.
#
#  Build targets:
#  --------------
#    `all` (default) - build all benchmarks
#    `run`           - run all benchmarks
#    `rerun`         - rerun all benchmarks
#    `clean`         - delete all generated files and folders
#
#  Usage examples:
#  ---------------
#    `make`                       - build and run all benchmarks
#    `make rerun`                 - rebuild and run all benchmarks
#    `make TARGET=host rerun`     - rebuild and run all benchmarks on host
#    `make BENCHMARK=fft rerun`   - rebuild and run a single benchmark
#    `make clean`                 - delete all generated files and folders
#
###########################################################################################

.NOTPARALLEL:

# current directory
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Output directory
OUT  ?= $(THIS_DIR)/_out

# Find all benchmark directories (excluding those starting with underscore)
BENCHMARKS_ALL := $(shell find . -maxdepth 1 -type d -not -name ".*" -not -name "_*" -not -name ".")

# Benchmarks excluded from default runs (long build/run time); run them
# explicitly with BENCHMARK=lulesh / BENCHMARK=hpc_exp_error.
SKIP_BENCHMARKS := ./lulesh ./hpc_exp_error
BENCHMARKS_DEFAULT := $(filter-out $(SKIP_BENCHMARKS),$(BENCHMARKS_ALL))
BENCHMARK ?= $(BENCHMARKS_DEFAULT)

help:
	@echo "FPMP benchmarks Makefile"
	@echo ""
	@echo "Usage: make [run|rerun|clean|help] [BENCHMARK=...] [TARGET=host|device] [LINKAGE=inline|static|lto] [OUT=...] [VERBOSE=0|1|2|3]"
	@echo ""
	@echo "Available benchmarks:"
	@echo "  $(notdir $(BENCHMARKS_ALL))"
	@echo ""
	@echo "Default benchmarks (run by CI):"
	@echo "  $(notdir $(BENCHMARKS_DEFAULT))"
	@echo ""
	@echo "Skipped by default (long compile time):"
	@echo "  $(notdir $(SKIP_BENCHMARKS))"
	@echo ""
	@echo "Notes:"
	@echo "  - BENCHMARK defaults to all except skipped benchmarks."
	@echo "  - Use BENCHMARK=lulesh or BENCHMARK=hpc_exp_error to run skipped benchmarks explicitly."
	@echo "  - TARGET/LINKAGE/OUT/VERBOSE are passed through to each benchmark via recursive make."

all: rerun

# Run all benchmarks
$(BENCHMARK):
	@echo "\nRunning benchmark: $@"
	@echo "==================================\n"
	@if [ -f "$@/Makefile" ]; then \
		$(MAKE) --no-print-directory -f $@/Makefile $(RUN_TARGET); \
	else \
		echo "No Makefile found in $@"; \
	fi

# Run all benchmarks
run: RUN_TARGET=run
run: $(BENCHMARK)

# Rerun all benchmarks
rerun: RUN_TARGET=rerun
rerun: $(BENCHMARK)

# Clean up
clean:
	rm -fr $(OUT)

.PHONY: help all run rerun clean $(BENCHMARK) 
