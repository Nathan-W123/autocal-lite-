# =============================================================================================
# AutoCal-lite convenience targets (layout reused from Aether6's Makefile).
#
#   make configure     configure the CMake build (Release)
#   make build         build the library, applications and tests
#   make test          run the Catch2 suite (make ctest runs it through CTest)
#   make run           one 60 s convergence run, prints its scores
#   make monte-carlo   the 50-run campaigns behind the results table
#   make eval          campaigns + sensitivity checks + every results table + plot data (+ figures)
#   make clean         remove the build directory
# =============================================================================================

BUILD_DIR      ?= build
BUILD_TYPE     ?= Release
JOBS           ?= $(shell nproc 2>/dev/null || echo 4)
PYTHON         ?= python3
SCENARIO       ?= convergence
TRIALS         ?= 50
CMAKE_ARGS     ?=

BIN            := $(BUILD_DIR)/bin

.PHONY: all configure build test ctest run monte-carlo eval clean help

all: build

help:
	@sed -n '4,11p' Makefile

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) $(CMAKE_ARGS)

build: configure
	cmake --build $(BUILD_DIR) -j $(JOBS)

test: build
	$(BIN)/autocal_tests

ctest: build
	cd $(BUILD_DIR) && ctest --output-on-failure

run: build
	$(BIN)/autocal_run --scenario $(SCENARIO)

monte-carlo: build
	$(BIN)/autocal_mc --scenario all --trials $(TRIALS)

eval: build
	$(PYTHON) python/scripts/eval.py --trials $(TRIALS) --binary $(BIN)/autocal_mc

clean:
	rm -rf $(BUILD_DIR)
