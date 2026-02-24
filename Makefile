# ImageClassifier Makefile

# Override from CLI/env when needed, e.g.:
#   make CXX=clang++ CONFIG=debug
CXX ?= g++
CONFIG ?= release
OPENMP ?= 1

BUILD_DIR := build
MODEL_DIR := models

CPPFLAGS := -Iinclude
CXXFLAGS_BASE := -std=c++17 -Wall -Wextra
CXXFLAGS_RELEASE := -O2
CXXFLAGS_DEBUG := -O0 -g
LDFLAGS :=
LDLIBS :=

ifeq ($(CONFIG),debug)
  CXXFLAGS := $(CXXFLAGS_BASE) $(CXXFLAGS_DEBUG)
else
  CXXFLAGS := $(CXXFLAGS_BASE) $(CXXFLAGS_RELEASE)
endif

ifeq ($(OPENMP),1)
  CXXFLAGS += -fopenmp
  LDFLAGS += -fopenmp
else
  CXXFLAGS += -Wno-unknown-pragmas
endif

CORE_SRC := src/core/classifier.cpp src/core/dataset.cpp
NN_SRC := src/classifiers/neural_net.cpp
KNN_SRC := src/classifiers/knn.cpp

TRAIN_SRC := src/apps/train.cpp $(CORE_SRC) $(NN_SRC) $(KNN_SRC)
PREDICT_SRC := src/apps/predict.cpp $(CORE_SRC) $(NN_SRC)
TUI_SRC := src/apps/tui.cpp $(CORE_SRC) $(NN_SRC) $(KNN_SRC)

TRAIN_BIN := $(BUILD_DIR)/train
PREDICT_BIN := $(BUILD_DIR)/predict
TUI_BIN := $(BUILD_DIR)/tui

.PHONY: all clean dirs train predict tui test run-train run-predict run-tui format-check help

all: dirs train predict tui

dirs:
	@mkdir -p $(BUILD_DIR) $(MODEL_DIR)

train: $(TRAIN_BIN)

predict: $(PREDICT_BIN)

tui: $(TUI_BIN)

test: $(TRAIN_BIN) $(PREDICT_BIN) $(TUI_BIN)
	@echo "Running smoke tests..."
	@./$(TRAIN_BIN) --help >/dev/null
	@./$(PREDICT_BIN) --help >/dev/null
	@printf '4\n' | ./$(TUI_BIN) >/dev/null
	@echo "Smoke tests passed."

$(TRAIN_BIN): $(TRAIN_SRC) | dirs
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(PREDICT_BIN): $(PREDICT_SRC) | dirs
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(TUI_BIN): $(TUI_SRC) | dirs
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

run-train: $(TRAIN_BIN)
	./$(TRAIN_BIN)

run-predict: $(PREDICT_BIN)
	./$(PREDICT_BIN)

run-tui: $(TUI_BIN)
	./$(TUI_BIN)

format-check:
	@echo "No formatter configured."

clean:
	rm -rf $(BUILD_DIR)/*
	rm -f $(MODEL_DIR)/*.model

help:
	@echo "Usage:"
	@echo "  make [all|train|predict|tui|test|clean|help]"
	@echo ""
	@echo "Build config:"
	@echo "  CONFIG=release|debug   (default: release)"
	@echo "  OPENMP=1|0             (default: 1)"
	@echo "  CXX=<compiler>         (default: g++)"
	@echo ""
	@echo "Run helpers:"
	@echo "  make run-train"
	@echo "  make run-predict"
	@echo "  make run-tui"
	@echo "  make test"
	@echo ""
	@echo "Examples:"
	@echo "  make -j4"
	@echo "  make CONFIG=debug"
	@echo "  make OPENMP=0"
