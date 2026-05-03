# Makefile — OpenCL Monte Carlo Option Pricer
# Build: make all
# Run:   make run   (or ./mc_pricer_ocl [options])
# Clean: make clean

CXX       = g++
CC        = gcc
CXXFLAGS  = -O2 -Wall -Wextra -Wno-deprecated-declarations \
            -std=c++14 -I include -DCL_TARGET_OPENCL_VERSION=300
CFLAGS    = -O2 -Wall -Wextra -std=c11 -I include
LDFLAGS   = -lOpenCL -lm

BUILD_DIR = build
BIN       = mc_pricer_ocl

# Sources
CPP_SRCS  = src/main.cpp src/ocl_helpers.cpp
C_SRCS    = src/black_scholes.c src/cpu_baseline.c

# Objects
CPP_OBJS  = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
C_OBJS    = $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(C_SRCS))
ALL_OBJS  = $(CPP_OBJS) $(C_OBJS)

.PHONY: all clean run

all: $(BUILD_DIR) $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN): $(ALL_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: all
	./$(BIN)

clean:
	rm -rf $(BUILD_DIR) $(BIN)
