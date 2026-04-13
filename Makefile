NVCC      = nvcc
CXX       = g++
NVCCFLAGS = -O2 -arch=sm_86 -std=c++14 --allow-unsupported-compiler -I include
CXXFLAGS  = -O2 -I include
LDFLAGS   = -lcurand -lm -lncurses -lGL -lGLEW -lglfw

SRC_DIR   = src
BUILD_DIR = build
BIN       = mc_pricer

# Source files
CU_SRCS   = $(SRC_DIR)/main.cu $(SRC_DIR)/monte_carlo.cu \
            $(SRC_DIR)/reduction.cu $(SRC_DIR)/black_scholes.cu \
            $(SRC_DIR)/render_ppm.cu $(SRC_DIR)/greeks.cu \
            $(SRC_DIR)/streams.cu $(SRC_DIR)/dashboard.cu \
            $(SRC_DIR)/gl_dashboard.cu
CPP_SRCS  = $(SRC_DIR)/cpu_baseline.cpp

# Object files
CU_OBJS   = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
CPP_OBJS  = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
ALL_OBJS  = $(CU_OBJS) $(CPP_OBJS)

.PHONY: all clean run debug

all: $(BUILD_DIR) $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN): $(ALL_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c -o $@ $<

debug: NVCCFLAGS += -g -G -DDEBUG
debug: clean all

run: all
	./$(BIN)

clean:
	rm -rf $(BUILD_DIR) $(BIN)
