CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
GPP := g++

# Compiler and linker flags
NVCC_FLAGS := -std=c++20 -arch=sm_86 --expt-relaxed-constexpr -Xcompiler -fPIC,-O3
GPP_FLAGS := -Iinclude -std=c++20 -march=native -O3 -Wall -fPIC -fopenmp -flto
LD_FLAGS := -L$(CUDA_PATH)/lib64 -L$(CONDA_PREFIX)/lib -lcudart -lcustatevec -llbfgs -lstdc++ -lgomp

# Additional include directories
INCLUDE := -I$(CUDA_PATH)/include -I./include -I$(CONDA_PREFIX)/include

# Library and binary target
LIB_TARGET := lib/libcuaoalg.so
TARGET := $(LIB_TARGET)

# Source and object files
CU_SRC := $(wildcard src/**/*.cu) $(wildcard src/*.cu)
CPP_SRC := $(wildcard src/**/*.cpp) $(wildcard src/*.cpp)
CU_OBJ := $(CU_SRC:src/%.cu=obj/%.o)
CPP_OBJ := $(CPP_SRC:src/%.cpp=obj/%.o)
OBJ := $(CU_OBJ) $(CPP_OBJ)

# Colors for printing
RED := \033[0;31m
GREEN := \033[0;32m
NC := \033[0m

all: $(TARGET)

$(LIB_TARGET): $(filter-out obj/main.o, $(OBJ))
	@echo -e "${GREEN}Building library${NC}"
	@mkdir -p $(@D)
	@$(GPP) $(GPP_FLAGS) $(INCLUDE) $^ -shared -o $@ $(LD_FLAGS)

obj/%.o: src/%.cu
	@echo -e "${GREEN}Compiling CUDA source${NC} $<"
	@mkdir -p $(@D)
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDE) -c $< -o $@

obj/%.o: src/%.cpp
	@echo -e "${GREEN}Compiling C++ source${NC} $<"
	@mkdir -p $(@D)
	@$(GPP) $(GPP_FLAGS) $(INCLUDE) -c $< -o $@

clean:
	@echo -e "${RED}Cleaning up...${NC}"
	@rm -rf obj $(TARGET)

.PHONY: all clean

