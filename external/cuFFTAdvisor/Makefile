SRC_DIR := .
OBJ_DIR := obj
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))
LDFLAGS := --cudart static -shared -link -Wno-deprecated-gpu-targets
CPPFLAGS := ...
CXXFLAGS := -O3 --use_fast_math -Xcompiler -fPIC -std=c++11 -Wno-deprecated-gpu-targets

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: libcuFFTAdvisor.so

# Tool invocations
libcuFFTAdvisor.so: $(OBJ_FILES)
	nvcc $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(@D)
	nvcc $(CXXFLAGS) -c -o $@ $<

# Other Targets
clean:
	rm -rf $(OBJ_DIR) libcuFFTAdvisor.so
