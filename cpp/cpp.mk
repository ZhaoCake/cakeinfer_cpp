# C++ related configurations
CXX := g++
CXXFLAGS := -std=c++11 -Wall -I./include

# Source directories
SRC_DIR := src
TEST_DIR := tests
BUILD_DIR := build

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/operations/*.cpp)
OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

# Test source files
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS := $(TEST_SRCS:%.cpp=$(BUILD_DIR)/%.o)

# Create build directories
$(shell mkdir -p $(BUILD_DIR)/src/operations $(BUILD_DIR)/tests)

# Build rules
$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Library target
$(BUILD_DIR)/liblenet.a: $(OBJS)
	ar rcs $@ $^

# Test targets
$(BUILD_DIR)/tests/test_conv2d: $(BUILD_DIR)/tests/test_conv2d.o $(BUILD_DIR)/tests/test_utils.o $(BUILD_DIR)/liblenet.a
	$(CXX) $^ -o $@ -lm

$(BUILD_DIR)/tests/test_lenet: $(BUILD_DIR)/tests/test_lenet.o $(BUILD_DIR)/tests/test_utils.o $(BUILD_DIR)/liblenet.a
	$(CXX) $^ -o $@ -lm

# Phony targets
.PHONY: build clean-cpp

build: $(BUILD_DIR)/liblenet.a $(BUILD_DIR)/tests/test_conv2d $(BUILD_DIR)/tests/test_lenet

clean-cpp:
	rm -rf $(BUILD_DIR) 