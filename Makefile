# Import related tasks
include python/python.mk
include cpp/cpp.mk

# Default target
.PHONY: all
all: train convert

# Clean everything
.PHONY: clean
clean: clean-all clean-cpp

# Test targets
.PHONY: test
test: test-conv 