# Python related tasks
.PHONY: train test clean-data convert generate-test test-conv test-lenet

# Training
train:
	python python/train.py

# Inference
infer:
	python python/infer.py

# Convert model to C readable format
convert:
	python python/convert.py

# Generate test data
generate-test:
	mkdir -p resources/test
	python python/test_utils.py

# Run C tests
test-conv: generate-test build
	./build/tests/test_conv2d

# Run LeNet tests
test-lenet: build
	./build/tests/test_lenet

# Clean downloaded MNIST data
clean-data:
	rm -rf resources/mnist

# Clean all generated files
clean-all: clean-data
	rm -f resources/lenet.pth
	rm -f resources/predictions.png
	rm -rf resources/weights
	rm -f resources/model_config.json
	rm -rf resources/test 

# Update test target
test: test-conv test-lenet 