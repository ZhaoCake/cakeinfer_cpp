# LeNet written with hands

> The aim of my computer not work well again, so I can only use English to write my log. Damn, This will slow down my speed to think certainly. But improve my English too.

## Network Structure

LeNet-5 consists of the following layers:

1. Input Layer: 28x28 grayscale image (MNIST format)

2. First Convolutional Block:
   - Conv2d: 1->6 channels, 5x5 kernel, padding=2
   - ReLU activation
   - MaxPool2d: 2x2 kernel, stride=2
   Output: 6x14x14

3. Second Convolutional Block:
   - Conv2d: 6->16 channels, 5x5 kernel
   - ReLU activation
   - MaxPool2d: 2x2 kernel, stride=2
   Output: 16x5x5

4. Fully Connected Layers:
   - Flatten: 16x5x5 = 400
   - FC1: 400->120 + ReLU
   - FC2: 120->84 + ReLU
   - FC3: 84->10 (output)

## Implementation Details

### Data Layout

- Input: NCHW format (PyTorch default)
- Weights: 
  - Conv: [out_channels, in_channels, kernel_h, kernel_w]
  - FC: [out_features, in_features]

### Memory Management

- Static arrays for weights and biases
- Dynamic vectors for intermediate results
- Contiguous memory layout for better cache performance

### Key Operations

1. Convolution:
   ```cpp
   void conv2d(const float* input, float* output, const float* weight, const float* bias,
               int batch_size, int in_height, int in_width, int in_channels,
               int out_channels, int kernel_size, int stride, int padding)
   ```

2. MaxPooling:
   ```cpp
   void maxpool2d(const float* input, float* output,
                  int batch_size, int in_height, int in_width, int in_channels,
                  int out_height, int out_width, int kernel_size, int stride)
   ```

3. ReLU:
   ```cpp
   void relu(const float* input, float* output,
             int batch_size, int height, int width, int channels)
   ```

## Testing Strategy

1. Layer-wise Testing:
   - Test each operation independently
   - Compare with PyTorch outputs
   - Use small, verifiable inputs

2. End-to-end Testing:
   - Test full network inference
   - Use real MNIST images
   - Compare with PyTorch model

## Known Issues

1. Conv2d Implementation:
   - Need to verify padding behavior
   - Memory access pattern could be optimized
   - SIMD optimization pending

2. Memory Layout:
   - Current implementation uses NCHW
   - Consider NHWC for potential optimization

3. Performance:
   - Basic implementation without optimization
   - No SIMD/OpenMP usage yet
   - Memory bandwidth could be improved

## Next Steps

1. Optimization:
   - Implement SIMD instructions
   - Add OpenMP support
   - Optimize memory access patterns

2. HLS Integration:
   - Add HLS pragmas
   - Analyze hardware implications
   - Pipeline and unroll strategies

3. Testing:
   - Add more test cases
   - Benchmark performance
   - Memory usage analysis

## References

1. LeNet-5 Paper
2. PyTorch Documentation
3. HLS User Guide