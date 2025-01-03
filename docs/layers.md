# Network Layers Implementation

## Convolution Layer (Conv2d)

### Implementation Details
```cpp
void conv2d(const float* input, float* output, const float* weight, const float* bias,
            int batch_size, int in_height, int in_width, int in_channels,
            int out_channels, int kernel_size, int stride, int padding)
```

1. Memory Layout:
   - Input: [N, C_in, H, W]
   - Weight: [C_out, C_in, K, K]
   - Output: [N, C_out, H_out, W_out]

2. Algorithm:
   ```
   for n in batch_size:
     for c_out in out_channels:
       for h_out in out_height:
         for w_out in out_width:
           sum = bias[c_out]
           for c_in in in_channels:
             for kh in kernel_size:
               for kw in kernel_size:
                 h_in = h_out * stride + kh - padding
                 w_in = w_out * stride + kw - padding
                 if h_in >= 0 && h_in < in_height && w_in >= 0 && w_in < in_width:
                   sum += input[n,c_in,h_in,w_in] * weight[c_out,c_in,kh,kw]
           output[n,c_out,h_out,w_out] = sum
   ```

3. Key Points:
   - Uses cross-correlation (not convolution)
   - Handles padding with zeros
   - Maintains NCHW format throughout

## MaxPool Layer

### Implementation Details
```cpp
void maxpool2d(const float* input, float* output,
               int batch_size, int in_height, int in_width, int in_channels,
               int out_height, int out_width, int kernel_size, int stride)
```

1. Memory Layout:
   - Input: [N, C, H, W]
   - Output: [N, C, H_out, W_out]

2. Algorithm:
   ```
   for n in batch_size:
     for c in channels:
       for h_out in out_height:
         for w_out in out_width:
           max_val = -inf
           for kh in kernel_size:
             for kw in kernel_size:
               h_in = h_out * stride + kh
               w_in = w_out * stride + kw
               if h_in < in_height && w_in < in_width:
                 val = input[n,c,h_in,w_in]
                 max_val = max(max_val, val)
           output[n,c,h_out,w_out] = max_val
   ```

3. Key Points:
   - No padding support (not needed for LeNet)
   - In-bounds checking for window operations

## ReLU Activation

### Implementation Details
```cpp
void relu(const float* input, float* output,
          int batch_size, int height, int width, int channels)
```

1. Memory Layout:
   - Input/Output: [N, C, H, W]

2. Algorithm:
   ```
   size = batch_size * channels * height * width
   for i in size:
     output[i] = max(input[i], 0)
   ```

3. Key Points:
   - Simple element-wise operation
   - Potential for vectorization
   - Cache-friendly linear memory access

## Linear/Fully Connected Layer

### Implementation Details
```cpp
void linear(const float* input, float* output, const float* weight, const float* bias,
           int batch_size, int in_features, int out_features)
```

1. Memory Layout:
   - Input: [N, in_features]
   - Weight: [out_features, in_features]
   - Output: [N, out_features]

2. Algorithm:
   ```
   for n in batch_size:
     for out in out_features:
       sum = bias[out]
       for in in in_features:
         sum += input[n * in_features + in] * weight[out * in_features + in]
       output[n * out_features + out] = sum
   ```

3. Key Points:
   - Matrix multiplication based
   - Row-major memory layout
   - Bias addition included

## Optimization Opportunities

1. Memory Access:
   - Use blocking/tiling for cache efficiency
   - Align data for SIMD operations
   - Consider data layout transformations

2. Parallelization:
   - OpenMP for batch processing
   - SIMD for inner loops
   - Thread-level parallelism

3. HLS Specific:
   - Loop unrolling for parallel computation
   - Pipeline for throughput
   - Memory partitioning for concurrent access 