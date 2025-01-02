#include "operations.hpp"

void relu(const float* input, float* output, 
          int batch_size, int height, int width, int channels) {
    int size = batch_size * height * width * channels;
    
    // 对所有元素应用ReLU
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
} 