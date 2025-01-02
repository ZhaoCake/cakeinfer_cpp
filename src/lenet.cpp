#include "lenet.hpp"
#include "operations.hpp"
#include "weights.hpp"
#include <algorithm>
#include <cstring>

LeNet::LeNet() {
    init_buffers();
}

void LeNet::init_buffers() {
    // 分配中间结果缓冲区
    conv1_output.resize(1 * 6 * 24 * 24);
    conv1_relu_output.resize(1 * 6 * 24 * 24);
    pool1_output.resize(1 * 6 * 12 * 12);
    conv2_output.resize(1 * 16 * 8 * 8);
    conv2_relu_output.resize(1 * 16 * 8 * 8);
    pool2_output.resize(1 * 16 * 4 * 4);
    fc1_input.resize(1 * 16 * 4 * 4);
    fc1_output.resize(1 * 120);
    fc1_relu_output.resize(1 * 120);
    fc2_output.resize(1 * 84);
    fc2_relu_output.resize(1 * 84);
    fc3_output.resize(1 * 10);
}

void LeNet::preprocess(const uint8_t* input, float* output) {
    // 将输入图像转换为float并归一化到[0,1]
    for (int i = 0; i < 28 * 28; i++) {
        output[i] = input[i] / 255.0f;
    }
}

bool LeNet::forward(const float* input, float* output) {
    // Conv1 + ReLU + MaxPool
    conv2d(input, conv1_output.data(), 
           reinterpret_cast<float*>(weights_conv1), bias_conv1,
           1, 28, 28, 1, 6, 5, 1, 0);
    
    relu(conv1_output.data(), conv1_relu_output.data(),
         1, 24, 24, 6);
    
    maxpool2d(conv1_relu_output.data(), pool1_output.data(),
              1, 24, 24, 6, 12, 12, 2, 2);

    // Conv2 + ReLU + MaxPool
    conv2d(pool1_output.data(), conv2_output.data(),
           reinterpret_cast<float*>(weights_conv2), bias_conv2,
           1, 12, 12, 6, 16, 5, 1, 0);
    
    relu(conv2_output.data(), conv2_relu_output.data(),
         1, 8, 8, 16);
    
    maxpool2d(conv2_relu_output.data(), pool2_output.data(),
              1, 8, 8, 16, 4, 4, 2, 2);

    // 展平操作
    std::memcpy(fc1_input.data(), pool2_output.data(), sizeof(float) * 16 * 4 * 4);

    // FC1 + ReLU
    linear(fc1_input.data(), fc1_output.data(),
           reinterpret_cast<float*>(weights_fc1), bias_fc1,
           1, 16 * 4 * 4, 120);
    
    relu(fc1_output.data(), fc1_relu_output.data(),
         1, 1, 1, 120);

    // FC2 + ReLU
    linear(fc1_relu_output.data(), fc2_output.data(),
           reinterpret_cast<float*>(weights_fc2), bias_fc2,
           1, 120, 84);
    
    relu(fc2_output.data(), fc2_relu_output.data(),
         1, 1, 1, 84);

    // FC3
    linear(fc2_relu_output.data(), fc3_output.data(),
           reinterpret_cast<float*>(weights_fc3), bias_fc3,
           1, 84, 10);

    // 复制最终输出
    std::memcpy(output, fc3_output.data(), sizeof(float) * 10);

    return true;
}

int LeNet::postprocess(const float* input) {
    // 找到最大值的索引
    return std::max_element(input, input + 10) - input;
}