#include "weights.hpp"
#include <iostream>
#include <fstream>
#include <string>

// 定义权重数组
float weights_conv1[6][1][5][5];
float weights_conv2[16][6][5][5];
float weights_fc1[120][16*5*5];
float weights_fc2[84][120];
float weights_fc3[10][84];

float bias_conv1[6];
float bias_conv2[16];
float bias_fc1[120];
float bias_fc2[84];
float bias_fc3[10];

static bool load_binary_file(const std::string& filename, void* buffer, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(buffer), size);
    if (file.gcount() != size) {
        std::cerr << "Failed to read file: " << filename 
                  << " (expected " << size << " bytes, got " 
                  << file.gcount() << " bytes)" << std::endl;
        return false;
    }
    
    return true;
}

bool load_weights() {
    // 加载卷积层权重
    if (!load_binary_file("./resources/weights/conv1_weight.bin", weights_conv1, 
                         sizeof(weights_conv1))) {
        return false;
    }
    if (!load_binary_file("./resources/weights/conv2_weight.bin", weights_conv2,
                         sizeof(weights_conv2))) {
        return false;
    }
    
    // 加载全连接层权重
    if (!load_binary_file("./resources/weights/fc_0_weight.bin", weights_fc1,
                         sizeof(weights_fc1))) {
        return false;
    }
    if (!load_binary_file("./resources/weights/fc_2_weight.bin", weights_fc2,
                         sizeof(weights_fc2))) {
        return false;
    }
    if (!load_binary_file("./resources/weights/fc_4_weight.bin", weights_fc3,
                         sizeof(weights_fc3))) {
        return false;
    }
    
    // 加载偏置
    if (!load_binary_file("./resources/weights/conv1_bias.bin", bias_conv1,
                         sizeof(bias_conv1))) {
        return false;
    }
    if (!load_binary_file("./resources/weights/conv2_bias.bin", bias_conv2,
                         sizeof(bias_conv2))) {
        return false;
    }
    if (!load_binary_file("./resources/weights/fc_0_bias.bin", bias_fc1,
                         sizeof(bias_fc1))) {
        return false;
    }
    if (!load_binary_file("./resources/weights/fc_2_bias.bin", bias_fc2,
                         sizeof(bias_fc2))) {
        return false;
    }
    if (!load_binary_file("./resources/weights/fc_4_bias.bin", bias_fc3,
                         sizeof(bias_fc3))) {
        return false;
    }
    
    return true;
}

void free_weights(void) {
    // 如果后续需要动态分配内存，在这里释放
}

