#include "operations.hpp"

void fc(const float* input, float* output, const float* weight, const float* bias,
        int batch_size, int in_features, int out_features) {
    // 对每个样本进行处理 (N)
    for (int n = 0; n < batch_size; n++) {
        // 对每个输出特征进行处理
        for (int out = 0; out < out_features; out++) {
            float sum = bias[out];
            
            // 计算输入特征的加权和
            for (int in = 0; in < in_features; in++) {
                sum += input[n * in_features + in] * weight[out * in_features + in];
            }
            
            // 存储输出值
            output[n * out_features + out] = sum;
        }
    }
}


