#include "operations.hpp"
#include <cstring>

// 辅助函数：获取填充后的输入值 (NCHW布局)
static float get_padded_value(const float* input, 
                            int batch_idx, int channel, int height, int width,
                            int row, int col, int padding) {
    // 如果位置在padding区域，返回0
    if (row < 0 || row >= height || col < 0 || col >= width) {
        return 0.0f;
    }
    // 返回实际的输入值 (NCHW布局)
    return input[((batch_idx * channel) * height + row) * width + col];
}

void conv2d(const float* input, float* output, const float* weight, const float* bias,
            int batch_size, int in_height, int in_width, int in_channels,
            int out_channels, int kernel_size, int stride, int padding) {
    // 计算输出尺寸
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // 对每个样本进行处理 (N)
    for (int n = 0; n < batch_size; n++) { // n: batch_size
        // 对每个输出通道进行处理 (C_out)
        for (int c_out = 0; c_out < out_channels; c_out++) { // c_out: out_channels
            // 初始化该通道的偏置
            float bias_val = bias[c_out];
            
            // 对输出特征图的每个位置进行处理
            for (int h_out = 0; h_out < out_height; h_out++) { // h_out: out_height
                for (int w_out = 0; w_out < out_width; w_out++) { // w_out: out_width
                    float sum = bias_val;
                    
                    // 对每个输入通道进行求和 (C_in)
                    for (int c_in = 0; c_in < in_channels; c_in++) { // c_in: in_channels
                        // 执行2D交叉相关
                        for (int kh = 0; kh < kernel_size; kh++) { // kh: kernel_size
                            for (int kw = 0; kw < kernel_size; kw++) { // kw: kernel_size
                                int h_in = h_out * stride + kh - padding;
                                int w_in = w_out * stride + kw - padding;
                                float in_val = get_padded_value(input, n, c_in, in_height, in_width, h_in, w_in, padding);
                                float w_val = weight[((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw];
                                sum += in_val * w_val;
                            }
                        }
                    }
                    
                    // 存储输出值 (NCHW布局)
                    output[((n * out_channels + c_out) * out_height + h_out) * out_width + w_out] = sum;
                }
            }
        }
    }
}
