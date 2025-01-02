#include "operations.hpp"
#include <cfloat>

void maxpool2d(const float* input, float* output, 
               int batch_size, int in_height, int in_width, int in_channels,
               int out_height, int out_width, int kernel_size, int stride) {
    // 对每个样本进行处理 (N)
    for (int n = 0; n < batch_size; n++) {
        // 对每个通道进行处理 (C)
        for (int c = 0; c < in_channels; c++) {
            // 对输出特征图的每个位置进行处理
            for (int h_out = 0; h_out < out_height; h_out++) {
                for (int w_out = 0; w_out < out_width; w_out++) {
                    float max_val = -FLT_MAX;
                    
                    // 在kernel_size x kernel_size的窗口中找最大值
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int h_in = h_out * stride + kh;
                            int w_in = w_out * stride + kw;
                            
                            if (h_in < in_height && w_in < in_width) {
                                float val = input[((n * in_channels + c) * in_height + h_in) * in_width + w_in];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    
                    // 存储最大值
                    output[((n * in_channels + c) * out_height + h_out) * out_width + w_out] = max_val;
                }
            }
        }
    }
} 