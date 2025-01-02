#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

/**
 * 2D卷积操作
 * @param input 输入特征图 [batch_size, in_height, in_width, in_channels]
 * @param output 输出特征图 [batch_size, out_height, out_width, out_channels]
 * @param weight 卷积核权重 [out_channels, in_channels, kernel_size, kernel_size]
 * @param bias 偏置项 [out_channels]
 * @param batch_size 批次大小
 * @param in_height 输入高度
 * @param in_width 输入宽度
 * @param in_channels 输入通道数
 * @param out_channels 输出通道数
 * @param kernel_size 卷积核大小
 * @param stride 步长
 * @param padding 填充大小
 */
void conv2d(const float* input, float* output, const float* weight, const float* bias,
            int batch_size, int in_height, int in_width, int in_channels,
            int out_channels, int kernel_size, int stride, int padding);

/**
 * 2D最大池化操作
 * @param input 输入特征图 [batch_size, in_height, in_width, in_channels]
 * @param output 输出特征图 [batch_size, out_height, out_width, out_channels]
 * @param batch_size 批次大小
 * @param in_height 输入高度
 * @param in_width 输入宽度
 * @param in_channels 输入通道数
 * @param out_height 输出高度
 * @param out_width 输出宽度
 * @param kernel_size 池化核大小
 * @param stride 步长
 */
void maxpool2d(const float* input, float* output, int batch_size, int in_height, int in_width, int in_channels,
            int out_height, int out_width, int kernel_size, int stride);

/**
 * 全连接层操作
 * @param input 输入数据 [batch_size, in_features]
 * @param output 输出数据 [batch_size, out_features]
 * @param weight 权重矩阵 [out_features, in_features]
 * @param bias 偏置向量 [out_features]
 * @param batch_size 批次大小
 * @param in_features 输入特征数
 * @param out_features 输出特征数
 */
void fc(const float* input, float* output, const float* weight, const float* bias,
        int batch_size, int in_features, int out_features);

/**
 * ReLU激活函数
 * @param input 输入数据
 * @param output 输出数据
 * @param batch_size 批次大小
 * @param height 高度
 * @param width 宽度
 * @param channels 通道数
 */
void relu(const float* input, float* output, 
          int batch_size, int height, int width, int channels);

/**
 * 线性层操作
 * @param input 输入数据
 * @param output 输出数据
 * @param weight 权重矩阵
 * @param bias 偏置向量
 * @param batch_size 批次大小
 * @param in_features 输入特征数
 * @param out_features 输出特征数
 */
void linear(const float* input, float* output, const float* weight, const float* bias,
           int batch_size, int in_features, int out_features);

#endif // OPERATIONS_HPP 
