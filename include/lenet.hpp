#ifndef LENET_HPP
#define LENET_HPP

#include <vector>
#include <cstdint>

class LeNet {
public:
    LeNet();
    ~LeNet() = default;

    /**
     * 对输入图像进行预处理
     * @param input 输入图像数据 [28x28]
     * @param output 预处理后的数据 [1x1x28x28]
     */
    static void preprocess(const uint8_t* input, float* output);

    /**
     * 执行推理
     * @param input 预处理后的输入数据 [1x1x28x28]
     * @param output 网络输出 [1x10]
     * @return 是否成功
     */
    bool forward(const float* input, float* output);

    /**
     * 对网络输出进行后处理
     * @param input 网络输出 [1x10]
     * @return 预测的数字 (0-9)
     */
    static int postprocess(const float* input);

private:
    // 中间结果缓冲区
    std::vector<float> conv1_output;      // [1, 6, 24, 24]
    std::vector<float> conv1_relu_output; // [1, 6, 24, 24]
    std::vector<float> pool1_output;      // [1, 6, 12, 12]
    std::vector<float> conv2_output;      // [1, 16, 8, 8]
    std::vector<float> conv2_relu_output; // [1, 16, 8, 8]
    std::vector<float> pool2_output;      // [1, 16, 4, 4]
    std::vector<float> fc1_input;         // [1, 16*4*4]
    std::vector<float> fc1_output;        // [1, 120]
    std::vector<float> fc1_relu_output;   // [1, 120]
    std::vector<float> fc2_output;        // [1, 84]
    std::vector<float> fc2_relu_output;   // [1, 84]
    std::vector<float> fc3_output;        // [1, 10]

    // 初始化缓冲区
    void init_buffers();
};

#endif // LENET_HPP