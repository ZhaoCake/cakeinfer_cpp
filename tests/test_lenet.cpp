#include "lenet.hpp"
#include "test_utils.hpp"
#include "weights.hpp"
#include <iostream>

// 测试数据
const uint8_t test_image[28*28] = {
    // @Data 这里应该是一个28x28的MNIST图像数据
    // 为了示例，我们使用一个简单的数字"1"图像
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// 期望的输出 (对应数字"1"的预测结果)
const float expected_output[10] = {
    -2.85954f,  // 0
     7.85236f,  // 1 (最大值)
    -3.85605f,  // 2
    -2.85734f,  // 3
    -4.28932f,  // 4
    -3.74498f,  // 5
    -4.91622f,  // 6
    -2.03158f,  // 7
    -3.93891f,  // 8
    -4.29451f   // 9
};

class LeNetTest {
private:
    LeNet net;
    float preprocessed_input[28*28];
    float output[10];

public:
    bool test_preprocess() {
        // 测试预处理
        LeNet::preprocess(test_image, preprocessed_input);
        
        // 验证预处理结果
        for (int i = 0; i < 28*28; i++) {
            float expected = test_image[i] / 255.0f;
            if (std::abs(preprocessed_input[i] - expected) > 1e-6) {
                std::cerr << "Preprocess failed at index " << i 
                         << ": expected " << expected 
                         << ", got " << preprocessed_input[i] << std::endl;
                return false;
            }
        }
        
        return true;
    }

    bool test_inference() {
        // 执行推理
        if (!net.forward(preprocessed_input, output)) {
            std::cerr << "Forward pass failed" << std::endl;
            return false;
        }
        
        // 验证输出
        return TestUtils::compare_arrays(output, expected_output, 10, 1e-4, "LeNet output");
    }

    bool test_postprocess() {
        // 测试后处理
        int prediction = LeNet::postprocess(output);
        if (prediction != 1) {  // 应该预测为数字1
            std::cerr << "Postprocess failed: expected 1, got " << prediction << std::endl;
            return false;
        }
        return true;
    }

    bool run_all_tests() {
        if (!load_weights()) {
            std::cerr << "Failed to load weights" << std::endl;
            return false;
        }

        std::cout << "Testing LeNet..." << std::endl;

        bool preprocess_ok = test_preprocess();
        std::cout << "Preprocess test: " << (preprocess_ok ? "PASSED" : "FAILED") << std::endl;

        bool inference_ok = test_inference();
        std::cout << "Inference test: " << (inference_ok ? "PASSED" : "FAILED") << std::endl;

        bool postprocess_ok = test_postprocess();
        std::cout << "Postprocess test: " << (postprocess_ok ? "PASSED" : "FAILED") << std::endl;

        return preprocess_ok && inference_ok && postprocess_ok;
    }
};

int main() {
    LeNetTest test;
    return test.run_all_tests() ? 0 : -1;
} 