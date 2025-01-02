#include "operations.hpp"
#include "test_utils.hpp"
#include "weights.hpp"
#include <iostream>
#include <vector>

#define INPUT_H 28
#define INPUT_W 28
#define INPUT_C 1
#define OUTPUT_C 6
#define KERNEL_SIZE 5

class Conv2DTest {
private:
    std::vector<float> input;
    std::vector<float> output;
    std::vector<float> expected;
    
public:
    bool run_test() {
        try {
            // 加载测试数据
            input = TestUtils::load_test_data(
                "./resources/test/input.bin", 
                INPUT_H * INPUT_W * INPUT_C
            );
            
            expected = TestUtils::load_test_data(
                "./resources/test/conv1_output.bin",
                OUTPUT_C * (INPUT_H-KERNEL_SIZE+1) * (INPUT_W-KERNEL_SIZE+1)
            );
            
            // 准备输出缓冲区
            output.resize(OUTPUT_C * (INPUT_H-KERNEL_SIZE+1) * (INPUT_W-KERNEL_SIZE+1));
            
            // 执行卷积
            conv2d(input.data(), output.data(), 
                  reinterpret_cast<float*>(weights_conv1), bias_conv1,
                  1, INPUT_H, INPUT_W, INPUT_C, 
                  OUTPUT_C, KERNEL_SIZE, 1, 0);
            
            // 比较结果
            return TestUtils::compare_arrays(
                output.data(), expected.data(),
                OUTPUT_C * (INPUT_H-KERNEL_SIZE+1) * (INPUT_W-KERNEL_SIZE+1),
                1e-5, "Conv2D Layer 1"
            );
            
        } catch (const std::exception& e) {
            std::cerr << "Test failed: " << e.what() << std::endl;
            return false;
        }
    }
};

int main() {
    try {
        if (!load_weights()) {
            std::cerr << "Failed to load weights" << std::endl;
            return -1;
        }
        
        Conv2DTest test;
        return test.run_test() ? 0 : -1;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
} 