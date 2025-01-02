#include "test_utils.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>

bool TestUtils::compare_arrays(const float* actual, const float* expected, 
                             size_t size, float epsilon, const std::string& name) {
    bool is_equal = true;
    int max_errors = 10;  // 最多显示10个错误
    int error_count = 0;
    
    for (size_t i = 0; i < size; i++) {
        float diff = std::abs(actual[i] - expected[i]);
        if (diff > epsilon) {
            if (error_count < max_errors) {
                std::cout << name << ": Mismatch at index " << i 
                         << ": actual=" << actual[i] 
                         << ", expected=" << expected[i] << std::endl;
            }
            error_count++;
            is_equal = false;
        }
    }
    
    if (is_equal) {
        std::cout << name << ": PASSED" << std::endl;
    } else {
        std::cout << name << ": FAILED (total " << error_count 
                 << " mismatches)" << std::endl;
    }
    
    return is_equal;
}

std::vector<float> TestUtils::load_test_data(const std::string& filename, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open test file: " + filename);
    }
    
    std::vector<float> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size * sizeof(float));
    
    if (file.gcount() != size * sizeof(float)) {
        throw std::runtime_error(
            "Failed to read test data: expected " + std::to_string(size) + 
            " items, got " + std::to_string(file.gcount() / sizeof(float))
        );
    }
    
    return buffer;
} 