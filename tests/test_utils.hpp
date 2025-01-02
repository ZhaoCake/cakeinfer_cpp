#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <string>
#include <vector>

class TestUtils {
public:
    /**
     * 比较两个浮点数组是否相等
     * @param actual 实际输出
     * @param expected 期望输出
     * @param size 数组大小
     * @param epsilon 允许的误差
     * @param name 测试名称
     * @return 是否相等
     */
    static bool compare_arrays(const float* actual, const float* expected, 
                             size_t size, float epsilon, const std::string& name);

    /**
     * 从文件加载测试数据
     * @param filename 文件名
     * @param size 数据大小
     * @return 加载的数据
     */
    static std::vector<float> load_test_data(const std::string& filename, size_t size);
};

#endif // TEST_UTILS_HPP 