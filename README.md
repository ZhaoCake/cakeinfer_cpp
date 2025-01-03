# cakeinfer_cpp

这是一个使用C++实现的LeNet-5神经网络推理项目（未完成）。该项目旨在将PyTorch训练的LeNet模型转换为纯C++实现，然后借助这个简单的模型来学习HLS的使用（还没开始，乐）。

---
对于算子的认识太不到位了，竟然写不出能用的方法，毫无疑问我应该从numpy开始搭一个。于是这个项目的维护就暂停吧，等我弄清楚了再继续。打基础太重要了。一直以来调包已经把我废得差不多了。



## 项目结构

```
.
├── include/            # 头文件
│   ├── lenet.hpp      # LeNet网络定义
│   ├── operations.hpp # 网络操作算子定义
│   └── weights.hpp    # 权重声明
├── src/               # 源文件
│   ├── lenet.cpp     # LeNet网络实现
│   ├── weights.cpp   # 权重加载实现
│   └── operations/   # 算子实现
│       ├── conv2d.cpp    # 卷积层
│       ├── maxpool2d.cpp # 最大池化层
│       ├── fc.cpp        # 全连接层
│       ├── linear.cpp    # 线性层
│       └── relu.cpp      # ReLU激活函数
├── tests/            # 测试文件
│   ├── test_conv2d.cpp  # 卷积测试
│   └── test_lenet.cpp   # 网络测试
└── python/           # Python工具
    ├── train.py     # 训练脚本
    ├── convert.py   # 模型转换脚本
    └── test_utils.py # 测试数据生成
```

## 已完成功能

- [x] 基础网络架构
- [x] 基础算子实现
  - [x] 卷积层 (Conv2d)
  - [x] 最大池化 (MaxPool2d)
  - [x] 全连接层 (FC)
  - [x] ReLU激活函数
- [x] 权重加载机制
- [x] 基础测试框架

## 待完成功能

- [ ] 性能优化
  - [ ] HLS编译
  - [ ] 内存优化
- [ ] 完整测试覆盖
- [ ] 模型量化
- [ ] FPGA推理性能分析

## 注意事项

1. 这是一个正在开发中的项目，功能可能会发生变化
2. 当前实现主要关注功能正确性，性能优化尚未完成
3. 主要目的是撸一遍网络，然后用HLS的宏优化和编译为HDL用于FPGA
4. 欢迎贡献代码和提出建议

## 许可证

[Apache-2.0](./LICENSE)

## 贡献

欢迎提交Issue和Pull Request！