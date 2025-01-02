import torch
import numpy as np
from model import LeNet
import os

def generate_test_data():
    """生成测试数据和每层的输出"""
    # 创建测试目录
    os.makedirs('./resources/test', exist_ok=True)
    
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load('./resources/lenet.pth'))
    model.eval()
    
    # 生成测试输入 (NCHW布局)
    test_input = torch.randn(1, 1, 28, 28)
    test_input.numpy().tofile('./resources/test/input.bin')
    
    # 保存每层的输出
    with torch.no_grad():
        # conv1 (保持NCHW布局)
        x = model.conv1[0](test_input)
        x.numpy().tofile('./resources/test/conv1_output.bin')
        
        # 保存形状信息
        with open('./resources/test/shapes.txt', 'w') as f:
            f.write(f"Input: {test_input.shape}, {test_input.stride()}\n")
            f.write(f"Conv1 output: {x.shape}, {x.stride()}\n")
            f.write(f"Conv1 weight: {model.conv1[0].weight.shape}, {model.conv1[0].weight.stride()}\n")
        
        print("测试数据生成完成！")

if __name__ == '__main__':
    generate_test_data() 