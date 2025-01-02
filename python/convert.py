import torch
import numpy as np
import json
from model import LeNet
import struct

def save_conv_layer(state_dict, layer_name, output_file):
    """保存卷积层参数"""
    weight = state_dict[f'{layer_name}.0.weight'].cpu().numpy()
    bias = state_dict[f'{layer_name}.0.bias'].cpu().numpy()
    
    # 保存权重
    weight.tofile(f'./resources/weights/{layer_name}_weight.bin')
    # 保存偏置
    bias.tofile(f'./resources/weights/{layer_name}_bias.bin')
    
    # 返回该层的配置信息
    return {
        'type': 'conv2d',
        'name': layer_name,
        'in_channels': weight.shape[1],
        'out_channels': weight.shape[0],
        'kernel_size': weight.shape[2],
        'weight_shape': list(weight.shape),
        'bias_shape': list(bias.shape),
        'weight_file': f'{layer_name}_weight.bin',
        'bias_file': f'{layer_name}_bias.bin'
    }

def save_fc_layer(state_dict, layer_name, fc_index, output_file):
    """保存全连接层参数"""
    weight = state_dict[f'{layer_name}.{fc_index}.weight'].cpu().numpy()
    bias = state_dict[f'{layer_name}.{fc_index}.bias'].cpu().numpy()
    
    # 保存权重和偏置
    weight.tofile(f'./resources/weights/{layer_name}_{fc_index}_weight.bin')
    bias.tofile(f'./resources/weights/{layer_name}_{fc_index}_bias.bin')
    
    return {
        'type': 'linear',
        'name': f'{layer_name}_{fc_index}',
        'in_features': weight.shape[1],
        'out_features': weight.shape[0],
        'weight_shape': list(weight.shape),
        'bias_shape': list(bias.shape),
        'weight_file': f'{layer_name}_{fc_index}_weight.bin',
        'bias_file': f'{layer_name}_{fc_index}_bias.bin'
    }

def convert_model(model_path='./resources/lenet.pth'):
    """转换模型为C语言可读格式"""
    import os
    # 创建保存权重的目录
    os.makedirs('./resources/weights', exist_ok=True)
    
    # 加载模型状态
    state_dict = torch.load(model_path)
    
    # 保存模型结构信息
    model_config = {
        'name': 'lenet',
        'input_shape': [1, 28, 28],  # MNIST输入形状
        'layers': []
    }
    
    # 保存第一个卷积层
    conv1_config = save_conv_layer(state_dict, 'conv1', './resources/weights')
    model_config['layers'].append(conv1_config)
    
    # 保存第二个卷积层
    conv2_config = save_conv_layer(state_dict, 'conv2', './resources/weights')
    model_config['layers'].append(conv2_config)
    
    # 保存全连接层
    fc_indices = [0, 2, 4]  # fc层在Sequential中的索引（跳过ReLU层）
    for idx in fc_indices:
        fc_config = save_fc_layer(state_dict, 'fc', idx, './resources/weights')
        model_config['layers'].append(fc_config)
    
    # 保存模型配置到JSON文件
    with open('./resources/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print("模型转换完成！")
    print(f"配置文件保存在: ./resources/model_config.json")
    print(f"权重文件保存在: ./resources/weights/")

if __name__ == '__main__':
    convert_model()
