import torch
from model import LeNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def infer(model_path='./resources/lenet.pth', batch_size=1):
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载测试数据集
    test_dataset = datasets.MNIST('./resources/mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 用于统计准确率
    correct = 0
    total = 0
    
    # 随机选择一些样本进行可视化
    examples = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 获取预测结果
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 收集一些例子用于可视化
            if len(examples) < 6 and batch_size == 1:
                examples.append({
                    'image': data[0].cpu().numpy(),
                    'truth': target.item(),
                    'pred': predicted.item()
                })
    
    # 打印整体准确率
    accuracy = 100 * correct / total
    print(f'准确率: {accuracy:.2f}%')
    
    # 可视化一些预测结果
    if examples:
        fig = plt.figure(figsize=(10, 5))
        for i, example in enumerate(examples):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example['image'][0], cmap='gray', interpolation='none')
            plt.title(f'Pred: {example["pred"]}\nTrue: {example["truth"]}')
            plt.xticks([])
            plt.yticks([])
        plt.savefig('./resources/predictions.png')
        print(f'预测结果可视化已保存到 ./resources/predictions.png')

if __name__ == '__main__':
    infer()
