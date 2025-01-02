import torch
from model import LeNet
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train(epochs=10, batch_size=32, learning_rate=0.001):
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载MNIST数据集
    train_dataset = datasets.MNIST('./resources/mnist', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = LeNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 对于分类任务使用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(data)
            
            # 计算损失
            loss = criterion(outputs, target)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            running_loss += loss.item()
            
            # 每100个batch打印一次训练信息
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # 打印每个epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss:.4f}')
    
    print('训练完成!')
    
    # 保存模型
    torch.save(model.state_dict(), './resources/lenet.pth')

if __name__ == '__main__':
    train()
