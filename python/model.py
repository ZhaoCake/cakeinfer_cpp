import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一个卷积层块, 输入1x28x28, 输出6x14x14, 卷积核5x5, 步长2, 填充2
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  
        
        # 第二个卷积层块， 输入6x14x14, 输出16x5x5, 卷积核5x5, 步长2, 填充2
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层, 输入16x5x5, 输出120, 输入120, 输出84, 输入84, 输出10
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # MNIST图像经过卷积池化后大小为5x5
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)  # 输出10个类别
        )

    def forward(self, x):
        x = self.conv1(x)  # 输入1x28x28 -> 6x14x14
        x = self.conv2(x)  # 6x14x14 -> 16x5x5
        x = x.view(x.size(0), -1)  # 展平，16x5x5=400
        x = self.fc(x)
        return x
