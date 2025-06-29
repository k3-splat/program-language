import torch.nn as nn

class CNN(nn.Module):
    # 畳み込み層 2層→5層 / 線形層 3層→4層 / バッチ正規化層 0層→8層 / ドロップアウト層 0層→3層
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.l1 = nn.Linear(1 * 1 * 256, 1024)
        self.bn6n7 = nn.BatchNorm1d(1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.l4 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = self.pool(self.act(self.bn4(self.conv4(x))))
        x = self.pool(self.act(self.bn5(self.conv5(x))))
        x = x.view(x.size()[0], -1)
        x = self.dropout(self.act(self.bn6n7(self.l1(x))))
        x = self.dropout(self.act(self.bn6n7(self.l2(x))))
        x = self.dropout(self.act(self.bn8(self.l3(x))))
        x = self.l4(x)
        return x

if __name__ == "__main__":
    model = CNN()
    print(model)