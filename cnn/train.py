# モジュールのインポート
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# dataset.py内のdatasets関数をインポート
from dataset import cifar_dataset
# model.py内のCNNクラスをインポート
from model import CNN

# 保存先のパス
model_path = 'cifar_cnn.pth'

# ローカルのGPUを使用するためにデバイスを設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データローダーからデータを受け取る
train_data, _ = cifar_dataset()
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# モデル、損失関数、最適化関数の定義
model = CNN().to(device)  # モデルをデバイスに転送
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
        
if __name__ == "__main__":
    # 学習回数
    epochs = 30 # 20→30

    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        
        #train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # データをデバイスに転送
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader.dataset)
        
        # モデルの保存
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss
        }, model_path)
        
        print ('Epoch: {}, Loss: {loss:.4f}'.format(epoch+1, i+1, loss=avg_train_loss))