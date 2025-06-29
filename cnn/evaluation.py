# モジュールのインポート
import torch
from torch.utils.data import DataLoader

# dataset.py内のdatasets関数をインポート
from dataset import cifar_dataset
# model.py内のCNNクラスをインポート
from model import CNN

# 保存されたモデルのパス
model_path = 'cifar_cnn.pth'

# データローダーからデータを受け取る
_, test_data = cifar_dataset()
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# モデルの定義
model = CNN()

# 保存されたモデルの読み込み
checkpoint = torch.load(model_path, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

if __name__=="__main__":
    val_acc = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_acc = val_acc / len(test_loader.dataset)
    
    print('Accuracy: {:.4f}'.format(avg_val_acc))