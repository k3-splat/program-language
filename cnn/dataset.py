from torchvision import transforms, datasets

def cifar_dataset():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),         # ランダム水平反転
        transforms.RandomCrop(32, padding=4),     # ランダムクロップ
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 正規化
    ])
    
    # データセットの読み込み
    train_data = datasets.CIFAR10(
        root="./", 
        train=True, 
        transform=transform_train, 
        download=True
    )
    
    test_data = datasets.CIFAR10(
        root="./", 
        train=False, 
        transform=transform_test, 
        download=True
    )
    
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = cifar_dataset()
    print("Number of training samples: ", len(train_data))
    print("Number of testing samples: ", len(test_data))