import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import os
import copy

# ==========================================
# 1. 配置参数 (根据你的实际情况修改)
# ==========================================
data_dir = '/data/njb/Emotion/RAF-DB'  # 数据集根目录
batch_size = 32
num_epochs = 50           # 训练轮数
learning_rate = 0.001     # 初始学习率
val_split = 0.1           # 划出 10% 的训练集作为验证集
random_seed = 42          # 固定随机种子，保证每次划分一致
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"使用设备: {device}")

# ==========================================
# 2. 数据预处理与加载 (核心修改部分)
# ==========================================

# 定义数据增强和转换
data_transforms = {
    # 训练集：需要数据增强
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(15),       # 随机旋转
        transforms.RandomHorizontalFlip(),   # 随机水平翻转
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 验证集和测试集：不需要增强，只需标准化
    'val_test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("正在准备数据集...")

# A. 加载 Test 集 (直接读取 test 文件夹)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), 
                                    transform=data_transforms['val_test'])

# B. 加载 Train 集并拆分
# 技巧：加载两次 train 文件夹
# 1. 用于训练的部分 (带增强)
full_train_aug = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                      transform=data_transforms['train'])
# 2. 用于验证的部分 (无增强)
full_train_clean = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                        transform=data_transforms['val_test'])

# 计算拆分索引
num_train = len(full_train_aug)
indices = list(range(num_train))
split = int(np.floor(val_split * num_train))

# 打乱索引
np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, val_idx = indices[split:], indices[:split]

# 创建 Subset
train_dataset = Subset(full_train_aug, train_idx)      # 90% 数据，带增强
val_dataset = Subset(full_train_clean, val_idx)        # 10% 数据，无增强

# 创建 DataLoaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    'valid': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
    'test':  DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
}

dataset_sizes = {
    'train': len(train_dataset),
    'valid': len(val_dataset),
    'test':  len(test_dataset)
}
class_names = full_train_aug.classes # 获取类别名称

print(f"数据准备完成:")
print(f"- 训练集 (Train): {dataset_sizes['train']} 张")
print(f"- 验证集 (Valid): {dataset_sizes['valid']} 张 (从Train拆分)")
print(f"- 测试集 (Test) : {dataset_sizes['test']} 张 (独立文件)")
print(f"- 类别: {class_names}")

# ==========================================
# 3. 定义训练函数
# ==========================================
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个 epoch 包含训练和验证两个阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有训练阶段才反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型 (只保存验证集准确率最高的模型)
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存当前最佳模型到文件
                torch.save(model.state_dict(), 'best_rafdb_model_1.pth')
                print(f'==> 发现新最佳模型，已保存 (Acc: {best_acc:.4f})')

    time_elapsed = time.time() - since
    print(f'\n训练完成，耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'验证集最佳准确率: {best_acc:.4f}')

    # 加载最佳权重
    model.load_state_dict(best_model_wts)
    return model

# ==========================================
# 4. 初始化模型并开始训练
# ==========================================
# 使用 ResNet18 (预训练)
model_ft = models.resnet18(pretrained=True)

# 修改最后的全连接层以匹配 7 个类别
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 7)

model_ft = model_ft.to(device)

# ==========================================
# 修改部分：定义类别权重
# ==========================================

# 这里的权重是根据 RAF-DB 训练集分布估算的
# 数量少的类别 (1和2)，权重给很高
# 数量多的类别 (3)，权重给低
# 格式对应: [Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral]

class_weights = [1.0, 10.0, 5.0, 0.5, 1.0, 2.0, 1.0]

# 转换为 Tensor 并移动到 GPU
weights_tensor = torch.FloatTensor(class_weights).to(device)

# 使用带权重的损失函数
criterion = nn.CrossEntropyLoss(weight=weights_tensor)


# 优化器
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

# 学习率衰减策略
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 开始训练
# 训练好的模型会保存在 best_rafdb_model.pth
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

# ==========================================
# 5. 最终测试 (使用 Test 集)
# ==========================================
print("\n" + "="*20)
print("正在使用 Test 集进行最终评估...")
print("="*20)

# 加载刚才训练保存的最佳权重 (确保是最佳状态)
model_ft.load_state_dict(torch.load('best_rafdb_model_1.pth'))
model_ft.eval()

correct = 0
total = 0
class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft(inputs)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print(f'\n最终测试集准确率 (Test Accuracy): {100 * correct / total:.2f}%')
print("-" * 30)
for i in range(7):
    if class_total[i] > 0:
        print(f'{class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
