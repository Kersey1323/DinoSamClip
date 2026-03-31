import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. 基础配置
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOCAL_MODEL_PATH = '/media/dell/新加卷1/LLM/models/CV-models/dinov2-large' # 你的本地 HF 模型路径
ROOT_PATH = '/media/dell/新加卷1/CV/DinoSamClip/dinov2_finetuning/data/Intel_Image_Dataset_Split' # 你的分类数据集路径

batch_size = 64
num_workers = 4
num_epoch = 3

# ==========================================
# 2. 数据准备
# 注：训练时使用 torchvision 的 transforms 做数据增强是最佳实践
# ==========================================
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    x: datasets.ImageFolder(os.path.join(ROOT_PATH, x), data_transforms[x]) 
    for x in ['train', 'val']
}

data_loaders = {
    x: DataLoader(image_datasets[x], shuffle=True, batch_size=batch_size, num_workers=num_workers)
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes
print(f"检测到的类别: {class_names}")

# ==========================================
# 3. 创建适配 Hugging Face 的分类模型
# ==========================================
class HFDinoVisionClassifier(nn.Module):
    def __init__(self, model_path, num_classes):
        super(HFDinoVisionClassifier, self).__init__()
        
        # 1. 加载本地 HF DINOv2 模型
        print(f"Loading backbone from {model_path}...")
        self.backbone = AutoModel.from_pretrained(model_path)
        
        # 2. 动态获取特征维度 (Large 模型为 1024)
        hidden_dim = self.backbone.config.hidden_size
        
        # 3. 冻结骨干网络参数 (只训练分类头) - 推荐做法！
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 4. 定义分类头 (多加了一个 Dropout 防止过拟合)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # HF 模型前向传播，接收像素值
        outputs = self.backbone(pixel_values=x)
        
        # 提取 [CLS] token 作为全局图像特征
        # outputs.last_hidden_state 形状: [batch_size, num_patches + 1, hidden_dim]
        # 第 0 个位置就是 [CLS] token
        cls_feature = outputs.last_hidden_state[:, 0, :] 
        
        # 传入分类头得出预测结果
        logits = self.classifier(cls_feature)
        return logits

# 实例化模型并移动到设备
model = HFDinoVisionClassifier(LOCAL_MODEL_PATH, len(class_names))
model = model.to(device)

# ==========================================
# 4. 损失函数与优化器
# ==========================================
criterion = nn.CrossEntropyLoss()

# 注意：因为我们冻结了 backbone，所以优化器只传入分类头(classifier)的参数。
# 这种情况下学习率可以设大一点，比如 1e-3 (参考代码全量微调用的 1e-6 太小了)
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

# ==========================================
# 5. 训练与验证循环
# ==========================================
print("开始训练...")
for epoch in range(num_epoch):
    model.train() # 设置为训练模式
    loop = tqdm(data_loaders['train'])
    
    for idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == labels).sum().item()
        accuracy = correct / labels.size(0) # 修正: 使用实际 batch 的大小

        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/{num_epoch}]")
        loop.set_postfix(loss=loss.item(), acc=accuracy)

print("训练完成，开始验证...")
model.eval() # 设置为评估模式
correct = 0
total = 0
val_predicted = []
val_labels = []

with torch.no_grad():
    for images, labels in data_loaders["val"]:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        val_labels.extend(labels.cpu().numpy().tolist())
        val_predicted.extend(predicted.cpu().numpy().tolist())

print(f'\n在验证集上的准确率 (Accuracy): {100 * correct / total:.2f} %')

# ==========================================
# 6. 打印报告与混淆矩阵
# ==========================================
print("\nClassification Report:")
print(classification_report(val_labels, val_predicted, target_names=class_names))

cm = confusion_matrix(val_labels, val_predicted)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

def show_confusion_matrix(confusion_matrix_df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=150)
    print("混淆矩阵已保存至当前目录: confusion_matrix.png")
    
show_confusion_matrix(df_cm)


# 保存整个分类模型的权重（包含分类头）
SAVE_PATH = "dinov2_custom_classifier.pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"模型已成功保存至: {SAVE_PATH}")



