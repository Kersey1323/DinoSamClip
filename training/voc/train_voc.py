import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F
from pytorch_metric_learning import losses

# ==========================================
# 1. 基础配置
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOCAL_MODEL_PATH = '/media/dell/新加卷1/LLM/models/CV-models/dinov2-large'
ROOT_PATH = '/media/dell/新加卷1/CV/DinoSamClip/dinov2_finetuning/data/voc' 

batch_size = 32 # ⚠️ 如果显存爆炸 (OOM)，请毫不犹豫地改成 16 或 8
num_workers = 4
num_epoch = 5   # 微调骨干网络需要多跑几轮
knn_k = 5       # KNN 验证寻找的最邻近样本数

# ==========================================
# 2. 数据准备
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

# ⚠️ 注意：计算 KNN 时，我们需要一个 "不打乱顺序(shuffle=False)" 且 "不使用数据增强" 的验证集加载器
# 因为我们需要提取稳定、确定的验证集特征。
val_loader_for_knn = DataLoader(image_datasets['val'], shuffle=False, batch_size=batch_size, num_workers=num_workers)
# 同样，提取训练集特征做 Gallery 时也不需要随机打乱
train_loader_for_knn = DataLoader(image_datasets['train'], shuffle=False, batch_size=batch_size, num_workers=num_workers)

# 这个是专门用来训练的 DataLoader (打乱顺序，应用增强)
train_loader = DataLoader(image_datasets['train'], shuffle=True, batch_size=batch_size, num_workers=num_workers)

class_names = image_datasets['train'].classes
print(f"检测到的类别 (共 {len(class_names)} 类): {class_names}")

# ==========================================
# 3. 创建纯特征提取模型
# ==========================================
class HFDinoFeatureExtractor(nn.Module):
    def __init__(self, model_path, unfreeze_blocks=2):
        super(HFDinoFeatureExtractor, self).__init__()
        print(f"Loading backbone from {model_path}...")
        self.backbone = AutoModel.from_pretrained(model_path)
        
        # 1. 先冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. 只解冻最后几层 Transformer Block
        print(f"Unfreezing the last {unfreeze_blocks} transformer blocks...")
        num_layers = len(self.backbone.encoder.layer)
        for i in range(num_layers - unfreeze_blocks, num_layers):
            for param in self.backbone.encoder.layer[i].parameters():
                param.requires_grad = True
                
        # 解冻 LayerNorm 层，有助于训练稳定
        for param in self.backbone.layernorm.parameters():
            param.requires_grad = True

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_feature = outputs.last_hidden_state[:, 0, :] 
        # 对比学习必须 L2 归一化
        cls_feature_norm = F.normalize(cls_feature, p=2, dim=1)
        return cls_feature_norm

model = HFDinoFeatureExtractor(LOCAL_MODEL_PATH, unfreeze_blocks=2).to(device)

# ==========================================
# 4. KNN 验证函数定义 (评估特征空间质量)
# ==========================================
def validate_knn(model, train_loader, val_loader, device, k=5):
    """使用 KNN 评估模型提取特征的区辨度"""
    model.eval() # 极其重要：必须切换到验证模式
    print("\n[开始 KNN 验证特征质量... (这可能需要一两分钟)]")
    
    # 1. 提取训练集的所有特征作为“知识库” (Gallery)
    train_features = []
    train_labels = []
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="提取 Train 特征做图库", leave=False):
            features = model(images.to(device))
            train_features.append(features.cpu())
            train_labels.append(labels)
            
    train_features = torch.cat(train_features, dim=0) # [N_train, 1024]
    train_labels = torch.cat(train_labels, dim=0)     # [N_train]
    
    # 2. 提取验证集特征作为“查询目标” (Query)
    val_features = []
    val_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="提取 Val 特征", leave=False):
            features = model(images.to(device))
            val_features.append(features.cpu())
            val_labels.append(labels)
            
    val_features = torch.cat(val_features, dim=0) # [N_val, 1024]
    val_labels = torch.cat(val_labels, dim=0)     # [N_val]

    # 3. 计算余弦相似度矩阵: [N_val, 1024] x [1024, N_train] -> [N_val, N_train]
    # 因为特征在 forward 时已经做过 L2 归一化，所以直接内积就是余弦相似度
    similarity_matrix = torch.mm(val_features, train_features.t())
    
    # 4. 找出最相似的 K 个邻居
    _, topk_indices = similarity_matrix.topk(k, dim=1, largest=True, sorted=True)
    topk_labels = train_labels[topk_indices]
    
    # 5. 多数表决算准确率
    correct = 0
    total = val_features.size(0)
    
    for i in range(total):
        query_label = val_labels[i].item()
        neighbor_labels = topk_labels[i].tolist()
        predicted_label = max(set(neighbor_labels), key=neighbor_labels.count)
        
        if predicted_label == query_label:
            correct += 1
            
    knn_accuracy = 100.0 * correct / total
    print(f"✅ 当前 Epoch 特征空间 KNN (k={k}) 准确率: {knn_accuracy:.2f}%")
    return knn_accuracy

# ==========================================
# 5. 损失函数与优化器
# ==========================================
criterion = losses.SupConLoss(temperature=0.1)
# 学习率极小：1e-5，防止特征崩溃
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)

# ==========================================
# 6. 训练与验证循环
# ==========================================
print("\n🚀 开始对比学习微调 (Contrastive Fine-Tuning)...")

# 在训练前，先跑一次未经微调的初始权重 KNN 精度，作为 Baseline
baseline_acc = validate_knn(model, train_loader_for_knn, val_loader_for_knn, device, k=knn_k)
best_knn_acc = baseline_acc
SAVE_PATH = "./dinov2_contrastive_finetuned_best.pth"

for epoch in range(num_epoch):
    model.train() # 切回训练模式
    loop = tqdm(train_loader)
    running_loss = 0.0
    
    for idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        features = model(images)
        loss = criterion(features, labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        loop.set_description(f"Epoch [{epoch+1}/{num_epoch}]")
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(train_loader)
    print(f"\nEpoch [{epoch+1}/{num_epoch}] Average Train SupConLoss: {avg_loss:.4f}")
    
    # 每个 epoch 结束，进行一次 KNN 特征质量评估
    current_knn_acc = validate_knn(model, train_loader_for_knn, val_loader_for_knn, device, k=knn_k)
    
    # 保存 KNN 精度最高的模型权重
    if current_knn_acc >= best_knn_acc:
        best_knn_acc = current_knn_acc
        torch.save(model.backbone.state_dict(), SAVE_PATH)
        print(f"🌟 发现更好的特征空间！已保存新权重至 {SAVE_PATH} (当前最高 KNN Acc: {best_knn_acc:.2f}%)")
    else:
        print(f"💡 本轮未超越最高记录 (最高仍为 {best_knn_acc:.2f}%)")

print(f"\n🎉 骨干网络微调全部完成！最高 KNN 准确率: {best_knn_acc:.2f}%")
print(f"最佳模型权重已保存至: {SAVE_PATH}")
print("你可以用这个权重直接替换 Pipeline 里的 DINOv2 了！")