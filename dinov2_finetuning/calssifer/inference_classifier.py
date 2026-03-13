import torch
import torch.nn as nn
from transformers import AutoModel

from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOCAL_MODEL_PATH = '/media/dell/新加卷1/LLM/models/CV-models/dinov2-large'
WEIGHT_PATH = "/media/dell/新加卷1/CV/DinoSamClip/dinov2_finetuning/data/dinov2_custom_classifier.pth"

# ==========================================
# 第一步：必须保留你自定义的模型类定义（相当于搭架子）
# ==========================================
class HFDinoVisionClassifier(nn.Module):
    def __init__(self, model_path, num_classes):
        super(HFDinoVisionClassifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_path)
        hidden_dim = self.backbone.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        cls_feature = outputs.last_hidden_state[:, 0, :] 
        logits = self.classifier(cls_feature)
        return logits

# ==========================================
# 第二步：实例化模型并加载你保存的权重（灌入灵魂）
# ==========================================
# 1. 实例化模型（需要知道一共有几个类别，Intel 数据集是 6 个）
num_classes = 6 
model = HFDinoVisionClassifier(LOCAL_MODEL_PATH, num_classes)

# 2. 加载你微调后的权重，覆盖掉刚实例化的初始权重
print(f"正在加载微调权重: {WEIGHT_PATH}")
# 使用 map_location 确保在没有 GPU 的机器上也能安全加载到 CPU
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device)) 

# 3. 放到对应设备上，并切换到评估模式（极其重要，否则 Dropout 会干扰预测）
model = model.to(device)
model.eval()

print("✅ 模型加载完毕，可以开始推理了！")

# 接下来就可以直接复用我上文给你的 `predict_single_image_ssh` 函数来预测新图片了。


class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # 请根据你的数据集实际类别修改这个列表
def predict_single_image_ssh(image_path, model, class_names, device, save_path="prediction_result.png"):
    # 1. 加载并预处理图片 (必须和验证集的预处理一致)
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 2. 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        # 计算 softmax 获取百分比概率
        probabilities = F.softmax(outputs, dim=1)[0] 
        
    # 3. 获取最高概率的类别
    max_prob, predicted_idx = torch.max(probabilities, 0)
    predicted_class = class_names[predicted_idx.item()]
    
    # ==========================================
    # 4. 终端直接打印结果 (非常适合 SSH 查看)
    # ==========================================
    print("\n" + "="*30)
    print(f"正在分析图片: {image_path}")
    print(f"🌟 最终预测结果: {predicted_class} (置信度: {max_prob.item()*100:.2f}%)")
    print("-" * 30)
    print("详细类别概率分布:")
    for i, name in enumerate(class_names):
        print(f"  - {name:<10}: {probabilities[i].item()*100:>6.2f}%")
    print("="*30 + "\n")
    
    # ==========================================
    # 5. 保存带有标签的图片到硬盘 (替代 plt.show)
    # ==========================================
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({max_prob.item()*100:.1f}%)", 
              fontsize=16, color='green', fontweight='bold')
    plt.axis('off')
    
    # 使用 savefig 替代 show
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close() # 必须 close 释放画布内存，否则多次运行会内存泄漏
    print(f"🖼️ 可视化结果已保存为: {save_path}，请在文件管理器中查看。")

# --- 测试调用示例 ---
# 请将这里的路径换成你服务器上的随便一张测试图片路径
test_image_path = "/media/dell/新加卷1/CV/DinoSamClip/src/imgs/20447.jpg" 

# 假设你的 model, class_names, device 变量还在当前的 Python 环境中
predict_single_image_ssh(test_image_path, model, class_names, device, save_path="demo_output.png")