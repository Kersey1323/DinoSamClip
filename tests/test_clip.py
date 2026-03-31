import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# 1. 配置你的本地模型路径 (确保路径下有 config.json)
local_model_path = "/media/dell/新加卷1/LLM/models/CV-models/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("1. 正在从本地加载 CLIP 模型...")
model = CLIPModel.from_pretrained(local_model_path).to(device)
processor = CLIPProcessor.from_pretrained(local_model_path)
model.eval()
print(f"模型加载成功！运行在 {device} 上。")

# 2. 准备测试数据
print("\n2. 准备图像和文本...")
url = 'https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg'
image = Image.open(requests.get(url, stream=True).raw)

# 注意：标准 CLIP 必须用英文标签！
candidate_classes = ["squirtle", "bulbasaur", "charmander", "pikachu"]

# 3. 推理计算
print("\n3. 开始计算图文匹配概率...")
inputs = processor(text=candidate_classes, images=image, return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    # logits_per_image 就是相似度得分 (已包含模型内部的温度系数)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 4. 打印结果
print("\n" + "="*30)
print("图文匹配概率:")
for i, name in enumerate(candidate_classes):
    print(f"{name}: {probs[0][i]:.2%}")
print("="*30)