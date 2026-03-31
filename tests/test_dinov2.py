import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import numpy as np

def test_dinov2_attention():
    # 1. 你的本地模型路径 (替换为你的实际路径)
    model_name = "/media/dell/新加卷1/LLM/models/CV-models/dinov2-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[{device}] 正在加载 DinoV2: {model_name}")
    
    # 尝试两种加载方式：有些版本必须在初始化时强行指定 output_attentions=True
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
    model.eval()

    # 2. 准备一张测试图片 (非正方形，用来测试长宽比 BUG 是否彻底解决)
    print("\n准备测试图像...")
    url = 'https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg'
    image = Image.open(requests.get(url, stream=True).raw)
    print(f"原始图像尺寸: {image.size} (宽x高)")

    inputs = processor(images=image, return_tensors="pt").to(device)
    
    actual_h = inputs['pixel_values'].shape[2]
    actual_w = inputs['pixel_values'].shape[3]
    print(f"预处理后 Tensor 尺寸: {inputs['pixel_values'].shape}")
    print(f"预期 Patch 数量: 高 {actual_h//14}, 宽 {actual_w//14}, 总计 {(actual_h//14)*(actual_w//14)}")

    # 3. 模型推理
    print("\n开始模型推理...")
    with torch.no_grad():
        # 这里同时也传入 output_attentions=True 双重保险
        outputs = model(**inputs, output_attentions=True)

    # 4. 暴力解剖 Outputs 结构
    print("\n" + "="*40)
    print("模型输出 (outputs) 结构分析:")
    
    # 检查有哪些 key
    if hasattr(outputs, "keys"):
        print(f"包含的 Keys: {list(outputs.keys())}")
    
    # 检查 attentions
    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        print("❌ 错误: outputs 中完全没有 attentions 属性！")
    else:
        print(f"attentions 类型: {type(outputs.attentions)}")
        if isinstance(outputs.attentions, tuple):
            print(f"attentions 元组长度: {len(outputs.attentions)}")
            if len(outputs.attentions) > 0:
                print("✅ 成功提取到注意力矩阵！")
                print(f"最后一层 attention shape: {outputs.attentions[-1].shape}")
                
                # 测试提取逻辑
                patch_num = (actual_h//14) * (actual_w//14)
                cls_attention = outputs.attentions[-1][0, :, 0, 1:1+patch_num].mean(0)
                print(f"CLS Attention shape (切片后): {cls_attention.shape}")
            else:
                print("❌ 错误: attentions 是一个空元组 ()！")
    print("="*40)

if __name__ == "__main__":
    test_dinov2_attention()