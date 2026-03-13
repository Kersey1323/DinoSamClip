import os
import xml.etree.ElementTree as ET
from PIL import Image
import random
from tqdm import tqdm

# ==========================================
# 1. 路径配置（请修改为你的实际路径）
# ==========================================
# 你的 VOC 数据集根目录（包含 Annotations 和 JPEGImages 文件夹）

VOC_ROOT = "/media/dell/新加卷1/CV/DinoSamClip/dinov2_finetuning/data/voc/pure_val" 
# 裁切后的数据集输出目录（用于给 ImageFolder 加载）
OUTPUT_DIR = f"/media/dell/新加卷1/CV/DinoSamClip/dinov2_finetuning/data/voc/val"

# VOC 的 20 个类 + 1 个我们人工制造的 background 类
CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 
    'background'
]

def check_overlap(box1, box2):
    """检查两个框是否重叠 (用于提取纯净的背景)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    if x1_max <= x2_min or x2_max <= x1_min: return False
    if y1_max <= y2_min or y2_max <= y1_min: return False
    return True

def main():
    # 创建 21 个类别的文件夹
    for c in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, c), exist_ok=True)

    annotations_dir = os.path.join(VOC_ROOT, "Annotations")
    jpeg_dir = os.path.join(VOC_ROOT, "JPEGImages")
    
    if not os.path.exists(annotations_dir):
        print(f"找不到文件夹: {annotations_dir}")
        return

    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    print(f"发现 {len(xml_files)} 个 XML 标注文件，开始精准裁切...")

    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        
        # 获取文件名并拼接图片路径
        img_filename = root.find('filename').text
        img_path = os.path.join(jpeg_dir, img_filename)
        
        if not os.path.exists(img_path):
            continue
            
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法读取图片 {img_path}: {e}")
            continue
            
        width, height = img.size
        obj_boxes = []
        
        # ==========================================
        # 2. 提取并裁切目标物体 (Object)
        # ==========================================
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in CLASSES: 
                continue # 忽略未知类别
            
            # 我们只取 Object 级别的 bbox，忽略 part 级别的 bbox
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # 防止框越界
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(width, xmax), min(height, ymax)
            
            if xmax <= xmin or ymax <= ymin: continue
                
            obj_boxes.append((xmin, ymin, xmax, ymax))
            
            # 裁切目标并保存到对应的类别文件夹
            crop_img = img.crop((xmin, ymin, xmax, ymax))
            save_name = f"{xml_file.replace('.xml', '')}_{xmin}_{ymin}.jpg"
            save_path = os.path.join(OUTPUT_DIR, name, save_name)
            crop_img.save(save_path)
            
        # ==========================================
        # 3. 提取纯净背景 (Background)
        # ==========================================
        # 随机尝试生成 3 个背景框
        bg_attempts = 3
        bg_size = 150 # 背景裁切块的基准大小
        
        for _ in range(bg_attempts):
            if width <= bg_size or height <= bg_size: break
                
            rx = random.randint(0, width - bg_size)
            ry = random.randint(0, height - bg_size)
            bg_box = (rx, ry, rx + bg_size, ry + bg_size)
            
            # 严格检查：如果这个随机框跟图里任何一个物体有重叠，就丢弃！
            is_valid_bg = True
            for obj_box in obj_boxes:
                if check_overlap(bg_box, obj_box):
                    is_valid_bg = False
                    break
                    
            # 如果完全没有重叠，说明是极其纯净的背景（墙壁、马路、天空等）
            if is_valid_bg:
                bg_crop = img.crop(bg_box)
                bg_save_name = f"{xml_file.replace('.xml', '')}_bg_{rx}_{ry}.jpg"
                bg_save_path = os.path.join(OUTPUT_DIR, "background", bg_save_name)
                bg_crop.save(bg_save_path)

    print("\n✅ VOC 数据裁切完成！")
    print(f"请检查目录: {OUTPUT_DIR}，里面应该有 21 个文件夹。")
    print("现在你可以把这个目录作为 ROOT_PATH，塞进我们上一版写的 Contrastive Tuning 代码里微调 DinoV2 啦！")

if __name__ == "__main__":
    main()