# NOTE: Historical cross-source script evaluating a ProGAN-trained checkpoint.
# It does NOT reproduce the paper's SDv1.4-trained GenImage protocol.
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, accuracy_score
from tqdm import tqdm
from PIL import Image
import random
import numpy as np

# ================= ⚙️ 配置区域 =================

# 1. 你的最强模型 Epoch
EPOCH = 8
DEVICE = "cuda"

# 2. 之前校准的黄金阈值 (ProGAN Source Domain)
FIXED_THRESHOLD = 0.0010

# 3. 本地 CLIP 权重路径 (你刚才 ls 确认过的路径)
CLIP_LOCAL_PATH = "/root/autodl-tmp/pretrained_models/clip-vit-large-patch14"

# 4. GenImage 根目录
GENIMAGE_ROOT = "/root/autodl-tmp/GenImage"

# 5. 数据集路径映射
DATASET_PATHS = {
    "Midjourney":       ("Midjourney",             "imagenet_midjourney"),
    "Stable Diffusion v1.4": ("stable_diffusion_v_1_4", "imagenet_ai_0419_sdv4"),
    "Stable Diffusion v1.5": ("stable_diffusion_v_1_5", "imagenet_ai_0424_sdv5"),
    "ADM":              ("ADM",                    "imagenet_ai_0508_adm"),
    "Glide":            ("glide",                  "imagenet_glide"),
    "Wukong":           ("wukong",                 "imagenet_ai_0424_wukong"),
    "VQDM":             ("VQDM",                   "imagenet_ai_0419_vqdm"),
}

# 6. 测试样本数 (2000张足够权威且速度快)
MAX_SAMPLES = 2000

# 7. 你的权重保存文件夹 (根据之前 benchmark_final.py 的设置)
CHECKPOINT_DIR = "./checkpoints/effort_universal_repro"

# ===============================================

def load_model(epoch, device):
    """
    加载模型，强制使用本地 CLIP 权重，无需联网
    """
    from models.clip_models import ClipModel
    try:
        print(f"⚡️ Loading CLIP from LOCAL path: {CLIP_LOCAL_PATH}")
        
        # 初始化模型 (指定本地路径)
        model = ClipModel(
            name=CLIP_LOCAL_PATH,  # <--- 核心修改：使用本地路径
            num_classes=1, 
            fix_backbone=True, 
            use_svd=True, 
            svd_rank_ratio=0.25
        )
        
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint not found: {ckpt_path}")
            return None
            
        print(f"⚡️ Loading Epoch {epoch} weights from {ckpt_path}...")
        
        # 加载训练好的权重
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # 兼容权重字典处理
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 去除 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

class GenImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.transform = transform
        self.samples = []
        
        # 定义路径: nature=真(0), ai=假(1)
        real_dir = os.path.join(root_dir, 'nature')
        fake_dir = os.path.join(root_dir, 'ai')
        
        # 1. 加载真图 (nature)
        real_imgs = []
        if os.path.exists(real_dir):
            for root, _, files in os.walk(real_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        real_imgs.append((os.path.join(root, file), 0))
        
        # 2. 加载假图 (ai)
        fake_imgs = []
        if os.path.exists(fake_dir):
            for root, _, files in os.walk(fake_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        fake_imgs.append((os.path.join(root, file), 1))
        
        # 3. 平衡与采样
        random.shuffle(real_imgs)
        random.shuffle(fake_imgs)
        
        if max_samples:
            real_imgs = real_imgs[:max_samples]
            fake_imgs = fake_imgs[:max_samples]
            
        # 强制平衡 (取最小值)
        min_len = min(len(real_imgs), len(fake_imgs))
        self.samples = real_imgs[:min_len] + fake_imgs[:min_len]
        
        print(f"    Found: {len(real_imgs)} Real (nature), {len(fake_imgs)} Fake (ai)")
        print(f"    Loaded: {min_len} Real + {min_len} Fake = {len(self.samples)} Total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            return torch.zeros(3, 224, 224), label

def main():
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    model = load_model(EPOCH, DEVICE)
    if model is None:
        return

    print(f"\n{'='*60}")
    print(f"🚀 GenImage Benchmark (Cross-Domain Generalization)")
    print(f"🎯 Threshold: {FIXED_THRESHOLD} (Calibrated on ProGAN)")
    print(f"{'='*60}\n")

    results = []

    for name, (folder, subfolder) in DATASET_PATHS.items():
        # 拼接完整路径：/root/.../GenImage/Midjourney/imagenet_midjourney/val
        val_path = os.path.join(GENIMAGE_ROOT, folder, subfolder, "val")
        
        print(f"📂 Testing: {name}")
        # print(f"   Path: {val_path}")
        
        if not os.path.exists(val_path):
            print(f"❌ Path not found! Skipping... ({val_path})")
            continue
            
        dataset = GenImageDataset(val_path, transform=transform, max_samples=MAX_SAMPLES)
        
        if len(dataset) == 0:
            print("⚠️  Dataset empty. Skipping.")
            continue
            
        loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
        
        y_true, y_scores = [], []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader, leave=False, desc=f"Evaluating {name}"):
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1]
                
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())
        
        # 计算指标
        ap = average_precision_score(y_true, y_scores) * 100
        y_pred = [1 if s >= FIXED_THRESHOLD else 0 for s in y_scores]
        acc = accuracy_score(y_true, y_pred) * 100
        
        print(f"   👉 Result: AP={ap:.2f}% | Acc={acc:.2f}%")
        results.append({"Dataset": name, "AP": ap, "Acc": acc})

    print("\n" + "="*60)
    print("🏆 FINAL GENIMAGE RESULTS (ProGAN Model)")
    print("="*60)
    print(f"{'Dataset':<25} | {'AP (%)':<10} | {'Acc (%)':<10}")
    print("-" * 55)
    for res in results:
        print(f"{res['Dataset']:<25} | {res['AP']:<10.2f} | {res['Acc']:<10.2f}")
    print("-" * 55)
    print("(Note: Zero-shot results using ProGAN-calibrated threshold)")

if __name__ == "__main__":
    main()