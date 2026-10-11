import os
import time
import random
from tensorboardX import SummaryWriter
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from validate import validate, find_best_threshold, RealFakeDataset
from data import create_dataloader
from earlystop import EarlyStopping
from models.trainer import Trainer
from options.train_options import TrainOptions
from dataset_paths import DATASET_PATHS

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    return val_opt

# --- [新增] 自定义二分类 Dataset ---
# 强制忽略 airplane/car 等文件夹分类，只看路径里有没有 fake
class BinaryImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index] # 获取文件路径，忽略原来的 label
        
        # 强制逻辑：只要路径包含 '1_fake' 或 'fake' 就是 1，否则是 0
        # 这种逻辑适配 wang2020 数据集的通常结构
        if '1_fake' in os.path.normpath(path).lower().split(os.sep):
            target = 1
        elif '0_real' in os.path.normpath(path).lower().split(os.sep):
            target = 0
        else:
            raise ValueError(f"Validation item missing class label: {path}")
        
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
# ----------------------------------

if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    
    set_seed()
 
    model = Trainer(opt)
    
    # --- [Auto-Patch] 断点续训逻辑 ---
    if opt.continue_train:
        print(f"🔄 Resuming training from epoch {opt.epoch_count}...")
        try:
            # 如果是 epoch 2，我们加载 epoch 1 的权重
            load_epoch = opt.epoch_count - 1
            # 这里处理一下特殊的命名逻辑
            load_path = os.path.join(opt.checkpoints_dir, opt.name, f"model_epoch_{load_epoch}.pth")
            
            print(f"⚡️ Force loading weights from: {load_path}")
            
            if os.path.exists(load_path):
                checkpoint = torch.load(load_path)
                
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    if 'total_steps' in checkpoint:
                        model.total_steps = checkpoint['total_steps']
                else:
                    state_dict = checkpoint
                
                if hasattr(model.model, "module"):
                    model.model.module.load_state_dict(state_dict)
                else:
                    model.model.load_state_dict(state_dict)
                print("✅ Weights loaded successfully!")
                # =========== 【新增】手动清理显存 ===========
                import torch
                torch.cuda.empty_cache()
                print("🧹 CUDA Cache Cleared after loading weights.")
                # ==========================================
            else:
                print(f"❌ Error: Checkpoint file not found at {load_path}")
                # 如果没找到 epoch_1，尝试找 epoch_init 或者直接报错
                exit()
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            exit()
    # ----------------------------------------------------
    
    data_loader = create_dataloader(opt)

    # Held-out ProGAN validation, never unseen-generator UFD test domains.
    val_root = opt.val_root or os.path.join(opt.wang2020_data_path, 'val')
    if not os.path.isdir(val_root):
        raise FileNotFoundError(
            f"ProGAN source validation directory not found: {val_root}. "
            "Use --val_root with held-out 0_real and 1_fake subfolders.")
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    val_dataset = BinaryImageFolder(val_root, transform=val_transform)
    if len(val_dataset) == 0:
        raise ValueError(f"Empty validation dataset: {val_root}")
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.num_threads)
    model.configure_scheduler(len(data_loader))

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))

    # === Training Loop ===
    for epoch in range(opt.epoch_count, opt.niter + opt.epoch_count):
        if epoch == opt.epoch_count and not opt.continue_train:
            model.save_networks('model_epoch_init.pth')
        
        for i, data in enumerate(data_loader):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                current_lr = model.optimizer.param_groups[0]['lr']
                print(f"Train loss: {model.loss:.4f} at step: {model.total_steps}")
                
                margin_info = ""
                if hasattr(model, 'e_real') and model.e_real is not None:
                    with torch.no_grad():
                        e_real_val = model.e_real
                        e_fake_val = model.e_fake
                        labels = model.label
                        
                        real_mask = (labels == 0)
                        if real_mask.sum() > 0:
                            margin_real = (e_fake_val[real_mask] - e_real_val[real_mask]).mean().item()
                            avg_e_real = e_real_val[real_mask].mean().item()
                            msg = f"  [Real] Avg E: {avg_e_real:.2f} | Margin: {margin_real:.3f}"
                            print(msg)
                            margin_info += msg + " "

                        fake_mask = (labels == 1)
                        if fake_mask.sum() > 0:
                            margin_fake = (e_real_val[fake_mask] - e_fake_val[fake_mask]).mean().item()
                            avg_e_fake = e_fake_val[fake_mask].mean().item()
                            msg = f"  [Fake] Avg E: {avg_e_fake:.2f} | Margin: {margin_fake:.3f}"
                            print(msg)
                            margin_info += msg

                train_writer.add_scalar('loss', model.loss, model.total_steps)
                iter_time = (time.time()-start_time)/model.total_steps
                print("Iter time: {:.4f}".format(iter_time))
                
                with open( os.path.join(opt.checkpoints_dir, opt.name,'log.txt'), 'a') as f:
                    f.write(f"Step: {model.total_steps}, Time: {iter_time:.4f}, Lr: {current_lr:.6f}, Loss: {model.loss:.4f} | {margin_info}\n")

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.train()
            model.save_networks('model_epoch_%s.pth' % epoch)

        # Validation
        print(f"Running validation at epoch {epoch}...")
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader)
        
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {:.4f}; ap: {:.4f}".format(epoch, acc, ap))
        
        with open( os.path.join(opt.checkpoints_dir, opt.name,'log.txt'), 'a') as f:
            f.write(f"(Val @ epoch {epoch}) acc: {acc:.4f}; ap: {ap:.4f}\n")

        model.train()