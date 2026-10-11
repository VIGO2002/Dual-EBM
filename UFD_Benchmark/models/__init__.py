"""CLIP model factory. The original local AutoDL ViT-L/14 path is preferred."""
from pathlib import Path
from .clip_models import ClipModel

PRETRAINED = {
    "CLIP:ViT-B/16_svd": "openai/clip-vit-base-patch16",
    "CLIP:ViT-B/32_svd": "openai/clip-vit-base-patch32",
    "CLIP:ViT-L/14_svd": "openai/clip-vit-large-patch14",
}
LOCAL_CLIP = Path("/root/autodl-tmp/pretrained_models/clip-vit-large-patch14")


def get_model(arch, opt):
    if arch not in PRETRAINED:
        raise ValueError(f"Unsupported model {arch!r}. Supported: {list(PRETRAINED)}")
    name = str(LOCAL_CLIP) if arch == "CLIP:ViT-L/14_svd" and LOCAL_CLIP.is_dir() else PRETRAINED[arch]
    return ClipModel(
        name=name, num_classes=1,
        fix_backbone=getattr(opt, "fix_backbone", True),
        use_svd=getattr(opt, "use_svd", True),
        noise_std=getattr(opt, "noise_std", 0.02),
        svd_rank_ratio=0.25,
    )
