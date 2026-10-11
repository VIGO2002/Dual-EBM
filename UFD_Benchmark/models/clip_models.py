"""CLIP ViT-L/14 backbone with the Dual-EBM decision module."""
from torch import nn
from transformers import CLIPModel as HuggingFaceCLIPModel
from dual_ebm.model import DualEnergyHead
from dual_ebm.svd import apply_svd_to_vision_attention


class ClipModel(nn.Module):
    def __init__(self, name, num_classes=1, fix_backbone=False, use_svd=False,
                 noise_std=0.02, svd_rank_ratio=0.25):
        super().__init__()
        self.model = HuggingFaceCLIPModel.from_pretrained(name)
        self.model.gradient_checkpointing_enable()
        if use_svd:
            apply_svd_to_vision_attention(self.model.vision_model, svd_rank_ratio)
        self.fc = DualEnergyHead(self.model.vision_model.config.hidden_size,
                                 init_noise_std=noise_std)

    def forward(self, x, return_feature=False, return_energy=False):
        if self.training:
            x = x.requires_grad_(True)  # checkpointed CLIP with frozen weights
        features = self.model.vision_model(x)["pooler_output"]
        if return_feature:
            return features
        if self.training:
            return self.fc(features, training=True)
        logits, er, ef = self.fc(features, training=False)
        return (logits, er, ef) if return_energy else logits
