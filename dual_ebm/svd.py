"""SVD residual adapter using explicit modules rather than forward hooks.

Checkpoint key and tensor shapes are compatible with the historical UFD
implementation (weight_main, S_residual, U_residual, V_residual).
"""
import torch
import torch.nn.functional as F
from torch import nn


class SVDResidualLinear(nn.Module):
    def __init__(self, original, ratio=0.25):
        super().__init__()
        if not 0 < ratio < 1:
            raise ValueError("SVD ratio must be between zero and one.")
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(original.weight.detach(), full_matrices=False)
            rank = max(1, min(len(S) - 1, int(len(S) * ratio)))
            main = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
        self.weight_main = nn.Parameter(main.contiguous(), requires_grad=False)
        self.S_residual = nn.Parameter(S[rank:].clone())
        self.U_residual = nn.Parameter(U[:, rank:].clone())
        self.V_residual = nn.Parameter(Vh[rank:, :].T.contiguous())
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        residual = (self.U_residual * self.S_residual[None, :]) @ self.V_residual.T
        return F.linear(x, self.weight_main + residual, self.bias)


def apply_svd_to_vision_attention(vision_model, ratio=0.25):
    for layer in vision_model.encoder.layers:
        attn = layer.self_attn
        for name in ("k_proj", "v_proj", "q_proj", "out_proj"):
            setattr(attn, name, SVDResidualLinear(getattr(attn, name), ratio))
    return vision_model
