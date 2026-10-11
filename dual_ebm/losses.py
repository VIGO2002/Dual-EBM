"""Published Eqs. (3)-(5): cross entropy + class conditional margin + consistency."""
import torch
import torch.nn.functional as F


def compute_dual_ebm_loss(logits, labels, e_real, e_fake,
                          noisy_real=None, noisy_fake=None,
                          margin=5.0, lambda_margin=0.5, lambda_consistency=0.1):
    labels = labels.long().reshape(-1)
    real, fake = e_real.reshape(-1), e_fake.reshape(-1)
    if logits.shape != (labels.numel(), 2):
        raise ValueError("Logits should have shape [batch, 2].")
    loss_cls = F.cross_entropy(logits, labels)
    loss_margin = logits.new_zeros(())
    real_mask, fake_mask = labels == 0, labels == 1
    if torch.any(real_mask):
        loss_margin = loss_margin + F.relu(real[real_mask] - fake[real_mask] + margin).mean()
    if torch.any(fake_mask):
        loss_margin = loss_margin + F.relu(fake[fake_mask] - real[fake_mask] + margin).mean()
    loss_cons = logits.new_zeros(())
    if (noisy_real is None) != (noisy_fake is None):
        raise ValueError("Both perturbed energies are required.")
    if noisy_real is not None:
        loss_cons = F.mse_loss(e_real, noisy_real) + F.mse_loss(e_fake, noisy_fake)
    return {
        "total": loss_cls + lambda_margin * loss_margin + lambda_consistency * loss_cons,
        "classification": loss_cls,
        "margin": loss_margin,
        "consistency": loss_cons,
    }
