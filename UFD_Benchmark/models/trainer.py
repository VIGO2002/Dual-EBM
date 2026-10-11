"""Paper-aligned UFD training wrapper. Author: Chenwei He."""
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from .base_model import BaseModel
from models import get_model
from dual_ebm.losses import compute_dual_ebm_loss
from dual_ebm.schedule import lr_multiplier


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.model = get_model(opt.arch, opt)
        for m in self.model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, opt.init_gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
        if opt.fix_backbone:
            params = []
            for name, param in self.model.named_parameters():
                active = name.startswith("fc.") or any(
                    key in name for key in ("S_residual", "U_residual", "V_residual")
                )
                param.requires_grad_(active)
                if active:
                    params.append(param)
        else:
            params = list(self.model.parameters())
        if opt.optim in ("adam", "adamw"):
            self.optimizer = torch.optim.AdamW(
                params, lr=opt.lr, betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay)
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim must be adamw, adam (legacy), or sgd")
        self.model.to(self.device)
        self.scheduler = None

    def configure_scheduler(self, steps_per_epoch):
        if self.opt.warmup_steps != 0:
            raise ValueError("Paper training protocol does not use warmup.")
        self.scheduler = LambdaLR(
            self.optimizer, lambda step: lr_multiplier(
                step, steps_per_epoch, self.opt.linear_end,
                self.opt.restart_epochs, self.opt.minimum_lr_factor))

    def set_input(self, data):
        self.input = data[0].to(self.device)
        self.label = data[1].to(self.device).long()

    def forward(self):
        output = self.model(self.input)
        (self.logits, self.e_real, self.e_fake,
         self.e_real_noisy, self.e_fake_noisy) = output
        self.output = self.logits

    def get_loss(self):
        self.loss_terms = compute_dual_ebm_loss(
            self.logits, self.label, self.e_real, self.e_fake,
            self.e_real_noisy, self.e_fake_noisy,
            margin=self.opt.energy_margin, lambda_margin=self.opt.lambda_margin,
            lambda_consistency=self.opt.lambda_smooth)
        return self.loss_terms["total"]

    def test(self):
        self.model.eval()
        with torch.no_grad():
            self.logits, self.e_real, self.e_fake = self.model(
                self.input, return_energy=True)
            self.output = self.logits

    def optimize_parameters(self):
        self.model.train()
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad(set_to_none=True)
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.opt.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()  # Once per optimizer step.
