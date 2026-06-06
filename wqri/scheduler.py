"""Learning-rate scheduler utilities."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Epoch-wise warm-up plus cosine decay scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, base_lr: float, final_lr: float, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            warmup_factor = float(step + 1) / float(max(1, self.warmup_steps))
            lr = self.base_lr * warmup_factor
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.final_lr + (self.base_lr - self.final_lr) * cosine_factor
        return [lr for _ in self.optimizer.param_groups]
