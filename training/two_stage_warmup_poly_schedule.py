# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from torch.optim.lr_scheduler import LRScheduler


class TwoStageWarmupPolySchedule(LRScheduler):
    def __init__(
        self,
        optimizer,
        num_backbone_params: int,
        warmup_steps: tuple[int, int],
        total_steps: int,
        poly_power: float,
        step_milestones: list = None,
        step_gamma: float = 0.1,
        last_epoch=-1,
    ):
        self.num_backbone_params = num_backbone_params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.poly_power = poly_power
        self.step_milestones = step_milestones  # fractions of post-warmup steps, e.g. [0.889, 0.963]
        self.step_gamma = step_gamma
        super().__init__(optimizer, last_epoch)

    def _apply_step_decay(self, adjusted: int, effective_steps: int, base_lr: float) -> float:
        lr = base_lr
        for frac in self.step_milestones:
            if adjusted >= int(frac * effective_steps):
                lr *= self.step_gamma
        return lr

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        non_vit_warmup, vit_warmup = self.warmup_steps
        for i, base_lr in enumerate(self.base_lrs):
            if i >= self.num_backbone_params:
                if non_vit_warmup > 0 and step < non_vit_warmup:
                    lr = base_lr * (step / non_vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup)
                    if self.step_milestones:
                        lr = self._apply_step_decay(adjusted, max_steps, base_lr)
                    else:
                        lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            else:
                if step < non_vit_warmup:
                    lr = 0
                elif step < non_vit_warmup + vit_warmup:
                    lr = base_lr * ((step - non_vit_warmup) / vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup - vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup - vit_warmup)
                    if self.step_milestones:
                        lr = self._apply_step_decay(adjusted, max_steps, base_lr)
                    else:
                        lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power

            lrs.append(lr)
        return lrs
    
