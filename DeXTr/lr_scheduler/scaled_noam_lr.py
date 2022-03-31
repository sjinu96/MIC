from random import gammavariate
import torch
from lib.config import cfg
from bisect import bisect_right

class ScaledNoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        model_size,
        factor,
        warmup,
        milestones,
        gamma,
        last_epoch=-1,
    ):

        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self.milestones = milestones
        self.gamma = gamma

        self.last_real_epoch=0
        super(ScaledNoamLR, self).__init__(optimizer, last_epoch)
        

    def get_lr(self):

        return [
            (base_lr * self.factor * self.model_size ** (-0.5) * \
            min((self.last_epoch + 1) ** (-0.5), (self.last_epoch + 1) * self.warmup ** (-1.5)))
            if param_group['type'] != 'extractor' else 
            (base_lr * self.factor * self.model_size ** (-0.5) * \
            min((self.last_epoch + 1) ** (-0.5), (self.last_epoch + 1) * self.warmup ** (-1.5))) * (self.gamma ** bisect_right(self.milestones, self.last_real_epoch))
            for (base_lr, param_group) in zip(self.base_lrs, self.optimizer.param_groups)
        ]
        