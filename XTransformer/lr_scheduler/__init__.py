from lr_scheduler.multi_step_lr import MultiStepLR
from lr_scheduler.noam_lr import NoamLR
from lr_scheduler.scaled_noam_lr import ScaledNoamLR
__factory = {
    'MultiStep': MultiStepLR,
    'Noam': NoamLR,
    'ScaledNoam': ScaledNoamLR
}

def names():
    return sorted(__factory.keys())

def create(name, optimizer, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown lr_scheduler:", name)
    return __factory[name](optimizer, *args, **kwargs)