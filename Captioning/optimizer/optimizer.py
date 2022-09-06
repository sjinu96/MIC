import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.config import cfg
import lr_scheduler
from optimizer.radam import RAdam, AdamW

class Optimizer(nn.Module):
    def __init__(self, model):
        super(Optimizer, self).__init__()


        if cfg.SOLVER.LR_POLICY.SUB_SCHEDULE: # Extractor를 위한 스케쥴러
            self.milestones = cfg.SOLVER.LR_POLICY.STEPS
            self.gamma = cfg.SOLVER.LR_POLICY.GAMMA

            print('Use sub Scheduler : Extractor', self.milestones, self.gamma)
        else:
            self.milestones = cfg.SOLVER.LR_POLICY.STEPS # 사실 수정하긴 해야함
            self.gamma = 1 # 사실 제대로 수정하긴 해야함 2 (Scaled Noamlr_scheduler를 바꿔야함.)

        self.setup_optimizer(model)

        
        

    def setup_optimizer(self, model):
        params = []
        for key, value in model.named_parameters():
            # print(key , ":", value.requires_grad)
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR 
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS


            if (cfg.MODEL.ENCODER_TYPE =='resnet152') and ('conv' not in key):   # Resnet152
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            elif (cfg.MODEL.ENCODER_TYPE !='resnet152') and ('extractor' not in key): # Dense
                # print('학습 가능한 Transformer', key)
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "type": 'transformer'}]
            else :  # Encoder : Fine-Tuning
                # print('학습 가능한 Conv-layer', key)
                params += [{"params": [value], "lr": lr*0.1, "weight_decay": 0.0001, "type": 'extractor'}]

        if cfg.SOLVER.TYPE == 'SGD':
            self.optimizer = torch.optim.StmuGD(
                params, 
                lr = cfg.SOLVER.BASE_LR, 
                momentum = cfg.SOLVER.SGD.MOMENTUM
            )
        elif cfg.SOLVER.TYPE == 'ADAM':
            self.optimizer = torch.optim.Adam(
                params,
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAMAX':
            self.optimizer = torch.optim.Adamax(
                params,
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAGRAD':
            self.optimizer = torch.optim.Adagrad(
                params,
                lr = cfg.SOLVER.BASE_LR
            )
        elif cfg.SOLVER.TYPE == 'RMSPROP':
            self.optimizer = torch.optim.RMSprop(
                params, 
                lr = cfg.SOLVER.BASE_LR
            )
        elif cfg.SOLVER.TYPE == 'RADAM':
            self.optimizer = RAdam(
                params, 
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )

            
        else:
            raise NotImplementedError


        
        # for epoch in range(0, 400):
        #     if epoch in self.milestones:
                
        #         num = 0
        #         for param_group in self.optimizer.param_groups:
        #             if param_group['type'] == 'extractor':
        #                 before = param_group['lr']
        #                 param_group['lr'] = param_group['lr'] * self.gamma
        #                 after = param_group['lr']
        #                 num+=1
        #             else:
        #                 pass          
        #         print(num)
        #         print(self.gamma, self.milestones)
        #         print(epoch, before, after)   
                
        # assert 1==0    

        if cfg.SOLVER.LR_POLICY.TYPE == 'Fix':
            self.scheduler = None
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size = cfg.SOLVER.LR_POLICY.STEP_SIZE, 
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,  
                factor = cfg.SOLVER.LR_POLICY.PLATEAU_FACTOR, 
                patience = cfg.SOLVER.LR_POLICY.PLATEAU_PATIENCE
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Noam':
            self.scheduler = lr_scheduler.create(
                'Noam', 
                self.optimizer,
                model_size = cfg.SOLVER.LR_POLICY.MODEL_SIZE,
                factor = cfg.SOLVER.LR_POLICY.FACTOR,
                warmup = cfg.SOLVER.LR_POLICY.WARMUP
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'ScaledNoam': # Encoder에도 적용
            print('Lr Scheduler : ScaledNoam') # jsp
            print('factor :', cfg.SOLVER.LR_POLICY.FACTOR*cfg.SOLVER.BASE_LR)
            
            self.scheduler = lr_scheduler.create(
                'ScaledNoam', 
                self.optimizer,
                model_size = cfg.SOLVER.LR_POLICY.MODEL_SIZE,
                factor = cfg.SOLVER.LR_POLICY.FACTOR,
                warmup = cfg.SOLVER.LR_POLICY.WARMUP,
                milestones = self.milestones,
                gamma = self.gamma
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'MultiStep':
            print('Lr Scheduler : MultiStep') # jsp
            print('Steps : ', cfg.SOLVER.LR_POLICY.STEPS)
            print('decay_rate : ', cfg.SOLVER.LR_POLICY.GAMMA)
            self.scheduler = lr_scheduler.create(
                'MultiStep', 
                self.optimizer,
                milestones = cfg.SOLVER.LR_POLICY.STEPS,
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def scheduler_step(self, lrs_type, val=None):
        if self.scheduler is None:
            return

        if cfg.SOLVER.LR_POLICY.TYPE != 'Plateau':
            val = None

        if lrs_type == cfg.SOLVER.LR_POLICY.SETP_TYPE:
            self.scheduler.step(val)

    def get_lr(self): # 수정
        lr= []
        for param_group in self.optimizer.param_groups:
            lr.append(param_group['lr'])
        lr = sorted(list(set(lr)))
        return lr
    

    def scheduler_extractor(self, epoch):

        if epoch in self.milestones:
            for param_group in self.optimizer.param_groups:
                if param_group['type'] == 'extractor':
                    param_group['lr'] = param_group['lr'] * self.gamma
                    temp = param_group['lr']
                else:
                    pass

            print('Extractor Scheduler - epoch : {}, lr : {}'.format(epoch, temp) )
                

                