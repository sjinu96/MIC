import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
from models.DeTR import DeTR
import datasets
from datasets.utils import init_train_transform # jsp
from datasets.utils import init_val_transform
from datasets.utils import init_normal_transform
from datasets.utils import init_normal_transform_ca
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")

        self.rl_stage = False
        self.setup_logging()
        
        self.train_transform = init_train_transform(cfg.DATA_LOADER.RESIZE, cfg.DATA_LOADER.CROP_SIZE)
        self.val_transform = init_val_transform(cfg.DATA_LOADER.RESIZE, cfg.DATA_LOADER.CROP_SIZE)
        
        if cfg.DATA_LOADER.LEVERAGE_NORMAL =='CA':
            print('Initialize normal transform (CA)')
            
            # only crop transform (Because of using pre-transformed tensor)
            self.normal_transform = init_normal_transform_ca(cfg.DATA_LOADER.CROP_SIZE)
        else:
            self.normal_transform = init_normal_transform(cfg.DATA_LOADER.RESIZE, cfg.DATA_LOADER.CROP_SIZE)
            


        self.setup_dataset()
        

        self.setup_network()


        self.val_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.VAL_ID,
            gv_feat = cfg.DATA_LOADER.VAL_GV_FEAT,
            input_images = cfg.DATA_LOADER.VAL_INPUT_IMAGES, # jsp
            normal_images = cfg.DATA_LOADER.VAL_NORMAL_IMAGES,  # jsp
            # att_feats = cfg.DATA_LOADER.VAL_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE,
            all_file = cfg.DATA_LOADER.ALL_ID,
            start_index = 865,
            input_transform = self.val_transform, # jsp
            normal_transform = self.normal_transform, # jsp
            leverage_normal = cfg.DATA_LOADER.LEVERAGE_NORMAL, 
            num_ca = cfg.DATA_LOADER.NUM_CA 
        ) # changed
        self.test_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            gv_feat = cfg.DATA_LOADER.TEST_GV_FEAT,
            input_images = cfg.DATA_LOADER.TEST_INPUT_IMAGES, # jsp
            normal_images = cfg.DATA_LOADER.TEST_NORMAL_IMAGES,  # jsp
            # att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE,
            all_file = cfg.DATA_LOADER.ALL_ID,
            input_transform = self.val_transform, # jsp - test도 VAL로.
            normal_transform = self.normal_transform, # jsp : 흠..
            start_index = 973,
            leverage_normal = cfg.DATA_LOADER.LEVERAGE_NORMAL, 
            num_ca = cfg.DATA_LOADER.NUM_CA 
        ) # changed
        self.scorer = Scorer()
        

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        transformer = models.create(cfg.MODEL.TYPE) # X-LAN, X-Tranfsormer # jsp : model 변경
        
        # if self.args.model_name =='resnet152':
        encoder = models.create_encoder(cfg.MODEL.ENCODER_TYPE, cfg.MODEL.ENCODER_PRETRAINED, cfg)# jsp : densenet / resnet

        contra_att = models.create_contra_att(cfg)
        # elif self.args.model_name =='densenet121':
        #     if args.ENCODER_PRETRAINED_PATH is not None: # custom pre-trained ENcoder
        #         encoder = models.create_encoder(cfg.MODEL.ENCODER_TYPE, cfg.MODEL.ENCODER_PRETRAINED, cfg.MODEL.ENCODER_CFG_PATH, cfg.MODEL.ENCODER_PRETRAINED_PATH) 
        #     else: # Image-net Pretrained
        #         print('Image-net DenseNet')
        #         encoder = models.create_encoder(cfg.MODEL.ENCODER_TYPE, cfg.MODEL.ENCODER_PRETRAINED, args.ENCODER_CFG_PATH)
        if cfg.DATA_LOADER.LEVERAGE_NORMAL =='CA':
            print('Use DeTR with contra_attention_block(# of normal images : {})'.format(cfg.DATA_LOADER.NUM_CA))
            model = DeTR(encoder, transformer, cfg, self.args, contra_att) # jsp : model 변경
        else:
            model = DeTR(encoder, transformer, cfg, self.args) # jsp : model 변경



        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel( # jsp : model 변경
                model.to(self.device), 
                device_ids = [self.args.local_rank], 
                output_device = self.args.local_rank,
                broadcast_buffers = False
            )
            
        else:
            self.model = torch.nn.DataParallel(model).cuda() # jsp : model 변경

        # jsp :  트랜스포머.
        if self.args.resume > 0:

            print('Training through resume : model {}'.format(self.args.resume))
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                    map_location=lambda storage, loc: storage)
            )

        # 모델 빌드할 때. 
        # for key, param in self.model.named_parameters():
        #     print(key, param.requires_grad, param.device)
        
        self.display_num_parameters()
        # import torchsummary
        # torchsummary.summary(self.model.module.encoder, input_size=(3,224,224), device='cuda')
        # assert 1==0
        self.optim = Optimizer(self.model)
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).cuda()
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).cuda()
  
    def setup_dataset(self):
        
        self.coco_set = datasets.coco_dataset.CocoDataset(            
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID, 
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH, 
            target_seq = cfg.DATA_LOADER.TARGET_SEQ_PATH,
            gv_feat_path = cfg.DATA_LOADER.TRAIN_GV_FEAT, 
            # att_feats_folder = cfg.DATA_LOADER.TRAIN_ATT_FEATS, 
            input_images_folder = cfg.DATA_LOADER.TRAIN_INPUT_IMAGES, # jsp
            normal_images_folder = cfg.DATA_LOADER.TRAIN_NORMAL_IMAGES, # Jsp
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feat_num = cfg.DATA_LOADER.MAX_FEAT,
            all_file = cfg.DATA_LOADER.ALL_ID,
            input_transform = self.train_transform,
            normal_transform = self.normal_transform,
            leverage_normal = cfg.DATA_LOADER.LEVERAGE_NORMAL,
            num_ca = cfg.DATA_LOADER.NUM_CA,
        )

    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None
            
        val_res = self.val_evaler(self.model, 'val_' + str(epoch + 1))  # jsp : model 변경완료 1.19 16시
        self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        self.logger.info(str(val_res))

        test_res = self.test_evaler(self.model,'test_' + str(epoch + 1)) # jsp : model 자동완료 1.19 16시
        self.logger.info('######## Epoch (TEST)' + str(epoch + 1) + ' ########')
        self.logger.info(str(test_res))

        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_folder):
            os.mkdir(snapshot_folder)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model", epoch+1))

    def make_kwargs(self, indices, input_seq, target_seq, gv_feat, input_images, normal_images): # jsp : att_feats, att_mask):
        seq_mask = (input_seq > 0).type(torch.cuda.LongTensor)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()

        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.GLOBAL_FEAT: gv_feat,
            cfg.PARAM.INPUT_IMAGES: input_images, # jsp
            cfg.PARAM.NORMAL_IMAGES: normal_images,  # jsp
            # cfg.PARAM.ATT_FEATS: att_feats,
            # cfg.PARAM.ATT_FEATS_MASK: att_mask
        }
        return kwargs



    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            # self.model.module.ss_prob = ss_prob # 
            self.model.module.transformer.ss_prob = ss_prob # jsp : model 변경완료 1.19 16시
    
    def display(self, iteration, data_time, batch_time, losses, loss_info):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        self.logger.info('Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
        for name in sorted(loss_info):
            self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()

    

    ### input : kwargs
    # kwargs 중 images 는 encoder로
    # 그로부터 att_feats를 추출
    # kwrags 중 나머지와 att_feats는 transformer로

    # 일단 model : transformer로 사용
    # 그리고 encoder : feature extractor로 사용
    # resnet, fc 포함 -> att_feats 대체
    # kwargs : idx, input_seq, target_seq, gv_feats, input_images, normal_imaegs.. 
    def forward(self, kwargs):
        if self.rl_stage == False:
           
            logit = self.model(**kwargs) # jsp: model
            #print('logit.shape : ', logit.shape) # [B, 51, 1807]
            #print('logit : ', logit) # tensor, device='cuda:0', grad_fn=<AddBackward0>

            loss, loss_info = self.xe_criterion(logit, kwargs[cfg.PARAM.TARGET_SENT])
            # print(loss, loss_info) # tensor(6.8362, device='cuda:0', grad_fn=<MeanBackward0>) {'LabelSmoothing Loss': 6.836213111877441}
            
        else:
            # print('SCSL Start')
            ids = kwargs[cfg.PARAM.INDICES]
            gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
            # att_feats = kwargs[cfg.PARAM.ATT_FEATS] # jsp
            # att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK] # jsp
            # input_images = kwargs[cfg.PARAM.INPUT_IMAGES]
            # normal_images = kwargs[cfg.PARAM.NORMAL_IMAGES]
            # max
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = True
            # kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat # jsp : 이게 필요?
            # kwargs[cfg.PARAM.ATT_FEATS] = att_feats  #jsp
            # kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask # jsp
            input_feats = self.model.module.encoder(kwargs['INPUT_IMAGES']) # [conv5, conv4, conv3, ...]
            if not cfg.MODEL.ENCODER_FUSION_MODE =='no':
                normal_feats = self.model.module.encoder(kwargs['NORMAL_IMAGES'])
            if cfg.MODEL.ENCODER_FUSION_GRAD =='detach':
                # print('detach')
                normal_feats = normal_feats.detach()
            # # jsp : 확실하지 않음. 
            # # 일단 encoder도 학습을 시켜야 하기 때문에 eval() 전에 포함해놨지만,,
            # if cfg.MODEL.ENCODER_FUSION_MODE == 'concat+ewp': # ewp + Concat
            #     diff_feats = input_feats * normal_feats # [196, B, 512]
            #     diff_feats = torch.cat([input_feats, diff_feats], axis=0) # [392, B, 512]
            #     att_feats = diff_feats.permute(1,0,2) # [B, 392, 512]
            #     # print(att_feats.shape)
            #     # print(att_feats.shape)
            # elif cfg.MODEL.ENCODER_FUSION_MODE == 'ewp':
            #     diff_feats = input_feats * normal_feats # [196, B, 512]
            #     att_feats = diff_feats.permute(1,0,2) # [B, 196, 512]
            # elif cfg.MODEL.ENCODER_FUSION_MODE == 'BiP': # Bi-Linear Pooling + Concat

            #     att_feats = '' # [B, 392, 512]
            # elif cfg.MODEL.ENCODER_FUSION_MODE =='no':
            #     diff_feats = input_feats # normal image 사용 x
            #     att_feats = diff_feats.permute(1,0,2) # [B, 196, 512]
            #     # print(att_feats.shape)
            # att_feats_copy = att_feats.data # jsp gpu 방지.
            # # print('1', att_feats.shape)
            # # print('2', att_feats_copy.shape)
            # # print(att_feats.shape)
            # att_feats_copy, att_mask = self.model.module.get_attn_relation(att_feats_copy)

            # # att_feats = att_feats.cuda()


            # # att_feats, att_mask = self.bridge(input_feats, normal_feats)

            # # print('att_feats.shape', att_feats.shape)
            # # print('att_mask.shape ', att_mask.shape)
    
            # # print(att_feats.shape, att_feats.requires_grad)
            # kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            # kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        

            # self.model.eval()
            
            # # 이름만 decode고 encoder도 진행함.
            # # decoder.forward는 최종 단어까지.
            # # 단, 학습 때는 (.. )까지)
            with torch.no_grad():
                # seq_max, logP_max = self.model.module.decode(**kwargs) #
                seq_max, logP_max = self.model.module.transformer.decode(**kwargs) # jsp : model
            
            self.model.train()
            
            rewards_max, rewards_info_max = self.scorer(ids, seq_max.data.cpu().numpy().tolist())
            rewards_max = utils.expand_numpy(rewards_max)

            ids = utils.expand_numpy(ids)
            # gv_feat = utils.expand_tensor(gv_feat, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_feats = utils.expand_tensor(att_feats, cfg.DATA_LOADER.SEQ_PER_IMG)
            att_mask = utils.expand_tensor(att_mask, cfg.DATA_LOADER.SEQ_PER_IMG)

            # sample
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = False
            # kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
            kwargs[cfg.PARAM.ATT_FEATS] = att_feats
            kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

            # seq_max, logP_max = self.model.module.decode(**kwargs) 
            seq_sample, logP_sample = self.model.module.transformer.decode(**kwargs) # jsp : model (module을 먼저 뽑는 게 맞을듯)
            # print('---------seq_sample----------')
            # print(seq_sample)
            
            rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist())

            rewards = rewards_sample - rewards_max
            rewards = torch.from_numpy(rewards).float().cuda()
            loss = self.rl_criterion(seq_sample, logP_sample, rewards)
            
            loss_info = {}
            for key in rewards_info_sample:
                loss_info[key + '_sample'] = rewards_info_sample[key]
            for key in rewards_info_max:
                loss_info[key + '_max'] = rewards_info_max[key]

        return loss, loss_info

    ### jsp : 
    # 
    # model은 그대로 이름을 x-transformer로 바꾸고,
    # resnet, fc를 encoder로 해서 투트랙으로 해놓자.(캡슐화)
    def train(self):
        self.model.train() # jsp 변화 x
        self.optim.zero_grad()

        iteration = 0
        for epoch in  range(cfg.SOLVER.MAX_EPOCH):
            if epoch == cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            self.setup_loader(epoch)

            start = time.time()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            losses = AverageMeter()

            ### jsp
            # gv_feat : 사용 x
            # att_feat : [1029,512] feature
            # att_mask
            # coco_dataset.py에서 조정.
            # for _, (indices, input_seq, target_seq, gv_feat, att_feats, att_mask) in enumerate(self.training_loader):
            # count=0 # 메모리 테스트용
            for _, (indices, input_seq, target_seq, gv_feat, input_images, normal_images) in enumerate(self.training_loader):
                # print('----------------Train Batch Start------------------')
                # print("indices:", indices)
            
                
                data_time.update(time.time() - start)

                input_seq = input_seq.cuda()
                target_seq = target_seq.cuda()
                gv_feat = gv_feat.cuda()
                # att_feats = att_feats.cuda()
                # att_mask = att_mask.cuda()
                input_images = input_images.cuda() # jsp
                normal_images = normal_images.cuda() # jsp

                # print('input_seq : ', type(input_seq), input_seq.shape) # [B,201]
                # print('target_seq : ', type(target_seq), target_seq.shape) # [B, 201]
                # print('gv_feat : ', type(gv_feat), gv_feat.shape) # [1,1] zeros
                # print('att_feats : ', type(att_feats), att_feats.shape) # [B,1029,51 2]
                # print('att_mask :', type(att_mask), att_mask.shape) # [B,1029]
                # print('input_images : ', type(input_images), input_images.shape) # [B, 3, 224, 224]
                # print('normal_images : ', type(normal_images), normal_images.shape) # [B, 3, 224, 224]
                
                kwargs = self.make_kwargs(indices, input_seq, target_seq, gv_feat, input_images, normal_images) # jsp : att_feats, att_mask)
                
                # print(kwargs.keys())



                loss, loss_info = self.forward(kwargs)
                # print(loss.shape, loss_info.shape)

                # test1=(self.model.module.encoder.conv5_fc.weight[:5, :5]).cpu().detach().numpy().copy()
                # test2=(self.model.module.transformer.decoder.proj_norm[0].weight[:5, :5]).cpu().detach().numpy().copy()
                # test3=(self.model.module.transformer.encoder.proj_norm[0].weight[:5, :5]).cpu().detach().numpy().copy()
                loss.backward()

                # print(loss)
                utils.clip_gradient(self.optim.optimizer, self.model,
                    cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                
                self.optim.step()

                self.optim.zero_grad()
                self.optim.scheduler_step('Iter')

                # print(self.optim)
                
                batch_time.update(time.time() - start)
                start = time.time()

                losses.update(loss.item())
                self.display(iteration, data_time, batch_time, losses, loss_info)
                iteration += 1

                # print(test1-self.model.module.encoder.conv5_fc.weight[:5, :5].cpu().detach().numpy())
                # print(test2-self.model.module.transformer.encoder.proj_norm[0].weight[:5, :5].cpu().detach().numpy())                
                # print(test3-self.model.module.transformer.decoder.proj_norm[0].weight[:5, :5].cpu().detach().numpy())
                
                if self.distributed:
                    dist.barrier()
                # print('----------------Train Batch End(------------------')
                
                # torch.cuda.empty_cache()  # 흐으음..
                del loss
                #    count+=1 
                # if count == 10:
                #     break
                
                
            torch.cuda.empty_cache()

            if cfg.SOLVER.LR_POLICY.SUB_SCHEDULE: 
                self.optim.scheduler.last_real_epoch +=1 # Extractor는 Epoch에 따라서 추가적인 감쇄 존재.
            
            self.save_model(epoch)
            val = self.eval(epoch)
            self.optim.scheduler_step('Epoch', val)
            
            
            # Encoder만 Learning rate 조절.
            # MultiStep Scheduler + Scaled Noam

            

            self.scheduled_sampling(epoch)

            if self.distributed:
                dist.barrier()
            # assert 1==0


    def display_num_parameters(self):
        n_parameters_transformer=0
        n_parameters_visual_encoder=0
        n_parameters_contra_att=0
        n_parameters_train_transformer = 0
        n_parameters_train_visual_encoder = 0
        n_parameters_train_contra_att=0
        for key, param in self.model.named_parameters():
            if 'transformer' in key:
                n_parameters_transformer+=param.numel()
                if param.requires_grad:
                    n_parameters_train_transformer+=param.numel()

            if 'extractor' in key:
                n_parameters_visual_encoder+=param.numel()
                if param.requires_grad:
                        n_parameters_train_visual_encoder+=param.numel()
            
            if 'contra_att' in key:
                n_parameters_contra_att+=param.numel()
                if param.requires_grad:
                    n_parameters_train_contra_att+=param.numel()

        n_parameters = n_parameters_transformer + n_parameters_visual_encoder + n_parameters_contra_att
        n_train_parameters = n_parameters_train_transformer + n_parameters_train_visual_encoder + n_parameters_train_contra_att
        print('Total parameters: {}\n(Visual Encoder : {}, Contrastive Attention: {},Transformer : {})'.format(n_parameters, n_parameters_visual_encoder, n_parameters_contra_att, n_parameters_transformer))
        print('Trainable parameters: {}\n(Visual Encoder : {}, Contrastive Attention : {}, Transformer : {})'.format(n_train_parameters, n_parameters_train_visual_encoder, n_parameters_train_contra_att,   n_parameters_train_transformer))

def load_encoder_model(folder, resume):

    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=folder)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=resume)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args=[])

    
    # args = parse_args()
    print('Called with args:')
    print(args.folder)
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
    cfg.ROOT_DIR = args.folder
    
    transformer = models.create(cfg.MODEL.TYPE) # X-LAN, X-Tranfsormer # jsp : model 변경
        
    # if self.args.model_name =='resnet152':
    encoder = models.create_encoder(cfg.MODEL.ENCODER_TYPE, cfg.MODEL.ENCODER_PRETRAINED, cfg)# jsp : densenet / resnet

    contra_att = models.create_contra_att(cfg)
    # elif self.args.model_name =='densenet121':
    #     if args.ENCODER_PRETRAINED_PATH is not None: # custom pre-trained ENcoder
    #         encoder = models.create_encoder(cfg.MODEL.ENCODER_TYPE, cfg.MODEL.ENCODER_PRETRAINED, cfg.MODEL.ENCODER_CFG_PATH, cfg.MODEL.ENCODER_PRETRAINED_PATH) 
    #     else: # Image-net Pretrained
    #         print('Image-net DenseNet')
    #         encoder = models.create_encoder(cfg.MODEL.ENCODER_TYPE, cfg.MODEL.ENCODER_PRETRAINED, args.ENCODER_CFG_PATH)
    if cfg.DATA_LOADER.LEVERAGE_NORMAL =='CA':
        print('Use DeTR with contra_attention_block(# of normal images : {})'.format(cfg.DATA_LOADER.NUM_CA))
        model = DeTR(encoder, transformer, cfg, args, contra_att) # jsp : model 변경
    else:
        model = DeTR(encoder, transformer, cfg, args) # jsp : model 변경


    model = torch.nn.DataParallel(model).cuda() # jsp : model 변경

    snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
    resume_path = os.path.join(snapshot_folder, "caption_model" + "_" + str(args.resume) + ".pth")

    # jsp :  트랜스포머.
    
    print('Training through resume : model {}'.format(args.resume))
    model.load_state_dict(
        torch.load(resume_path, map_location=lambda storage, loc: storage)
    )

    # return True
    
    return model.module.encoder.extractor, cfg


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config.yml'))
         
    cfg.ROOT_DIR = args.folder

    print('---', cfg.ROOT_DIR)


    
   
    trainer = Trainer(args)
    trainer.train()




      