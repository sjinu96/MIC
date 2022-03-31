import os
import torch
from torchvision import transforms
from lib.config import cfg
from datasets.coco_dataset import CocoDataset
import samplers.distributed
import numpy as np
from datasets.utils import * # jsp (init_transform..)


def sample_collate(batch):
    # batch : [B, 6]
    # indices, input_seq, target_seq, gv_feat, att_feats = zip(*batch) # jsp
    indices, input_seq, target_seq, gv_feat, input_images, normal_images = zip(*batch) # jsp
    # print(type(indices))
    # print(type(gv_feat))
    # print(type(input_images))
    indices = np.stack(indices, axis=0).reshape(-1)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0) # 튜플 처리.
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    
    input_images = torch.cat([b.unsqueeze(0) for b in input_images], 0) # jsp
    normal_images = torch.cat([b.unsqueeze(0) for b in normal_images], 0) # jsp
    
    # for x in input_images:
    #     print('---Train----')
    #     print(x.shape)
    # print(type(indices), indices.shape) # [B, ]
    # print(type(target_seq), target_seq.shape) # [B,201]
    # print(type(input_seq), input_seq.shape) # [B, 201]
    # print(type(input_images), input_images.shape) # [B, 3, 224, 224]
    # print(type(normal_images), normal_images.shape) # [B, 3, 224, 224] In CA : [B, 10, 3, 224, 224]

    # print('DATALOADER - normal_images.shape', normal_images.shape)

    ### jsp
    # att_feats : 없음.


    # atts_num = [x.shape[0] for x in att_feats]
    # max_att_num = np.max(atts_num)

    # feat_arr = []
    # mask_arr = []
    # for i, num in enumerate(atts_num):
    #     tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
    #     tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
    #     feat_arr.append(torch.from_numpy(tmp_feat))

    #     tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
    #     tmp_mask[:, 0:num] = 1
    #     mask_arr.append(torch.from_numpy(tmp_mask))

    # att_feats = torch.cat(feat_arr, 0)
    # att_mask = torch.cat(mask_arr, 0)

    return indices, input_seq, target_seq, gv_feat, input_images, normal_images #att_feats, att_mask


def sample_collate_val(batch):
   
    indices, gv_feat, input_images, normal_images = zip(*batch) # att_feats = zip(*batch)
    indices = np.stack(indices, axis=0).reshape(-1)
    gv_feat = torch.cat([torch.from_numpy(b) for b in gv_feat], 0)

    # for x in input_images:
    #     print('---Test----')
    #     print(x.shape)

    # print('input_image.shape', input_images.shape)
    input_images = torch.cat([b.unsqueeze(0) for b in input_images], 0) # jsp
    normal_images = torch.cat([b.unsqueeze(0) for b in normal_images], 0) # jsp
   
    # atts_num = [x.shape[0] for x in att_feats]
    # max_att_num = np.max(atts_num)

    # feat_arr = []
    # mask_arr = []
    # for i, num in enumerate(atts_num):
    #     tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
    #     tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]
    #     feat_arr.append(torch.from_numpy(tmp_feat))

    #     tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
    #     tmp_mask[:, 0:num] = 1
    #     mask_arr.append(torch.from_numpy(tmp_mask))

    # att_feats = torch.cat(feat_arr, 0)
    # att_mask = torch.cat(mask_arr, 0)
    # print('---------sample_collate_val------------')
    # print(type(indices), indices.shape) # [B, ]
    # print(type(input_images), input_images.shape) # [B, 3, 224, 224]
    # print(type(normal_images), normal_images.shape) # [B, 3, 224, 224]  , In CA : [B, 10, 3, 224, 224]
    # print('CA 사용 시 normal_image.shape', normal_image.shape)
    # print('--------------------------------------------')
    return indices, gv_feat, input_images, normal_images #att_feats, att_mask


def load_train(distributed, epoch, coco_set):
    sampler = samplers.distributed.DistributedSampler(coco_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TRAIN.BATCH_SIZE,
        shuffle = shuffle, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = cfg.DATA_LOADER.DROP_LAST, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = sampler, 
        collate_fn = sample_collate,
    )

    return loader
    
def load_normal(distributed, epoch, coco_set):
    sampler = samplers.distributed.DistributedSampler(coco_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = 1,
        shuffle = shuffle, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = cfg.DATA_LOADER.DROP_LAST, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = sampler, 
        collate_fn = sample_collate
    )
    return loader

def load_val(image_ids_path, gv_feat_path, input_images_folder, normal_images_folder,
input_transform, normal_transform, leverage_normal, num_ca):  ##att_feats_folder):
    coco_set = CocoDataset(
        image_ids_path = image_ids_path, 
        input_seq = None, # 
        target_seq = None, # eval때는 두 개 안 받음.
        gv_feat_path = gv_feat_path, 
        # att_feats_folder = att_feats_folder,
        seq_per_img = 1, 
        max_feat_num = cfg.DATA_LOADER.MAX_FEAT,
        all_file = cfg.DATA_LOADER.ALL_ID,
        input_transform = input_transform,
        normal_transform = normal_transform,
        input_images_folder = input_images_folder, # jsp
        normal_images_folder = normal_images_folder, # Jsp
        leverage_normal = leverage_normal,
        num_ca = num_ca
    )

    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TEST.BATCH_SIZE,
        shuffle = False, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = False, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, 
        collate_fn = sample_collate_val
    )
    return loader
    
def load_normal_val(image_ids_path, gv_feat_path, input_images_folder, normal_images_folder, 
input_transform, normal_transform, leverage_normal, num_ca): #att_feats_folder):
    coco_set = CocoDataset(
        image_ids_path = image_ids_path, 
        input_seq = None, 
        target_seq = None, 
        gv_feat_path = gv_feat_path, 
        # att_feats_folder = att_feats_folder,
        seq_per_img = 1, 
        max_feat_num = cfg.DATA_LOADER.MAX_FEAT,
        all_file = cfg.DATA_LOADER.ALL_ID,
        input_transform = input_transform, 
        normal_transform = normal_transform,
        input_images_folder = input_images_folder, # jsp
        normal_images_folder = normal_images_folder, # Jsp
        leverage_normal = leverage_normal,
        num_ca = num_ca
    )

    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = 1,
        shuffle = False, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = False, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, 
        collate_fn = sample_collate_val
    )
    return loader
