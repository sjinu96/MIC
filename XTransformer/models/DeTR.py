from ast import Import
import ssl
import torch
from torch import nn
import os
import sys
from models.xtransformer import XTransformer
from models.visual_extractor import VisualFeatureExtractor
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd()))) # XTransformer.
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd())))) # Transformer
from XTransformer.lib.config import cfg, cfg_from_file



# 1.19d
# Densenet + XTransformer 

class DeTR(nn.Module):

    def __init__(self, visual_encoder, transformer, cfg, args, contra_att = False):

        super().__init__()
        self.cfg = cfg
        self.args = args
        self.encoder = visual_encoder
        self.transformer = transformer

        # Contrastive Attention
        self.contra_att = contra_att
    
    def forward(self, **kwargs):
        # print('---------------DeTR.forward() Start---------------')
        input_feats = self.encoder(kwargs['INPUT_IMAGES']) # [196, B, 512]
        if not self.cfg.MODEL.ENCODER_FUSION_MODE =='no':

            if 'CA' in self.cfg.MODEL.ENCODER_FUSION_MODE:
                
                # kwargs['NORMAL_IMAGES'] : [B, 10, 3, 448, 448]
                batch, N, _, _, _ = kwargs['NORMAL_IMAGES'].shape
                # Gradient는 기록 x
                # 1장 했을 때는 detach만으로 충분하지만, 여러 장이기 때문에 Encoder에서 메모리 부하올 수 있음.
                with torch.no_grad():                 
                    # input : [B*10, 3, 448, 448]
                    # output : [196, B*10, 1024]
                    normal_feats = self.encoder(kwargs['NORMAL_IMAGES'].flatten(0,1))  

                    seq_len, _, hid_dim = normal_feats.shape # [196, B*10, 1024]


                normal_feats = normal_feats.view(seq_len, batch, N, hid_dim) # [196, B, 10, 1024]

                global_normal_feats = normal_feats.mean(axis=0) # [B,.10, 1024]
                
                global_normal_feats.requires_grad = True # Attention 부터는 Gradient 필요.

            else:
                normal_feats = self.encoder(kwargs['NORMAL_IMAGES'])
                # output : [196, B, 512] 

                if self.cfg.MODEL.ENCODER_FUSION_GRAD =='detach':
                    # print('detach')
                    normal_feats = normal_feats.detach()
     

        if self.cfg.MODEL.ENCODER_TYPE =='resnet152':
            if 'ewp':
                diff_feats = []
                for input_feat, normal_feat in zip(input_feats, normal_feats):
                    diff_feat = input_feat * normal_feat
                    diff_feats.append(diff_feat)
                
                diff_feats = torch.cat(diff_feats, 0) # [1029, B, 512]
                att_feats = diff_feats.permute(1,0,2) # [B, 1029, 512]

        # Input : (448, 448)기준
        elif self.cfg.MODEL.ENCODER_TYPE =='densenet121':
            if self.cfg.MODEL.ENCODER_FUSION_MODE == 'concat+ewp': # ewp + Concat
                diff_feats = input_feats * normal_feats # [196, B, 512]
                diff_feats = torch.cat([input_feats, diff_feats], axis=0) # [392, B, 512]
                att_feats = diff_feats.permute(1,0,2) # [B, 392, 512]
                # print(att_feats.shape)
                # print(att_feats.shape)
            elif self.cfg.MODEL.ENCODER_FUSION_MODE == 'ewp':
                diff_feats = input_feats * normal_feats # [196, B, 512]
                att_feats = diff_feats.permute(1,0,2) # [B, 196, 512]
            elif self.cfg.MODEL.ENCODER_FUSION_MODE == 'BiP': # Bi-Linear Pooling + Concat

                att_feats = '' # [B, 392, 512]
            elif self.cfg.MODEL.ENCODER_FUSION_MODE =='no':
                diff_feats = input_feats # normal image 사용 x
                att_feats = diff_feats.permute(1,0,2) # [B, 196, 512]
                # print(att_feats.shape)
            elif self.cfg.MODEL.ENCODER_FUSION_MODE =='concat+CA':
                
                # input : input_feats - [196, B, 1024] , global_normal_feats - [B, N, 1024]
                # output : contra_feats - [196, B, 1024]
                # print('DETR : self.contra_att(input_feats, global_normal_feats')
                # print('input_feats', input_feats.shape)
                # print('global_normal_feats', global_normal_feats.shape) 
                contra_feats = self.contra_att(input_feats, global_normal_feats) # [196, B, 1024]
                # print('cotra_feats.shape', contra_feats.shape)
                # assert 1==0
                
                att_feats = torch.cat([input_feats, contra_feats], axis = 0) # [392, B, 1024]
                att_feats = att_feats.permute(1, 0, 2) # [B, 392, 1024]

                # print(att_feats.shape)
                # assert 1==0
            elif self.cfg.MODEL.ENCODER_FUSION_MODE =='CA':
                
                contra_feats = self.contra_att(input_feats, global_normal_feats) # [196, B, 1024] (x4일 경우 196 ->784)
                
                att_feats = contra_feats.permute(1, 0, 2) # [B, 196, 1024]

        # print('Detr, att_feats', att_feats.shape) # [2, 392, 512]
        att_feats_copy = att_feats.data # jsp gpu 방지.
        # print('1', att_feats.shape)
        # print('2', att_feats_copy.shape)
        # print(att_feats.shape)
        att_feats_copy, att_mask = self.get_attn_relation(att_feats_copy)

        # att_feats = att_feats.cuda()


        # att_feats, att_mask = self.bridge(input_feats, normal_feats)

        # print('att_feats.shape', att_feats.shape)
        # print('att_mask.shape ', att_mask.shape)
   
        # print(att_feats.shape, att_feats.requires_grad)
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask

        
        decoder_out = self.transformer(**kwargs) # [B, 51, 1807]
        # print(decoder_out.shape)
        
        # print('---------------DeTR.forward() End---------------')
  
        return decoder_out
    
    def bridge(self, input_feats, normal_feats):
        
        # print('---------------DeTR.bridge Start---------------')

        ##### 1월 20일, att_feats 쿠다 처리하기
        # (쿠다로 다시 반환해주는 편이 좋음)

        
        att_feats = self.get_att_feats(input_feats, normal_feats, cfg.MODEL.ENCODER_FUSION_MODE)
        
        att_feats = att_feats.cpu()

        att_feats, att_mask = self.get_attn_relation(att_feats)

        att_feats = att_feats.cuda()
        # print('---------------DeTR.bridge() End---------------')
        return att_feats, att_mask

    def get_att_feats(self, input_feats, normal_feats, fusion_mode):
        """배치단위 Feature 추출"""

        if fusion_mode == 'ewp':
            
            diff_feats = []
            for input_feat, normal_feat in zip(input_feats, normal_feats):
                diff_feat = input_feat * normal_feat
                diff_feats.append(diff_feat)
            
            diff_feats = torch.cat(diff_feats, 0) # [1029, B, 512]
            total_features = diff_feats.permute(1,0,2) # [B, 1029, 512]
        # # input image features
        # # [7x7, B, 512] , [14x14, B, 512], [28x28, B, 512]
        # conv5_diff_features = conv5_fc_features * conv5_norm_features  
        # conv4_diff_features = conv4_fc_features * conv4_norm_features   
        # conv3_diff_features = conv3_fc_features * conv3_norm_features 

        # [7x7+14x14+28x28, B, 512] = [1029, B, 512]
        # diff_features = torch.cat([conv5_diff_features, conv4_diff_features, conv3_diff_features])

        # [B, 1029, 512]
        # total_features = diff_features.permute(1,0,2)
        
        return total_features 
    # 기존에 data_loader.sample_collate 에서 처리하던 친구들.


    # att_feats : 그냥 그대로 넘기면 된다.
    # tmp_mask : [B, 1029] 의 모두 1임.
    # 나중에 sc_att Layer에서 이쁘게 잘 처리됨.
    # 우리는 굳이 att_feat가 모두 똑같기 때문에 신경 쓸 필요가 업슴

 
    def get_attn_relation(self, att_feats):
        ''' att_feats : cuda
            att_masks : to cuda'''
        # print('---------------get_attn_relation start---------------')
        # print(att_feats.shape)
        # 이전 att_feats = tuple([1029 512], [1029 512], ..., [1029 512], ) 
        # 각각 np.ndarray, Batch_size 개수만큼 튜플.
        # print(att_feats.shape # [B, 1029, 512]
        
        # 이전 atts_num : [1029, 1029, ..., 1029] 
        # 역시 배치 갯수만큼
        atts_num = [x.shape[0] for x in att_feats] #
        # print(atts_num)
        # print('att_feat.shape', att_feats[0].shape)
        max_att_num = np.max(atts_num) # 
        # print(max_att_num)

        ### jsp
        # 사실상 모든 feature 길이와 hiden dim이 갖기 때문에 아래 코드는 필요 없음.

        # feat_arr = []
        # mask_arr = []
        # for i, num in enumerate(atts_num):

        #     # batch 2이면, 
        #     # att_feats[0] : 첫번째 np.ndarray feature
        #     # att_feats[1] : 두번째 np.ndarray feature

        #     # tmp_feat : [1, 1029, 512] 의 np.ndarray
        #     tmp_feat = np.zeros((1, max_att_num, att_feats[i].shape[1]), dtype=np.float32)
        #     print('i', i)
            
        #     # 변경 전 : att_feats[0].shape : [1029, 512] (이전) - 단 , numpy
        #     # 변경 후 : att_feats[0].shape : [1029, 512] (이후) = 단, tensor)
        
            
        #     # 복사. 
        #     tmp_feat[:, 0:att_feats[i].shape[0], :] = att_feats[i]

        #     # cpu -> gpu
        #     feat_arr.append(torch.from_numpy(tmp_feat)) # 


        #     tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        #     tmp_mask[:, 0:num] = 1
        #     mask_arr.append(torch.from_numpy(tmp_mask))

        # att_feats.shape : tensor [B, 1029, 512] (이전)
        # att_feats = torch.cat(feat_arr, 0)
        
        # att_mask.shape : tensor [B, 1029] (이전)
        # print(att_feats.shape)
        att_mask = torch.ones((att_feats.shape[:2])).cuda()
        # print(att_feats.shape[:2])
        # print(att_mask.shape)
        # 


        return att_feats, att_mask



if __name__ =='__main__':
    print(cfg)
