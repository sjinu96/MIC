from pydantic import annotated_types
import torch
import torch.nn as nn
import numpy as np


# 기존의 모델에 Contrastive Attention을 적용해봅시다.

class ContraAtt(nn.Module):


    def __init__(self, cfg):
        """ 
        att_type : dot product(Default) or Bi-Linear
        embed_dim : same to input dim(Default)
        """

        super(ContraAtt, self).__init__()
        self.cfg = cfg
        self.att_type = cfg.MODEL.CONTRA_ATT_TYPE
        self.att_dim = cfg.MODEL.ATT_FEATS_EMBED_DIM # xtransformer : 512 (1-31, 수정(1024->512))
        self.num_heads = cfg.MODEL.CONTRA_ATT_NUM_HEADS # 6 (Default)

        self.aggre_att = AggregatedAttention(self.att_dim, self.num_heads, self.att_type)
        self.diff_att = DifferentiateAttention(self.att_dim, self.att_type)
        
    
        self.update_feats = nn.Sequential(
            nn.Linear(in_features=2*self.att_dim, out_features=self.att_dim),
            nn.ReLU(),
            nn.LayerNorm(self.att_dim), # Normalization 추가.
            nn.Dropout(cfg.MODEL.DROPOUT_CA)
        )
        # self.drop_out = nn.Dropout(cfg.MODEL.DROPOUT_ATT_EMBED) 

    def forward(self, input_feats, global_normal_feats):
        """
        input_feats : [196, B, 1024] (혹은 512)
        normal_feats : [B, N, 1024]
        """

        src_len = input_feats.shape[0] # 196
        global_input_feats = input_feats.mean(axis=0) # [B,1024]


        closest_normal_feats = self.aggre_att(global_input_feats, global_normal_feats) # [B, 6, 1024]  
        # print('closest_normal_feats.shape', closest_normal_feats.shape) 
        common_information = self.diff_att(global_input_feats, closest_normal_feats) # [B, 1, 7, 1024] -- 1 : num of DA heads, 6 : # num of AA heads
        # print('common_information.shape', common_information.shape)


        # basic
        # AP
        common_information = common_information.squeeze(1).mean(axis=1) # [B, 1024]
        # print('AP common information',  common_information.shape)

        diff_input_feats = global_input_feats - common_information # [B, 1024]
        # print('diff_input_feats', diff_input_feats.shape)


        # input feats[196, B, 1024] + diff_input_feats[B, 1024] 
        # --> [196, B, 1024] + [196, B, 1024] by expand diff_input_feats
        # --> [196, B, 2048]  by concat 
        # --> [196, B, 1024]  by update_feats(Linear(2048, 2012))

        diff_input_feats_par = diff_input_feats.unsqueeze(0).expand(src_len, -1, -1) # [196, B, 1024]
        # print('diff_input_feats_par', diff_input_feats_par.shape)
        # print('input_feats.shape', input_feats.shape)

        contra_feats = self.update_feats(torch.cat([input_feats, diff_input_feats_par], dim=2)) # [196, B, 1024]
        # print('contra_feats', contra_feats.shape)
        
        
        return contra_feats


class AggregatedAttention(nn.Module): # Aggregated Attention  

    def __init__(self, att_dim, num_heads, att_type):
        super(AggregatedAttention, self).__init__()
        
        self.att_dim = att_dim
        self.num_heads = num_heads
        self.att_type = att_type

        if self.att_type =='dot':
            self.att_blocks = nn.ModuleList([DotAttentionBlock(att_dim) for _ in range(num_heads)])
            # self.att_blocks = [DotAttentionBlock(att_dim) for _ in range(num_heads)]
        if self.att_type =='BiP':
            self.att_blocks = nn.ModuleList([BilinearPoolingAttentionBlock(att_dim) for _ in range(num_heads)])


        # print(self.att_blocks)

    def forward(self, global_input_feats, global_normal_feats):
        """
        input : global_input_feats : [B, 1024], global_normal_feats : [B, N, 1024]
        output : closest_normal_feats : [B, n, 1024]
        """

        closest_normal_feats = []

        for idx in range(self.num_heads):
            
            # if self.att_type =='dot':
            # [B, query_len, hid_dim] * [B, key_len, hid_dim]
            closest_normal_feat = self.att_blocks[idx](global_input_feats.unsqueeze(1), global_normal_feats) # [B, 1, 1024]
            # print('closest_normal_feat.shape', closest_normal_feat.shape)    
            # print('closest_normal_feat in AggregatedAttention.forward ', closest_normal_feat.shape)
            closest_normal_feats.append(closest_normal_feat.unsqueeze(0))

        closest_normal_feats = torch.cat(closest_normal_feats) # [n, B, 1, 1024] (n=6)
        # print('closest_normal_feats.shape', closest_normal_feats.shape)
        closest_normal_feats = closest_normal_feats.permute(1,0,2,3) # [B, n, 1, 1024] 
        closest_normal_feats = closest_normal_feats.squeeze(2) # [B, n, 1024]

        return closest_normal_feats

class DifferentiateAttention(nn.Module): # Aggregated Attention  

    def __init__(self, att_dim, att_type, num_heads = 1):
        super(DifferentiateAttention, self).__init__()
        
        self.att_dim = att_dim
        self.att_type = att_type
        self.num_heads = num_heads # Default : 1

        if self.att_type =='dot':
            self.att_blocks = nn.ModuleList([DotAttentionBlock(att_dim) for _ in range(num_heads)])
        if self.att_type =='BiP':
            self.att_blocks = nn.ModuleList([BilinearPoolingAttentionBlock(att_dim) for _ in range(num_heads)])

    def forward(self, global_input_feats, closest_normal_feats):
        """
        input : global_input_feats ([B, hid_dim]), cloasest_normal_feats ([B, n, hid_dim]
        output : diff_att_feats ([B, 1+n, hid_dim]) """


        common_feats = torch.cat([global_input_feats.unsqueeze(1), closest_normal_feats], axis=1) # [B, n+1, hid_dim]

        common_att_feats=[]


        for idx in range(self.num_heads): # default : 1 
            # if self.att_type =='dot':
            common_att_feat = self.att_blocks[idx](common_feats, common_feats) # [B, n+1, hid_dim]
            # print('common_att_feat.shape', common_att_feat.shape)
            # print('closest_normal_feat in AggregatedAttention.forward ', closest_normal_feat.shape)
            common_att_feats.append(common_att_feat.unsqueeze(0))

        common_att_feats = torch.cat(common_att_feats) # [1, B, n+1, hid_dim] (n=6)
        # print('common_att_feats.shape', common_att_feats.shape)

        common_att_feats = common_att_feats.permute(1,0,2,3) # [B, 1, n+1, hid_dim]  (1 : DA의 head 개수, n : AA의 head 개수)
        # print('Final common_att_feats.shape', common_att_feats.shape)
        return common_att_feats

class DotAttentionBlock(nn.Module):
   
    def __init__(self, hid_dim):
        super(DotAttentionBlock, self).__init__()
        self.hid_dim = hid_dim
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).cuda()
        self.proj_input = nn.Linear(in_features=hid_dim, out_features = hid_dim)
        self.proj_normal = nn.Linear(in_features=hid_dim, out_features=hid_dim)

    def forward(self, global_input_feats, global_normal_feats):
        """
        input : global_input_feats([B, 1 hid_dim]), global_normal_feats([B, N, hid_dim])
        output : closeset_normal_feat([B, 1, hid_dim])
        """

        # for key,value in self.proj_input.named_parameters():
        #     print(key, value.requires_grad, value.device)

        # print('glo device', global_input_feats.device)

        Q = self.proj_input(global_input_feats) # [B, 1, hid_dim]
        K = self.proj_normal(global_normal_feats) # [B, N, hid_dim]

        # print('Q', Q.shape)
        # print('K', K.shape)
        # print('K.permute(0,2,1)', K.permute(0,2,1).shape)
        
        # # cuda or gpu
        # if self.scale.device != global_input_feats.device:
        #    self.scale = self.scale.to(global_input_feats.device)

        # Attention Value
        M = torch.matmul(Q, K.permute(0,2,1))/self.scale # [B, 1, N] 
        # print('M', M.shape)

        # Attention map
        # print('M', M)
        # print('M', M)
        attention = torch.softmax(M, dim=-1) # [B, 1, N}
        # print('Attention', attention)
        # print('attention', attention.shape)
        

        closest_normal_feats = torch.matmul(attention, global_normal_feats) # [B, 1, hid_dim] (=[B, 1, N] * [B, N, hid_dim])
        
        
        return closest_normal_feats
        


class BilinearPoolingAttentionBlock(nn.Module):
    
    def __init__(self, hid_dim):

        super(BilinearPoolingAttentionBlock, self).__init__()

        self.hid_dim = hid_dim
        squeeze_dim = int(hid_dim/2)
        self.squeeze_dim = squeeze_dim

        # self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])) #.cuda()

        self.proj_input_key = nn.Linear(in_features=hid_dim, out_features = hid_dim)
        self.proj_normal_key = nn.Linear(in_features=hid_dim, out_features= hid_dim)
        self.proj_input_value = nn.Linear(in_features=hid_dim, out_features = hid_dim)
        self.proj_normal_value = nn.Linear(in_features=hid_dim, out_features= hid_dim)
        
        self.embed1 = nn.Linear(in_features =  hid_dim, out_features = squeeze_dim) # : self.squeeze
        self.embed2 = nn.Linear(in_features = squeeze_dim, out_features = 1)
        self.excitation = nn.Linear(in_features = squeeze_dim, out_features = hid_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, global_input_feats, global_normal_feats):
        """
        input : global_input_feats([B, 1 hid_dim]), global_normal_feats([B, N, hid_dim])
        output : closeset_normal_feat([B, 1, hid_dim])
        """
        B, N, hid_dim = global_normal_feats.shape

        # Query - Key Bilinear Pooling
        Q_k = self.proj_input_key(global_input_feats) # [B, 1, hid_dim]
        K = self.proj_normal_key(global_normal_feats) # [B, N, hid_dim]
        
        ## B_k = [B, N, hid_dim] * [B, N, hid_dim]
        B_k = self.sigmoid(Q_k.expand(-1, N, -1)) * self.sigmoid(K) # expand는 생략해도 무방하나, 직관성을 위해 표기 
        
        # print('Q_k', Q_k.shape) # [B, 1, hid_dim]
        # print('K', K.shape) # [B, N, hid_dim]
        # print('B_k', B_k.shape) # [B, N, hid_dim]

        # embed 1 (squeeze)
        B_k_prime = self.relu(self.embed1(B_k)) # [B, N, hid_dim/2]
        # print('B_k_prime', B_k_prime.shape) 
        
        # spatial attention (beta_s)
        b_s = self.embed2(B_k_prime) # [B, N, 1]
        # print('b_s', b_s.shape)

        beta_s = b_s.softmax(dim=1) # [B, N, 1]
        # print('beta_s', beta_s.shape)

        # channel-wise attention (excitation) (beta_c)

        B_bar = B_k_prime.mean(dim=1) # [B, hid_dim/2]
        # print('B_bar', B_bar.shape)

        b_c = self.excitation(B_bar) # [B, hid_dim]
        beta_c = self.sigmoid(b_c) # [B, hid_dim]
        # print('beta_c', beta_c.shape)

        

        # Query - Value Bilinear Pooling

        Q_v = self.proj_input_value(global_input_feats) # [B, 1, hid_dim] 
        V = self.proj_normal_value(global_normal_feats) # [B, N, hid_dim]

        B_v = self.relu(Q_v.expand((-1, N, -1))) * self.relu(V)   # expand : [B, 1, hid_dim] -> [B, N, hid_dim]
        
        # print('Q_v', Q_v.shape) # [B, 1, hid_dim]
        # print('V', V.shape) # [B, N, hid_dim]
        # print('B_v', B_v.shape) # [B, N, hid_dim]
        
        
        # spatial-attended value (논문 내 식 (6))
        att_v=(B_v*beta_s).sum(dim=1) # [B, hid_dim]
        # print('att_v', att_v.shape) # [B, hid_dim]
        
        ## 아래 식으로 해도 상관은 없다.
        ## Att_v = torch.matmul(beta_s.permute(0,2,1), B_v).squeeze(1)

        v_hat=beta_c * att_v # [B, hid_dim]
        # print('v_hat', v_hat.shape) # [B, hid_dim]
        
        v_hat = v_hat.unsqueeze(1) # [B, 1, hid_dim]
        
        # print('v_hat', v_hat.shape) # [B, hid_dim]
        
           
        return v_hat

