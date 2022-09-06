import os
import sys
import numpy as np
import torch
import tqdm
import json
import evaluation
import lib.utils as utils
import datasets.data_loader as data_loader
from lib.config import cfg

class Evaler(object):
    def __init__(
        self,
        eval_ids,
        gv_feat,
        # att_feats,
        input_images, # jsp
        normal_images, # jsp
        eval_annfile,
        all_file,
        start_index,
        input_transform, # jsp
        normal_transform,  # jsp
        leverage_normal,
        num_ca
    ):
        super(Evaler, self).__init__()
        self.vocab = utils.load_vocab(cfg.INFERENCE.VOCAB)
        
        self.image_ids = utils.load_lines(all_file) # changed
        # changed
        self.jpg_to_id = {} 
        self.id_to_jpg = {}
        for i, jpg in enumerate(self.image_ids):
            self.jpg_to_id[jpg] = i
            self.id_to_jpg[i] = jpg

        self.eval_ids_original = np.array(utils.load_ids(eval_ids, self.jpg_to_id)) # changed
        self.start_index = start_index                                     # changed

        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, input_images, normal_images, 
        input_transform, normal_transform, leverage_normal, num_ca)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)


    # jsp : main_mimic.py와는 다르게 att를 포함해도 괜찮음.
    def make_kwargs(self, indices, ids, gv_feat, input_images, normal_images, att_feats, att_mask): #att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.INPUT_IMAGES] = input_images
        kwargs[cfg.PARAM.NORMAL_IMAGES] = normal_images
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs
        
    def __call__(self, model, rname):
        model.eval() #jsp model 변경 x
        
        results = []
        self.eval_ids = self.eval_ids_original - self.start_index # changed
        
        # jsp : model 변경
        with torch.no_grad():
            # for _, (indices, gv_feat, att_feats, att_mask) in tqdm.tqdm(enumerate(self.eval_loader)):
            for _, (indices, gv_feat, input_images, normal_images) in tqdm.tqdm(enumerate(self.eval_loader)):
                #print("indices:", indices)
                #print("self.eval_ids:", self.eval_ids)
                #print("self.eval_ids:", self.eval_ids)
                

                ###### jsp : Feature Extraction
                # att_feats을 encoder로부터 얻어야함. 

                # detach 안 해줘도 됨. 

                input_images = input_images.cuda()
                input_feats = model.module.encoder(input_images)

                if not cfg.MODEL.ENCODER_FUSION_MODE =='no':
                    normal_images = normal_images.cuda()

                    if 'CA' in cfg.MODEL.ENCODER_FUSION_MODE:
                
                        batch, N, _, _, _ = normal_images.shape

                        # Gradient는 기록 x
                        # 1장 했을 때는 detach만으로 충분하지만, 여러 장이기 때문에 Encoder에서 메모리 부하올 수 있음.

                        # with torch.no_grad() :         
                        # input : [B*10, h.no_grad(3, 448, 448]
                        # output : [196, B*10, 1024]
                        normal_feats = model.module.encoder((normal_images).flatten(0,1))  
                        seq_len, _, hid_dim = normal_feats.shape # [196, B*10, 1024]
                            
                        normal_feats = normal_feats.view(seq_len, batch, N, hid_dim) # [196, B, 10, 1024]

                        global_normal_feats = normal_feats.mean(axis=0) # [B,.10, 1024]
                        
                        global_normal_feats.requires_grad = True # Attention 부터는 Gradient 필요.
                    else:
                        
                        normal_feats = model.module.encoder(normal_images)

                # jsp : 1/21, input feats 얘네들 아마 cuda로 받는 게.. 
                # att_feats, att_mask = model.module.bridge(input_feats, normal_feats)

                if cfg.MODEL.ENCODER_TYPE =='resnet152':
                    if 'ewp':
                        diff_feats = []
                        for input_feat, normal_feat in zip(input_feats, normal_feats):
                            diff_feat = input_feat * normal_feat
                            diff_feats.append(diff_feat)
                        
                        diff_feats = torch.cat(diff_feats, 0) # [1029, B, 512]
                        att_feats = diff_feats.permute(1,0,2) # [B, 1029, 512]

                elif cfg.MODEL.ENCODER_TYPE =='densenet121':

                    if cfg.MODEL.ENCODER_FUSION_MODE == 'concat+ewp': # ewp + Concat
                        diff_feats = input_feats * normal_feats # [196, B, 512]
                        diff_feats = torch.cat([input_feats, diff_feats], axis=0) # [392, B, 512]
                        att_feats = diff_feats.permute(1,0,2) # [B, 392, 512]
                        # print(att_feats.shape)
                    elif cfg.MODEL.ENCODER_FUSION_MODE == 'ewp':
                        diff_feats = input_feats * normal_feats # [196, B, 512]
                        att_feats = diff_feats.permute(1,0,2) # [B, 196, 512]
                    elif cfg.MODEL.ENCODER_FUSION_MODE == 'BiP': # Bi-Linear Pooling + Concat

                        att_feats = '' # [B, 392, 512]
                    elif cfg.MODEL.ENCODER_FUSION_MODE =='no':
                        diff_feats = input_feats # normal image 사용 x
                        att_feats = diff_feats.permute(1,0,2) # [B, 196, 512]
                    elif cfg.MODEL.ENCODER_FUSION_MODE =='concat+CA':
                
                        contra_feats = model.module.contra_att(input_feats, global_normal_feats) # [196, B, 1024]
                        
                        att_feats = torch.cat([input_feats, contra_feats], axis = 0) # [392, B, 1024]
                        att_feats = att_feats.permute(1, 0, 2) # [B, 392, 1024]

                    if cfg.MODEL.ENCODER_FUSION_MODE =='CA':

                        contra_feats = model.module.contra_att(input_feats, global_normal_feats) # [196, B, 1024] (x4일 때 196->784)

                        att_feats = contra_feats.permute(1, 0, 2) # [B, 196, 1024]
                
                att_feats_copy = att_feats.data # jsp gpu 방지.
                # print('1', att_feats.shape)
                # print('2', att_feats_copy.shape)

                att_feats_copy, att_mask = model.module.get_attn_relation(att_feats_copy)

                
                #########################

                ids = self.eval_ids[indices]
                gv_feat = gv_feat.cuda()
                
                # att_feats = att_feats.cuda() 
                att_mask = att_mask.cuda()
                # input_images = input_images.cuda() # 없어도 되긴 하는데 언젠간 필요하겠거니.
                # normal_images = normal_images.cuda()  # jsp

                kwargs = self.make_kwargs(indices, ids, gv_feat, input_images, normal_images, att_feats, att_mask)

                if kwargs['BEAM_SIZE'] > 1:
                    # seq, _ = model.module.decode_beam(**kwargs)
                    seq, _ = model.module.transformer.decode_beam(**kwargs) # jsp
                    #print("seq:", seq)
                else:
                    # seq, _ = model.module.decode(**kwargs)
                    seq, _ = model.module.transformer.decode_beam(**kwargs) # jsp
                
                sents = utils.decode_sequence(self.vocab, seq.data)
                #print("sents:", sents)
                
                ids = ids + self.start_index
                for sid, sent in enumerate(sents):
                    
                    #print("sent:", sent)
                    result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                    #print("result:", result)
                    results.append(result)
                
        # print('---------------evaler-------------------')           
        # print('results :', results)  # 'image_id:' : 865, 'caption': 'the <eos> the <eos> no <eos> ... 
        #print("results:", results)
        eval_res = self.evaler.eval(results)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        model.train()
        return eval_res
