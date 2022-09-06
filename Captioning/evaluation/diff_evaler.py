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
        att_feats,
        eval_annfile,
        all_file,
        start_index,
        normal_ids,
        norm_feats
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
        self.normal_ids = np.array(utils.load_ids(normal_ids, self.jpg_to_id))

        self.eval_loader = data_loader.load_val(eval_ids, gv_feat, att_feats)
        self.normal_loader = data_loader.load_normal_val(normal_ids, gv_feat, norm_feats)
        self.evaler = evaluation.create(cfg.INFERENCE.EVAL, eval_annfile)



    def make_kwargs(self, indices, ids, gv_feat, att_feats, att_mask):
        kwargs = {}
        kwargs[cfg.PARAM.INDICES] = indices
        kwargs[cfg.PARAM.GLOBAL_FEAT] = gv_feat
        kwargs[cfg.PARAM.ATT_FEATS] = att_feats
        kwargs[cfg.PARAM.ATT_FEATS_MASK] = att_mask
        kwargs['BEAM_SIZE'] = cfg.INFERENCE.BEAM_SIZE
        kwargs['GREEDY_DECODE'] = cfg.INFERENCE.GREEDY_DECODE
        return kwargs
        
    def __call__(self, model, rname):
        model.eval()
        
        results = []
        self.eval_ids = self.eval_ids_original - self.start_index # changed
        
        with torch.no_grad():
            for _, (indices, gv_feat, att_feats, att_mask) in tqdm.tqdm(enumerate(self.eval_loader)):
                for _, (_, _, norm_att_feats, _) in enumerate(self.normal_loader):
                    #print("indices:", indices)
                    #print("self.eval_ids:", self.eval_ids)
                    #print("self.eval_ids:", self.eval_ids)
                
                    ids = self.eval_ids[indices]
                    gv_feat = gv_feat.cuda()
                    att_mask = att_mask.cuda()
                    
                    att_feats = att_feats.numpy()
                    norm_att_feats = norm_att_feats.numpy()
                    
                    # difference feature vector
                    diff_att_feats = []
                    for att_feat in att_feats:
                        norm_att_feat = norm_att_feats[0]
                        att_len = att_feat.shape[0]
                        norm_att_len = norm_att_feat.shape[0]
                        
                        #print("norm_att_feat.shape:", norm_att_feat.shape)
                        #print("att_feat.shape:", att_feat.shape)
                        
                        if att_len > norm_att_len:
                            #print("before norm_att_feat.shape:", norm_att_feat.shape)
                            norm_att_feat = np.append(norm_att_feat, np.zeros((att_len-norm_att_len, att_feat.shape[1]),dtype=np.float32), axis=0)
                            #print("after norm_att_feat.shape:", norm_att_feat.shape)
                        elif att_len < norm_att_len:
                            #print("before att_feat.shape:", att_feat.shape)
                            att_feat = np.append(att_feat, np.zeros((norm_att_len-att_len, att_feat.shape[1])),dtype=np.float32, axis=0)
                            #print("after att_feat.shape:", att_feat.shape)
                            
                        # diff_att_feat = att_feat - norm_att_feat
                        diff_att_feat = att_feat * norm_att_feat
                        diff_att_feats.append(diff_att_feat)
                    
                    diff_att_feats = torch.from_numpy(np.array(diff_att_feats)).cuda()
                    
                    kwargs = self.make_kwargs(indices, ids, gv_feat, diff_att_feats, att_mask)
                    if kwargs['BEAM_SIZE'] > 1:
                        seq, _ = model.module.decode_beam(**kwargs)
                        #print("seq:", seq)
                    else:
                        seq, _ = model.module.decode(**kwargs)
                    sents = utils.decode_sequence(self.vocab, seq.data)
                    #print("sents:", sents)
                
                    ids = ids + self.start_index
                    for sid, sent in enumerate(sents):
                        #print("sent:", sent)
                        result = {cfg.INFERENCE.ID_KEY: int(ids[sid]), cfg.INFERENCE.CAP_KEY: sent}
                        #print("result:", result)
                        results.append(result)
                    
                    
        #print("results:", results)
        eval_res = self.evaler.eval(results)

        result_folder = os.path.join(cfg.ROOT_DIR, 'result')
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        json.dump(results, open(os.path.join(result_folder, 'result_' + rname +'.json'), 'w'))

        model.train()
        return eval_res
