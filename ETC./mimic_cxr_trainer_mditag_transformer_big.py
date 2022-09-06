import time
import pickle
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.mditag_transformer_models_big import *
from utils.mimic_dataset_transformer import *
from utils.loss import *
#from utils.logger import Logger
from datetime import date, datetime

import random
import numpy as np
import csv, codecs
#from pycocoevalcap_original.eval_val import calculate_metrics
import json
from tqdm import tqdm
from utils.build_tag import *

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

class DebuggerBase:
    def __init__(self, args):
        self.args = args
        self.max_val_scores = 0
        #self.min_val_loss = 10000000000
        #self.min_val_tag_loss = 1000000
        #self.min_val_stop_loss = 1000000
        #self.min_val_word_loss = 10000000

        self.min_train_loss = 10000000000
        self.min_train_tag_loss = 1000000
        self.min_train_stop_loss = 1000000
        self.min_train_word_loss = 10000000

        self.params = None

        self._init_model_path()
        self.model_dir = self._init_model_dir()
        #self.writer = self._init_writer()
        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.normal_transform = self._init_normal_transform()
        self.vocab = self._init_vocab()
        self.model_state_dict = self._load_mode_state_dict()

        self.train_data_loader = self._init_data_loader(self.args.train_file_list, self.train_transform)
        self.val_data_loader = self._init_test_data_loader(self.args.val_file_list, self.val_transform)
        self.normal_data_loader = self._init_data_loader(self.args.normal_file_list, self.normal_transform)
        self.val_normal_data_loader = self._init_test_data_loader(self.args.normal_file_list, self.normal_transform)

        self.extractor = self._init_visual_extractor()
        #self.mlc = self._init_mlc()
        #self.co_attention = self._init_co_attention()
        self.transformer = self._init_transformer_model()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.tagger = self.__init_tagger()
        #self.logger = self._init_logger()
        print("{}\n".format(self.args))

    def train(self):
        with codecs.open(self.model_dir+"/loss.csv", "w", "utf-8") as fp:
            
            writer = csv.writer(fp, delimiter=",", quotechar='"')
            #writer.writerow(["epoch", "train tag loss", "train word loss", "train loss",\
            #                "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "cider"])
            writer.writerow(["epoch", "train word loss", "train loss",\
                             "bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l", "cider"])
            
            for epoch_id in range(self.start_epoch, self.args.epochs):
                train_word_loss, train_loss = self._epoch_train()
                metrics = self._epoch_val(epoch_id)
        
                if self.args.mode == 'train':
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step(val_loss)
                print(
                    "[{} - Epoch {}] metrics:{} - train loss:{} - lr:{}\n".format(self._get_now(),
                                                                     epoch_id,
                                                                     metrics,
                                                                     train_loss,
                                                                     self.optimizer.param_groups[0]['lr']))
                self._save_model(epoch_id,
                                 metrics,
                                 train_loss)
                """self._log(train_tags_loss=train_tag_loss,
                          train_stop_loss=train_stop_loss,
                          train_word_loss=train_word_loss,
                          train_loss=train_loss,
                          val_tags_loss=val_tag_loss,
                          val_stop_loss=val_stop_loss,
                          val_word_loss=val_word_loss,
                          val_loss=val_loss,
                          lr=self.optimizer.param_groups[0]['lr'],
                          epoch=epoch_id)"""
                
                epoch_all = [epoch_id, train_word_loss.item(), train_loss.item()]
                epoch_val_scores = [metrics['Bleu_1'], metrics['Bleu_2'], metrics['Bleu_3'], metrics['Bleu_4'], metrics['METEOR'], metrics['ROUGE_L'], metrics['CIDEr']]
                epoch_all.extend(epoch_val_scores)
                writer.writerow(epoch_all)

                
    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self, epoch_id):
        raise NotImplementedError

    def _init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
            #transforms.Normalize((0.5, 0.5, 0.5),
            #                     (0.5, 0.5, 0.5))])
        return transform

    def _init_val_transform(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.crop_size, self.args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
            #transforms.Normalize((0.5, 0.5, 0.5),
            #                     (0.5, 0.5, 0.5))])
        return transform

    def _init_normal_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
            #transforms.Normalize((0.5, 0.5, 0.5),
            #                     (0.5, 0.5, 0.5))])
        return transform

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.model_path, self.args.saved_model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_dir = os.path.join(model_dir, str(date.today()))  # ./report_v4_models/v4\2019-10-17
        #model_dir = os.path.join(model_dir, self._get_now())   # _get_now(): 날짜랑 시간 가져옴 / 경로 error나서 위 코드로 바꿔줌

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        return model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        print("Vocab Size:{}\n".format(len(vocab)))
        #print(vocab.get_word_by_id(0))
        #print(vocab.get_word_by_id(1))
        #print(vocab.get_word_by_id(2))
        #print(vocab.get_word_by_id(59))
        return vocab

    def _load_mode_state_dict(self):
        self.start_epoch = 0
        try:
            model_state = torch.load(self.args.load_model_path)
            self.start_epoch = model_state['epoch']
            print("[Load Model-{} Succeed!]\n".format(self.args.load_model_path))
            print("Load From Epoch {}\n".format(model_state['epoch']))
            return model_state
        except Exception as err:
            print("[Load Model Failed] {}\n".format(err))
            return None

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor(model_name=self.args.visual_model_name,
                                       pretrained=self.args.pretrained)

        try:
            model_state = torch.load(self.args.load_visual_model_path)
            model.load_state_dict(model_state['extractor'])
            print("[Load Visual Extractor Succeed!]\n")
        except Exception as err:
            print("[Load Model Failed] {}\n".format(err))

        if not self.args.visual_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.to(torch.device('cuda:0'))

        return model

    """def _init_mlc(self):
        model = MLC(classes=self.args.classes,
                    sementic_features_dim=self.args.sementic_features_dim,
                    fc_in_features=self.extractor.out_features,
                    k=self.args.k)

        try:
            model_state = torch.load(self.args.load_mlc_model_path)
            model.load_state_dict(model_state['model'])
            print("[Load MLC Succeed!]\n")
        except Exception as err:
            print("[Load MLC Failed {}!]\n".format(err))

        if not self.args.mlc_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.to(torch.device('cuda:0'))
        return model"""

    """def _init_co_attention(self):
        model = CoAttention(version=self.args.attention_version,
                            embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.extractor.out_features,
                            k=self.args.k,
                            momentum=self.args.momentum)

        try:
            model_state = torch.load(self.args.load_co_model_path)
            model.load_state_dict(model_state['model'])
            print("[Load Co-attention Succeed!]\n")
        except Exception as err:
            print("[Load Co-attention Failed {}!]\n".format(err))

        if not self.args.co_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.to(torch.device('cuda:0'))
        return model"""

    def _init_transformer_model(self):
        raise NotImplementedError

    def _init_word_model(self):
        raise NotImplementedError

    def _init_data_loader(self, file_list, transform):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        return data_loader
    
    def _init_test_data_loader(self, file_list, transform):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=transform,
                                 batch_size=self.args.test_batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        return data_loader

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()
        #return nn.SmoothL1Loss()

    def _init_optimizer(self):
        return torch.optim.Adam(params=self.params, lr=self.args.learning_rate, betas=(0.9,0.98))

    """def _log(self,
             train_tags_loss,
             train_stop_loss,
             train_word_loss,
             train_loss,
             val_tags_loss,
             val_stop_loss,
             val_word_loss,
             val_loss,
             lr,
             epoch):
        info = {
            'train tags loss': train_tags_loss,
            'train stop loss': train_stop_loss,
            'train word loss': train_word_loss,
            'train loss': train_loss,
            'val tags loss': val_tags_loss,
            'val stop loss': val_stop_loss,
            'val word loss': val_word_loss,
            'val loss': val_loss,
            'learning rate': lr
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)"""

    """def _init_logger(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger"""

    """def _init_writer(self):
        writer = open(os.path.join(self.model_dir, 'logs.txt'), 'w')
        return writer"""

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.to(torch.device('cuda:0'))
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M', time.gmtime()))

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.args.patience, factor=0.1)
        return scheduler

    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

#     def _init_log_path(self):
#         if not os.path.exists(self.args.log_path):
#             os.makedirs(self.args.log_path)
            
    def _save_model(self,
                    epoch_id,
                    metrics,
                    train_loss):
        def save_whole_model(_filename):
            print("Saved Model in {}\n".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        #'mlc': self.mlc.state_dict(),
                        #'co_attention': self.co_attention.state_dict(),
                        'transformer': self.transformer.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        def save_part_model(_filename, value):
            print("Saved Model in {}\n".format(_filename))
            torch.save({"model": value},
                       os.path.join(self.model_dir, "{}".format(_filename)))
        
        val_bleu = metrics['Bleu_1']+metrics['Bleu_2']+metrics['Bleu_3']+metrics['Bleu_4']/4
        val_meteor = metrics['METEOR']
        val_rouge = metrics['ROUGE_L']
        val_cider = metrics['CIDEr']
        val_scores = val_bleu + val_meteor + val_rouge + val_cider
        
        if val_scores > self.max_val_scores:
            file_name = "val_best{}_loss.pth.tar".format(epoch_id)
            save_whole_model(file_name)
            self.max_val_scores = val_scores
            """if not os.path.exists(result_path):
                os.makedirs(result_path)
            with open(os.path.join(result_path, '{}.json'.format("val_best_"+str(epoch_id)+".pth.tar")), 'w') as f:
                json.dump(result, f)"""
            

        if True:
            file_name = "train_best{}_loss.pth.tar".format(epoch_id)
            save_whole_model(file_name)
            self.min_train_loss = train_loss

        # if val_tag_loss < self.min_val_tag_loss:
        #     save_part_model("extractor.pth.tar", self.extractor.state_dict())
        #     save_part_model("mlc.pth.tar", self.mlc.state_dict())
        #     self.min_val_tag_loss = val_tag_loss
        #
        # if val_stop_loss < self.min_val_stop_loss:
        #     save_part_model("sentence.pth.tar", self.sentence_model.state_dict())
        #     self.min_val_stop_loss = val_stop_loss
        #
        # if val_word_loss < self.min_val_word_loss:
        #     save_part_model("word.pth.tar", self.word_model.state_dict())
        #     self.min_val_word_loss = val_word_loss
            
    def __init_tagger(self):
        return Tag()


class LSTMDebugger(DebuggerBase):
    def _init_(self, args):
        DebuggerBase.__init__(self, args)
        self.args = args

    def _epoch_train(self):
        tag_loss, word_loss, loss = 0, 0, 0
        self.extractor.train()
        #self.mlc.train()
        #self.co_attention.train()
        self.transformer.train()
        self.criterion1 = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        #total_num = 0
        
        for i, (images, _, label, captions, prob) in enumerate(self.train_data_loader):
            batch_tag_loss, batch_word_loss, batch_loss = 0, 0, 0
            images = self._to_var(images)
            #print("label:", label)
            if list(images.size())[0] == self.args.batch_size:                      # 
                conv5_fc_features, conv4_fc_features, conv3_fc_features = self.extractor.forward(images)      # torch.Size([14*14, B, 512])) / torch.Size([B, 2048])

                # normal image feature vector
                for j, (norm_images, _, norm_label, norm_captions, norm_prob) in enumerate(self.normal_data_loader):
                    # image : torch.Size([1, 3, 224, 224]), label : torch.Size([1, 210]), captions: (1,5,10), prob: (1,5)
                    norm_images = self._to_var(norm_images)
                    conv5_norm_features, conv4_norm_features, conv3_norm_features = self.extractor.forward(norm_images)  # torch.Size([1, 2048, 7, 7]), torch.Size([2048])

                # (patient - normal) image feature vector, mode: subtraction
                if args.feature_difference_mode == "subtraction":
                    conv5_diff_features = conv5_fc_features - conv5_norm_features   # ([7*7, B, 512])
                    conv4_diff_features = conv4_fc_features - conv4_norm_features   # ([14*14, B, 512])
                    conv3_diff_features = conv3_fc_features - conv3_norm_features   # ([28*28, B, 512])
                    #conv2_diff_features = conv2_fc_features - conv2_norm_features   # ([56*56, B, 512])
                elif args.feature_difference_mode == "addition":
                    conv5_diff_features = conv5_fc_features + conv5_norm_features 
                    conv4_diff_features = conv4_fc_features + conv4_norm_features  
                    conv3_diff_features = conv3_fc_features + conv3_norm_features  
                    #conv2_diff_features = conv2_fc_features + conv2_norm_features   
                elif args.feature_difference_mode == "multiplication":
                    conv5_diff_features = conv5_fc_features * conv5_norm_features  
                    conv4_diff_features = conv4_fc_features * conv4_norm_features   
                    conv3_diff_features = conv3_fc_features * conv3_norm_features 
                    #conv2_diff_features = conv2_fc_features * conv2_norm_features 
                else: print("Stop")
                """norm = sub_avg_features.norm(p=2, dim=1, keepdim=True) # [70, 1]
                norm_sub_avg_features = sub_avg_features.div(norm) # [70, 2048]"""
                
                #avg_features = torch.cat((avg_features, sub_avg_features), dim=1)  # torch.Size([16, 4096])
                #avg_features = sub_avg_features

                #tags, semantic_features = self.mlc.forward(diff_avg_features)            # torch.Size([16, 210]) / torch.Size([16, 10, 512])
                #print("tags:", tags) # [32, 14]
                #print("label:", label) # [32, 14]

                #batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()
                #print("batch_tag_loss:", batch_tag_loss)
                #sentence_states = None
                prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))
                context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
                prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)
                
                #ctx, _, _ = self.co_attention.forward(diff_avg_features,
                #                                      semantic_features,
                #                                      prev_hidden_states)  #[B, 512]
                
                """#print("diff_features:", diff_features.shape) # ([B, 2048, 7, 7])
                #self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14)) # why 14?
                #diff_features = self.adaptive_pool(diff_features)
                #print("diff_features:", diff_features.shape) # ([B, 2048, 14, 14])
                #diff_features = torch.flatten(diff_features, 2, 3)
                #print("diff_features:", diff_features.shape) # ([B, 2048, 14*14])
                #diff_features = diff_features.permute(2,0,1)
                #print("diff_features:", diff_features.shape) # ([14*14, B, 2048])"""
                
                
                #print("context:", context) # [B, max_word_num+1] with start token
                #print("context[:,:-1]:", context[:,:-1]) # [B, max_word_num] 
                                
                #print("diff_features:", diff_features.shape) # [14*14, B, 512]
                #print("diff_features.permute(1,0,2):", diff_features.permute(1,0,2).shape) #[B,14*14,512]
                # put context vector into model
                #ctx = ctx.unsqueeze(1) # [B, 1, 512]
                #total_features = torch.cat([diff_features.permute(1,0,2), ctx], 1) #[B,14*14+1,512]
                
                """conv5_diff_features = conv5_diff_features.unsqueeze(0)
                conv4_diff_features = conv4_diff_features.unsqueeze(0)
                conv3_diff_features = conv3_diff_features.unsqueeze(0)
                #conv2_diff_features = conv2_diff_features.unsqueeze(0)"""
                diff_features = torch.cat([conv5_diff_features, conv4_diff_features, conv3_diff_features], 0)  # [4165,B,512]
                total_features = diff_features.permute(1,0,2) #[B,4156,512]
                
                """del diff_features, conv5_diff_features, conv4_diff_features, conv3_diff_features, conv2_diff_features, conv5_fc_features, conv4_fc_features, conv3_fc_features, conv2_fc_features, conv5_norm_features, conv4_norm_features, conv3_norm_features, conv2_norm_features, images, norm_images, norm_label, norm_captions, norm_prob
                torch.cuda.empty_cache()"""
                
                logits = []
                
                logits,(enc_att,self_att) = self.transformer(context[:,:-1], total_features)   # [B, max_word_num, 1808]
                
                #print("train_logits.shape", logits.shape) # [B,92(max_word_num),1808] [B,118,1808]
                logits = logits.permute(1,0,2) # [92, B, 1808] [118, B, 1808]
                
                #print("context.T.shape:", context.T.shape) # [max_word_num+1, B] [93, B] [118, B]
                #print("context:", context)
                #print("context[0]:", context[0])
                context = context.T[1:]                    # remove start token
                #print("context.shape:", context.shape)     # [max_word_num, B] [92, B] [118]
                #print("context:", context)
                
                mask = context == self.vocab('<pad>') # don't need to do this. already have pad
                #print("mask:", mask.shape)
                #print("mask:", mask)
                
                mask = mask & torch.cat([mask[:1], mask[:-1]], 0) # don't need to do this.
                #print("mask:", mask.shape)
                #print("mask:", mask)
                
                #context = context.masked_fill(mask, -1)
                
                #print("context.shape:", context.shape) # [max_word_num, B]
                """for i in range(context.shape[0]):
                    print("context[i]:", context[i])"""
                
                #print("context:", context)
                #print("torch.flatten(logits, 0, 1):", torch.flatten(logits, 0, 1)) #[736(max_word_num*B), 1808] [944,1808]
                #print("torch.flatten(context):", torch.flatten(context)) #[736] [944]
                if self.args.smoothing:
                    eps = self.args.Lepsilon
                    context = context.masked_fill(mask,0) #just to make the scatter work so no indexing issue occurs
                    gold = context.contiguous().view(-1)
                    
                    logits = torch.flatten(logits, 0, 1)
                    n_class = logits.shape[-1]
                    one_hot = torch.zeros_like(logits, device=logits.device).scatter(1, gold.view(-1, 1), 1)
                    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
                    log_prb = torch.log_softmax(logits, dim=1)
                    
                    context = context.masked_fill(mask, self.vocab('<pad>'))
                                                  
                    gold = context.contiguous().view(-1)
                    non_pad_mask = gold.ne(-1)

                    batch_word_loss = -(one_hot * log_prb).sum(dim=1)
                    batch_word_loss = batch_word_loss.masked_select(non_pad_mask).sum()  # average later

                    del gold, log_prb, non_pad_mask
                    torch.cuda.empty_cache()
                                                  
                else:
                    batch_word_loss = self.criterion1(torch.flatten(logits, 0, 1), torch.flatten(context))
                #print("batch_word_loss:", batch_word_loss)
                
                
                
                batch_loss = batch_word_loss #self.args.lambda_word * batch_word_loss
                #print("batch_loss:", batch_loss)

                self.optimizer.zero_grad()
                batch_loss.backward()
                if self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm(self.transformer.parameters(), self.args.clip)
                self.optimizer.step()

                #tag_loss += self.args.lambda_tag * batch_tag_loss.data
                word_loss += batch_word_loss.data #self.args.lambda_word * batch_word_loss.data
                #print("tag_loss:", tag_loss)
                #print("word_loss:", word_loss)
                loss += batch_loss.data
                
                #total_num += self.args.batch_size
                #print("tag_loss, stop_loss, word_loss, loss:", tag_loss, stop_loss, word_loss, loss)

        # epoch_loss = [i, tag_loss.item(), stop_loss.item(), word_loss.item(), loss.item()]

        return word_loss, loss # tag_loss

    def _epoch_val(self, epoch_id):
        self.extractor.eval()
        #self.mlc.eval()
        #self.co_attention.eval()
        self.transformer.eval()
        
        progress_bar = tqdm(self.val_data_loader, desc='Generating')
        #results = {}
        #datasetGTS = {'annotations': []}    # real sentence
        #datasetRES = {'annotations': []}    # pred sentence
        #gts = {}
        #res = {}
        #image_id_list = []
        real_captions = []
        predictions = []
        
        with torch.no_grad():
            #torch.cuda.empty_cache()
            for images, image_id, label, captions, _ in progress_bar:
                references = []
                hypotheses = []
                images = self._to_var(images, requires_grad=False)

                conv5_fc_features, conv4_fc_features, conv3_fc_features = self.extractor.forward(images)
                # normal image feature vector
                for j, (norm_images, _, norm_label, norm_captions, norm_prob) in enumerate(self.val_normal_data_loader):
                    # image : torch.Size([1, 3, 224, 224]), label : torch.Size([1, 210]), captions: (1,5,10), prob: (1,5)
                    norm_images = self._to_var(norm_images)
                    conv5_norm_features, conv4_norm_features, conv3_norm_features = self.extractor.forward(norm_images)  # torch.Size([1, 2048, 7, 7]), torch.Size([2048])

                # (patient - normal) image feature vector, mode: subtraction
                if args.feature_difference_mode == "subtraction":
                    conv5_diff_features = conv5_fc_features - conv5_norm_features   # ([7*7, B, 512])
                    conv4_diff_features = conv4_fc_features - conv4_norm_features   # ([14*14, B, 512])
                    conv3_diff_features = conv3_fc_features - conv3_norm_features   # ([28*28, B, 512])
                    #conv2_diff_features = conv2_fc_features - conv2_norm_features   # ([56*56, B, 512])
                elif args.feature_difference_mode == "addition":
                    conv5_diff_features = conv5_fc_features + conv5_norm_features 
                    conv4_diff_features = conv4_fc_features + conv4_norm_features  
                    conv3_diff_features = conv3_fc_features + conv3_norm_features  
                    #conv2_diff_features = conv2_fc_features + conv2_norm_features   
                elif args.feature_difference_mode == "multiplication":
                    conv5_diff_features = conv5_fc_features * conv5_norm_features  
                    conv4_diff_features = conv4_fc_features * conv4_norm_features   
                    conv3_diff_features = conv3_fc_features * conv3_norm_features 
                    #conv2_diff_features = conv2_fc_features * conv2_norm_features 
                else: print("Stop")

                '''#normalize 하는게 좋을지 안하는게 좋을지 고민
                norm = sub_avg_features.norm(p=2, dim=1, keepdim=True)
                norm_sub_avg_features = sub_avg_features.div(norm)
                print(norm_sub_avg_features)'''

                #tags, semantic_features = self.mlc.forward(diff_avg_features)

                #prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, self.args.hidden_size))

                #ctx, v_att, a_att = self.co_attention.forward(diff_avg_features,
                #                                              semantic_features,
                #                                              prev_hidden_states)

                # put context vector into model
                #ctx = ctx.unsqueeze(1) # [B, 1, 512]
                #total_features = torch.cat([diff_features.permute(1,0,2), ctx], 1) #[B,14*14+1,512]
                """conv5_diff_features = conv5_diff_features.unsqueeze(0)
                conv4_diff_features = conv4_diff_features.unsqueeze(0)
                conv3_diff_features = conv3_diff_features.unsqueeze(0)
                #conv2_diff_features = conv2_diff_features.unsqueeze(0)"""
                diff_features = torch.cat([conv5_diff_features, conv4_diff_features, conv3_diff_features], 0)  # [4165,B,512]
                total_features = diff_features # [4165, B, 512]
                del diff_features, conv5_diff_features, conv4_diff_features, conv3_diff_features, conv5_fc_features, conv4_fc_features, conv3_fc_features, conv5_norm_features, conv4_norm_features, conv3_norm_features, images, norm_images, norm_label, norm_captions, norm_prob
                torch.cuda.empty_cache()

                # beam search
                # total_features, max_T=100, on_max='halt'
                beam_width = 4
                on_max = 'halt'
                max_T = self.args.n_max
                
                #print("total_features.shape:", total_features.shape) # [196, B, 512]
                random_placeholder = torch.randn(total_features.shape[1], self.args.decoder_hidden_size, device=total_features.device) # self.args???

                logpb_tm1 = torch.where(
                    torch.arange(beam_width, device=total_features.device) > 0,  # K
                    torch.full_like(
                        random_placeholder[..., 0].unsqueeze(1), -float('inf')),  # k > 0
                    torch.zeros_like(
                        random_placeholder[..., 0].unsqueeze(1)),  # k == 0
                )  # (N, K)

                assert torch.all(logpb_tm1[:, 0] == 0.)
                assert torch.all(logpb_tm1[:, 1:] == -float('inf'))

                b_tm1_1 = torch.full_like(  # (t, N, K)
                logpb_tm1, self.vocab('<start>'), dtype=torch.long).unsqueeze(0)

                #print("b_tm1_1.shape:", b_tm1_1.shape) # [1,196,4]
                #print("b_tm1_1:", b_tm1_1) # all 3

                # We treat each beam within the batch as just another batch when
                # computing logits, then recover the original batch dimension by
                # reshaping
                #print("total_features.shape:", total_features.shape) # [196, B, 512]
                
                total_features = total_features.unsqueeze(2).repeat(1, 1, beam_width, 1) # [196,B,4,512]
                #print("total_features.shape:", total_features.shape)
                total_features = total_features.flatten(1, 2)  # (S, N * K, L) # [196,B*4, 512]
                #print("total_features.shape:", total_features.shape)
                v_is_eos = torch.arange(len(self.vocab), device=total_features.device)
                #print("v_is_eos:", v_is_eos) # [1808] -> [0,1,2,3....,1807]
                v_is_eos = v_is_eos == self.vocab('<eod>')  # (V,)
                #print("v_is_eos.shape:", v_is_eos.shape) # [1808] -> [False,False,True,...,False]
                #print("v_is_eos:", v_is_eos)
                t = 0
                logits_tm1 = None
                cur_transformer_ip = None

                while torch.any(b_tm1_1[-1, :, 0] != self.vocab('<eod>')):
                    #print("b_tm1_1[-1, :, 0]:", b_tm1_1[-1, :, 0])
                    if t == max_T:
                        print(f'Beam search not finished by t={t}. Halted')
                        break
                    finished = (b_tm1_1[-1] == self.vocab('<eod>'))
                    #print("finished:", finished)

                    E_tm1 = b_tm1_1[-1].flatten().unsqueeze(1)  # (N * K, 1)
                    #print("E_tm1.shape", E_tm1.shape) # [B*4, 1] all 3

                    if cur_transformer_ip == None:
                        cur_transformer_ip = E_tm1
                    else:
                        cur_transformer_ip = torch.cat([cur_transformer_ip, E_tm1], axis=1)
                    #del E_tm1
                    #torch.cuda.empty_cache()

                    #print("cur_transformer_ip.shape:", cur_transformer_ip.shape) #[B*4,1],[B*4,2],[196,3]
                    #print("total_features.permute(1,0,2).shape:", total_features.permute(1,0,2).shape) # [B*4, 196, 512]

                    op, _ = self.transformer(cur_transformer_ip, total_features.permute(1,0,2)) #.permute(1,0,2))
                    #print("op.shape:", op.shape) #[32(B*4),1,1808] [32,2,1808]
                    #print("op:", op)
                    logits_t = op[:, -1, :]
                    #print("logits_t.shape", logits_t.shape) #[32, 1808]
                    
                    logits_tm1 = logits_t
                    logits_t = logits_t.view(
                        -1, beam_width, len(self.vocab))  # (N, K, V)
                    logpy_t = nn.functional.log_softmax(logits_t, -1)

                    # We length-normalize the extensions of the unfinished paths
                    if t:
                        logpb_tm1 = torch.where(
                            finished, logpb_tm1, logpb_tm1 * (t / (t + 1)))
                        logpy_t = logpy_t / (t + 1)
                    # For any path that's finished:
                    # - v == <eos> gets log prob 0
                    # - v != <eos> gets log prob -inf
                    logpy_t = logpy_t.masked_fill(
                        finished.unsqueeze(-1) & v_is_eos, 0.)
                    logpy_t = logpy_t.masked_fill(
                        finished.unsqueeze(-1) & (~v_is_eos), -float('inf'))

                    # update_beam
                    V = logpy_t.shape[2] #Vocab size
                    K = logpy_t.shape[1] #Beam width

                    s = logpb_tm1.unsqueeze(-1).expand_as(logpy_t) + logpy_t
                    logy_flat = torch.flatten(s, 1, 2)
                    top_k_val, top_k_ind = torch.topk(logy_flat, K, dim = 1)
                    temp = top_k_ind // V #This tells us which beam that top value  is from
                    logpb_t = top_k_val

                    temp_ = temp.expand_as(b_tm1_1)
                    b_t_1 = torch.cat((torch.gather(b_tm1_1, 2, temp_), (top_k_ind % V).unsqueeze(0)))

                    del logits_t, logpy_t, finished
                    torch.cuda.empty_cache()
                    
                    logpb_tm1, b_tm1_1 = logpb_t, b_t_1
                    #print("logpb_tm1.shape:", logpb_tm1.shape) #[8,4]
                    #print("b_tm1_1.shape:", b_tm1_1.shape)     #[2,8,4] [3,8,4]
                    t += 1

                
                b_1 = b_tm1_1  # [501, B, beam_size]
                #print("b_1.shape:", b_1.shape)
                captions_cand = b_1[..., 0] # [501, B]
                #print("captions_cand.shape:", captions_cand.shape)

                cands = captions_cand.T # [B, 501]
                #print("cands.shape:", cands.shape)
                #print("cands:", cands)
                cands_list = cands.tolist()

                for i in range(len(cands_list)): #Removes sos, pad tags
                    #print("cands_list[i]:", cands_list[i])
                    cands_list[i] = list(filter((self.vocab('<start>')).__ne__, cands_list[i]))
                    cands_list[i] = list(filter((self.vocab('<pad>')).__ne__, cands_list[i]))
                    cands_list[i] = list(filter((self.vocab('<eod>')).__ne__, cands_list[i]))
                    cands_list[i] = list(filter((self.vocab('<eos>')).__ne__, cands_list[i]))
                    #print("cands_list[i]:", cands_list[i])
                    
                # hypotheses
                hypotheses += cands_list
                #print("len(hypotheses):", len(hypotheses)) # B
                #print("hypotheses:", hypotheses)
                
                # references
                #print("captions:", captions) # [B, max_word_num]
                #print("captions:", captions.shape)
                captions = captions.tolist()
                
                for i in range(len(captions)):
                    #print("captions[i]:", captions[i])
                    captions[i] = list(filter(float(self.vocab('<eod>')).__ne__, captions[i]))
                    captions[i] = list(filter(float(self.vocab('<eos>')).__ne__, captions[i]))
                    #print("captions[i]:", captions[i])
                
                captions_word = [] # <eos>, <eod> included, <pad> excluded
                for caption in captions:
                    #print("self.__vec2sent(caption):", self.__vec2sent(caption))
                    captions_word.append(self.__vec2sent(caption)) # .cpu().detach().numpy()
                #print("captions_word:", captions_word) # [B, variable length]
                #print("captions_word:", len(captions_word))
                
                references += captions_word
                real_captions += references
                
                for i in range(len(references)):
                    #hypotheses[i] = [self.vocab(token - 1) for token in hypotheses[i]]
                    #hypotheses[i] = " ".join(hypotheses[i])
                    
                    #hypotheses[i] = [self.vocab.get_word_by_id(word_id) for word_id in hypotheses[i]]
                    list_1 = []
                    for word_id in hypotheses[i]:
                        #print("self.vocab.get_word_by_id(word_id):", self.vocab.get_word_by_id(word_id))
                        list_1.append(self.vocab.get_word_by_id(word_id))
                        #print("list_1:", list_1)
                    hypotheses[i] = " ".join(list_1)
                    
                predictions += hypotheses
                
                assert(len(references) == len(hypotheses))
                assert(len(real_captions) == len(predictions))
                #print("references[0]:", references[0]) # eos, eod 포함 / start, pad 제외
                #print("hypotheses[0]:", hypotheses[0])
                
                """hypo = {idx: [h] for idx, h in enumerate(hypotheses)}
                ref = {idx: [" ".join(l) for l in r] for idx, r in enumerate(references)}
                print("hypo:", hypo)
                print("ref:", ref)"""
                # end of batch
                del random_placeholder, logpb_tm1, b_tm1_1, total_features, v_is_eos, logits_tm1, cur_transformer_ip, E_tm1, op
                torch.cuda.empty_cache()
                
        assert(len(real_captions) == len(predictions)) # 108
        hypo = {idx: [h] for idx, h in enumerate(predictions)}
        ref = {idx: [r] for idx, r in enumerate(real_captions)}

        #print("hypo:", hypo)
        #print("ref:", ref)
        metrics = self.score(ref, hypo)
        #print(metrics)
        del hypo, ref, predictions, real_captions, references, captions_cand, cands, cands_list, captions, captions_word
        torch.cuda.empty_cache()

        return metrics  #, results 

    def _init_transformer_model(self):        
        model = TransformerDecoder(vocab_size=len(self.vocab), 
                                   hidden_size=self.args.decoder_hidden_size, 
                                   num_layers=self.args.num_layers, 
                                   num_heads=self.args.num_heads, 
                                   dropout=self.args.dropout)

        try:
            model_state = torch.load(self.args.load_transformer_model_path)
            model.load_state_dict(model_state['transformer'])
            print("[Load Transformer Model Succeed!\n")
        except Exception as err:
            print("[Load Transformer Model Failed {}!]\n".format(err))

        if not self.args.transformer_trained:
            for i, param in enumerate(model.parameters()):
                param.requires_grad = False
        else:
            if self.params:
                self.params += list(model.parameters())
            else:
                self.params = list(model.parameters())

        if self.args.cuda:
            model = model.to(torch.device('cuda:0'))
        return model
        
    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>' or word == '<pad>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)
    
    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores


def seed_everything():
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    """
    Data Argument
    """
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--mode', type=str, default='train')

    # Path Argument
    parser.add_argument('--vocab_path', type=str, default='./data/new_data/vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='./data/images_frontal',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/new_data/captions.json',
                        help='path for captions')
    parser.add_argument('--train_file_list', type=str, default='./data/new_data/train_data_2.txt',
                        help='the train array')
    parser.add_argument('--val_file_list', type=str, default='./data/new_data/val_data_2.txt',
                        help='the val array')
    parser.add_argument('--normal_file_list', type=str, default='./data/new_data/normal_data.txt',
                        help='the normal array')

    # transforms argument
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    # Load/Save model argument
    parser.add_argument('--model_path', type=str, default='./report_v4_models/',
                        help='path for saving trained models')
    parser.add_argument('--load_model_path', type=str, default='',
                        help='The path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='v4',
                        help='The name of saved model')

    """
    Model Argument
    """
    parser.add_argument('--momentum', type=float, default=0.1)
    # VisualFeatureExtractor
    parser.add_argument('--visual_model_name', type=str, default='resnet152',
                        help='CNN model name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='not using pretrained model when training')
    parser.add_argument('--load_visual_model_path', type=str,
                        default='.')
    parser.add_argument('--visual_trained', action='store_true', default=True,
                        help='Whether train visual extractor or not')

    # MLC
    parser.add_argument('--classes', type=int, default=210)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--load_mlc_model_path', type=str,
                        default='.')
    parser.add_argument('--mlc_trained', action='store_true', default=True)

    # Co-Attention
    parser.add_argument('--attention_version', type=str, default='v4')
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--load_co_model_path', type=str, default='.')
    parser.add_argument('--co_trained', action='store_true', default=True)

    # Transformer
    parser.add_argument('--decoder_hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--load_transformer_model_path', type=str,
                        default='.')
    parser.add_argument('--transformer_trained', action='store_true', default=True)
    

    """
    Training Argument
    """
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: 0.35)')
    parser.add_argument('--s_max', type=int, default=6)
    parser.add_argument('--n_max', type=int, default=30)

    # Loss Function
    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)
    
    parser.add_argument('--seed', type=int, default=42)
    
    # Feature Difference Generation Method
    parser.add_argument('--feature_difference_mode', type=str, default='subtraction')
                                                  
    # Smoothing
    parser.add_argument('--smoothing', type=int, default=1)
    parser.add_argument('--Lepsilon', type=float, default=0.1)                           
    
    # Evaluate 
    parser.add_argument('--result_path', type=str,                             default='C:/Users/medinfo_phr/Anaconda3/envs/medinfo/Medical_Report_Generation/report_v4_models/v4/2020-12-04/results/test_best_loss_train_966.pth.tar.json')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print ('Available devices ', torch.cuda.device_count())
    torch.cuda.set_device('cuda:0')
    print ('Current cuda device ', torch.cuda.current_device())
    
    seed_everything()
    
    debugger = LSTMDebugger(args)
    debugger.train()
