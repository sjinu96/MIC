import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms
import os,sys
import json

from easydict import EasyDict as edict

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from Chexpert.model.classifier import Classifier  # densenet-121 - Chexpert
from Chexpert.model.backbone import densenet # densenet-121 - Imagenet
from lib.utils import activation
from PIL import Image
import os

# from XTransformer.lib.config import cfg


# model_urls = {
#     'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'

class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='densenet121', pretrained=False, cfg=False) : # encoder_cfg_path=False, pretrained_path=False): 
        """cfg_path : Only in Densenet121
           pretrained_path : Only in Densenet121 using Chexpert pre-trained Weights"""
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.cfg = cfg
        
        
        if model_name =='densenet121':
            
            # print('cfg.ROOT_DIR', cfg.ROOT_DIR)
            # print('cfg.MODEL.ENCODER_CFG_PATH:', cfg.MODEL.ENCODER_CFG_PATH)
            
            self.encoder_cfg_path = os.path.join(cfg.ROOT_DIR,  cfg.MODEL.ENCODER_CFG_PATH)
            if cfg.MODEL.ENCODER_PRETRAINED_PATH is not False:
                print('Load Pre-Trained Encoder : ', cfg.MODEL.ENCODER_PRETRAINED_PATH)
                self.pretrained_path = os.path.join(cfg.ROOT_DIR, cfg.MODEL.ENCODER_PRETRAINED_PATH)


        self.activation = nn.ReLU()


        if self.model_name =='resnet152':
            self.conv5_model, self.conv5_out_features, self.conv5_avg_func, \
            self.conv4_out_features, self.conv4_avg_func, \
            self.conv3_out_features, self.conv3_avg_func = self.__get_model()
            
            ## Fully Connected Layer
            self.conv5_fc = nn.Linear(in_features=self.conv5_out_features, out_features=512) # 512 OOM
            self.conv4_fc = nn.Linear(in_features=self.conv4_out_features, out_features=512)
            self.conv3_fc = nn.Linear(in_features=self.conv3_out_features, out_features=512)

        if self.model_name =='densenet121':
            # print('Dense!!!!!')
            self.extractor = self.__get_model()



            sequential = []
            if cfg.MODEL.ENCODER_PROJ =='1x1conv':
                final_proj = nn.Conv2d(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM, kernel_size=1) # 1024 : Final features in Densenet Block 4, 우선 상수로 고정.
            elif cfg.MODEL.ENCODER_PROJ =='fc':
                # print(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM)
                final_proj = nn.Linear(cfg.MODEL.ATT_FEATS_DIM, cfg.MODEL.ATT_FEATS_EMBED_DIM, )
            elif cfg.MODEL.ENCODER_PROJ =='no':
                pass
            
            sequential.append(final_proj)
            sequential.append(activation(cfg.MODEL.ATT_FEATS_EMBED_ACT))
            
            if cfg.MODEL.ATT_FEATS_NORM == True: # 거의 True
                sequential.append(nn.LayerNorm(cfg.MODEL.ATT_FEATS_EMBED_DIM))
            
            if cfg.MODEL.DROPOUT_ATT_EMBED > 0: # Default : 0.3
                sequential.append(nn.Dropout(cfg.MODEL.DROPOUT_CA)) 
               
            self.att_embed = nn.Sequential(*sequential) if len(sequential) > 0 else None


                
        self.__init_weight()
        
    # FC weight, bias for each Conv Layer
    def __init_weight(self):
        
        if self.model_name == 'resnet152':
            self.conv5_fc.weight.data.uniform_(-0.1, 0.1)
            self.conv5_fc.bias.data.fill_(0)
            self.conv4_fc.weight.data.uniform_(-0.1, 0.1)
            self.conv4_fc.bias.data.fill_(0)
            self.conv3_fc.weight.data.uniform_(-0.1, 0.1)
            self.conv3_fc.bias.data.fill_(0)

        # get_model에서 추출.
        elif self.model_name =='densenet121':

            if not (self.cfg.MODEL.ENCODER_PROJ =='no'): # 'no' : final feature가 1024dim. 
                # print('Dense_init!!!')
                # 1x1 convolution or linear
                
                self.att_embed[0].weight.data.uniform_(-0.1, 0.1)
                self.att_embed[0].bias.data.fill_(0)
            


    def __get_model(self):
        print('start Getting Model..')
        conv5_model, conv5_out_features, conv5_avg_func = None, None, None
        conv4_out_features, conv3_out_features = None, None
        conv4_avg_func, conv3_avg_func = None, None
        if self.model_name == 'resnet152':
            resnet = models.resnet152(pretrained=self.pretrained)
            
            ### ~ Stage5 Model

            conv5_modules = list(resnet.children())[:-2]        # ~Conv5
            # for module in conv5_modules:
            #     print(module)
            conv5_model = nn.Sequential(*conv5_modules)         # builds a sequential model based on it that excludes the final two modules (e.g., the one that does average pooling and the fully connected one)
            # for param in conv5_model.parameters():
            #     param.requires_grad = False # jsp 바꿔야될 것 같기도,,;; 학습을 안 하면 필요가           
            
            for key, param in conv5_model.named_parameters():
                if ('7.0' in key) or ('7.1' in key) or ('7.2' in key):
                    pass
                else:
                    param.requires_grad = False
            conv5_out_features = resnet.fc.in_features # output nodes of the last layer of ResNet-152
            #conv5_avg_func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

            ### ~ Stage4 Model
            conv4_out_features = 1024   #resnet.layer3.bn3.num_features
            #conv4_avg_func = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

            ### ~ Stage3 Model
            conv3_out_features = 512
            #conv3_avg_func = torch.nn.AvgPool2d(kernel_size=28, stride=1, padding=0)

            ### ~ Stage2 Model
            #conv2_out_features = 256
            #conv2_avg_func = torch.nn.AvgPool2d(kernel_size=56, stride=1, padding=0)

            return conv5_model, conv5_out_features, conv5_avg_func, conv4_out_features, conv4_avg_func, \
               conv3_out_features, conv3_avg_func #, conv2_out_features, conv2_avg_func

        
        elif self.model_name == 'densenet121':
            # print('Dense Get_model !!')
            with open(self.encoder_cfg_path) as f:
                encoder_cfg = edict(json.load(f))

            if self.pretrained =='ImageNet':
                encoder_cfg.pretrained = True    
                model = densenet.densenet121(encoder_cfg)
                model = model.features
            
            if self.pretrained =='Chexpert':
                print('Load Chexpert-pretrained densenet-121 : ', self.pretrained_path)
                model = Classifier(encoder_cfg)
                state_dict = torch.load(self.pretrained_path)
                model.load_state_dict(state_dict)
                model = model.backbone.features


            for idx, (key, param) in enumerate(model.named_parameters()):
                
                if self.cfg.MODEL.ENCODER_TRAINING =='Last':
                    if idx<=263:
                        param.requires_grad = False # Dense 1~ Dense3
                    if idx>263:
                        param.requires_grad = True # Dense 4
                elif self.cfg.MODEL.ENCODER_TRAINING =='All': # Dense 1~4
                    if not param.requires_grad:
                        param.requires_grad = True
                elif self.cfg.MODEL.ENCODER_TRAINING =='No': # All Freezing (except final projection layer)
                    if param.requires_grad:
                        param.requires_grad = False


            return model

    def preprop_one_image(self, image_path):
        
        image = Image.open(image_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            
        ])

        return transform(image).unsqueeze(0)

    def forward(self, images):
        """
        :param images:
        :return:
        """

        # # Feature Map 저장
        # conv5, conv4, conv3 = [], [], []

        # # forward 단계에서 필요한 변수 저장
        
        # hooks = [
        #     self.conv5_model[-1].register_forward_hook(
        #         lambda self, input, output: conv5.append(output)
        #     ),
        #     self.conv5_model[-2].register_forward_hook(
        #         lambda self, inpupt, output: conv4.append(output)
        #     ),
        #     self.conv5_model[-3].register_forward_hook(
        #         lambda self, inpupt, output: conv3.append(output)
        #     ), 
        # ]

        # # forward
        # self.conv5_model(images)
        
        # # list to feature
        # conv5_visual_features = conv5[0]
        # conv4_visual_features = conv4[0]
        # conv3_visual_features = conv3[0]

        # del conv5
        # del conv4
        # del conv3

        # torch.cuda.empty_cache()

        # print(np.array(conv5_visual_features).shape)
        # print(np.array(conv4_visual_features).shape)
        # print(np.array(conv3_visual_features).shape)

        if self.model_name == 'resnet152':
            conv5_visual_features = self.conv5_model(images)
            #images:[B,3,224,224], conv5_visual_features: [B,2048,7,7]

            conv5_visual_features = torch.flatten(conv5_visual_features, 2, 3) # ([B,2048,7*7])
            conv5_visual_features = conv5_visual_features.permute(2,0,1)       # ([7*7, B, 2048])
            conv5_visual_features = self.conv5_fc(conv5_visual_features)             # ([7*7, B, 512])
            #print("conv5_visual_features.shape:", conv5_visual_features.shape)
            # print("conv5 : requires_grad", conv5_visual_features.requires_grad)
            conv4_visual_features = self.conv5_model[:-1](images)
            # conv4_visual_features: [B,1024,14,14]
            conv4_visual_features = torch.flatten(conv4_visual_features, 2, 3) # ([B,1024,14*14])
            conv4_visual_features = conv4_visual_features.permute(2,0,1)       # ([14*14, B, 1024])
            conv4_visual_features = self.conv4_fc(conv4_visual_features)             # ([14*14, B, 512])

            conv3_visual_features = self.conv5_model[:-2](images)    
            # conv3_visual_features: [B, 512, 28, 28]
            conv3_visual_features = torch.flatten(conv3_visual_features, 2, 3) # ([B,512,28*28])
            conv3_visual_features = conv3_visual_features.permute(2,0,1)       # ([28*28, B, 512])
            conv3_visual_features = self.conv3_fc(conv3_visual_features)             # ([28*28, B, 512])
            #print("conv3_visual_features.shape:", conv3_visual_features.shape)

            """conv2_visual_features = self.conv5_model[:-3](images)    
            # conv2_visual_features: [B, 256, 56, 56]
            conv2_visual_features = torch.flatten(conv2_visual_features, 2, 3) # ([B,128,56*56])
            conv2_visual_features = conv2_visual_features.permute(2,0,1)       # ([56*56, B, 256])
            conv2_visual_features = self.conv2_fc(conv2_visual_features)             # ([56*56, B, 512])
            #print("conv2_visual_features.shape:", conv2_visual_features.shape)"""
            
            return [conv5_visual_features, conv4_visual_features, conv3_visual_features] #, conv2_visual_features
        
        if self.model_name == 'densenet121':
            # print('Dense... Forward !!')
            
            if self.cfg.MODEL.ENCODER_PROJ =='1x1conv':
                features = self.extractor(images) # [B, 1024, 14, 14] (input : 448일 때.)
                # print('feature.shape',  features.shape)
                att_feats = self.att_embed(features) # [B, 512, 14, 14]
                # att_feats = self.final_proj(features) # [B, 512, 14, 14]
                # print('att_feats.shape', att_feats.shape)
                att_feats = torch.flatten(att_feats, 2, 3) # [B, 512, 14*14]
                # print('flatten.shape', att_feats.shape)
                att_feats = att_feats.permute(2, 0, 1) # [14*14, B, 512] 
                # print('att_feats.shape', att_feats.shape)
            elif self.cfg.MODEL.ENCODER_PROJ =='fc':
                features = self.extractor(images) # [B, 1024, 14, 14] (input : 448일 때.)
                att_feats = torch.flatten(features, 2, 3) # [B, 1024, 14*14]
                att_feats = att_feats.permute(2, 0, 1) # [14*14, B, 1024]
                att_feats = self.att_embed(att_feats) # [14*14, B, 512]
                
            elif self.cfg.MODEL.ENCODER_PROJ =='no': # Projection을 하지 않아도 Transformer의 dimension과 일치하는 경우(embeding no..)
                features = self.extractor(images) # [B, 1024, 14, 14]
                att_feats = torch.flatten(features, 2, 3) # [B, 1024, 14*14]
                att_feats = att_feats.permute(2, 0, 1) # [14*14, B, 1024]
                # print(att_feats.shape)
            

            return att_feats #[src*src, batch, hid_dim]





if __name__ =='__main__':
    image_root = '../../H_LSTM_Transformer/data/all_jpgs'
    image_name= os.listdir(image_root)[0]
    image_path = os.path.join(image_root, image_name)
    print('Load Encoder')
    
    encoder = VisualFeatureExtractor('resnet152', 'Image_Net')
    encoder.train()
    
    print('Complete Load.')
    image = encoder.preprop_one_image(image_path)

    input_feats = encoder.forward(image)
 
    normal_feats=[input_feats[0]*2, input_feats[0]*2, input_feats[0]*2]
    diff_feats = []
    for input_feat, normal_feat in zip(input_feats, normal_feats):
        diff_feat = input_feat * normal_feat
        diff_feats.append(diff_feat)
    
    diff_feats = torch.cat(diff_feats, 0) #
    print(diff_feats.shape, diff_feats.requires_grad)
    total_features = diff_feats.permute(1,0,2)
    print(total_features.shape)
    # print(input_feats[0].shape, normal_feats[0].shape)
    # diff_feats = []
    # for input_feat, normal_feat in zip(input_feats, normal_feats):
    #     diff_feat = input_feat * normal_feat

    #     diff_feats.append(torch.from_numpy(diff_feat))
    #     print('diff:', diff_feat.shape)
    
    # total = torch.cat(diff_feats)
    # print(total.shape)

    # diff_feats = torch.cat(torch.from_numpy(diff_feats)) #
    # total_features = diff_feats.permute(1,0,2)
    # print(total_features.shape)
    