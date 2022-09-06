import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models


class VisualEncoder(nn.Module):
    def __init__(self, model_name='resnet152', pretrained=False):
        super(VisualEncoder, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.conv5_model, self.conv5_out_features, self.conv5_avg_func, \
        self.conv4_out_features, self.conv4_avg_func, \
        self.conv3_out_features, self.conv3_avg_func = self.__get_model()
        self.activation = nn.ReLU()
        
        ## Fully Connected Layer
        self.conv5_fc = nn.Linear(in_features=self.conv5_out_features, out_features=512) # 512 OOM
        self.conv4_fc = nn.Linear(in_features=self.conv4_out_features, out_features=512)
        self.conv3_fc = nn.Linear(in_features=self.conv3_out_features, out_features=512)
        #self.conv2_fc = nn.Linear(in_features=self.conv2_out_features, out_features=128)
        
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14)) # why 14?
        #self.fc = nn.Linear(in_features=2048, out_features=512)
        
        self.__init_weight()
        
    # FC weight, bias for each Conv Layer
    def __init_weight(self):
        self.conv5_fc.weight.data.uniform_(-0.1, 0.1)
        self.conv5_fc.bias.data.fill_(0)
        self.conv4_fc.weight.data.uniform_(-0.1, 0.1)
        self.conv4_fc.bias.data.fill_(0)
        self.conv3_fc.weight.data.uniform_(-0.1, 0.1)
        self.conv3_fc.bias.data.fill_(0)
        #self.conv2_fc.weight.data.uniform_(-0.1, 0.1)
        #self.conv2_fc.bias.data.fill_(0)

    def __get_model(self):
        conv5_model, conv5_out_features, conv5_avg_func = None, None, None
        conv4_out_features, conv3_out_features = None, None
        conv4_avg_func, conv3_avg_func = None, None
        if self.model_name == 'resnet152':
            
            # ImageNet - pretrained
            
            if self.pretrained =='ImageNet':
                resnet = models.resnet152(pretrained=True)
            elif self.pretrained =='CheXpert':
                pass
            else: 
                pass
            
            ### ~ Stage5 Model
            conv5_modules = list(resnet.children())[:-2]        # ~Conv5
            conv5_model = nn.Sequential(*conv5_modules)         # builds a sequential model based on it that excludes the final two modules (e.g., the one that does average pooling and the fully connected one)
            for param in conv5_model.parameters():
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
        elif self.model_name == 'densenet201':
            densenet = models.densenet201(pretrained=self.pretrained)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            out_features = densenet.classifier.in_features
        
        return conv5_model, conv5_out_features, conv5_avg_func, conv4_out_features, conv4_avg_func, \
               conv3_out_features, conv3_avg_func #, conv2_out_features, conv2_avg_func

    def forward(self, images):
        """
        :param images:
        :return:
        """

        conv5_visual_features = self.conv5_model(images)
        #images:[B,3,224,224], conv5_visual_features: [B,2048,7,7]
        conv5_visual_features = torch.flatten(conv5_visual_features, 2, 3) # ([B,2048,7*7])
        conv5_visual_features = conv5_visual_features.permute(2,0,1)       # ([7*7, B, 2048])
        conv5_visual_features = self.conv5_fc(conv5_visual_features)             # ([7*7, B, 512])
        #print("conv5_visual_features.shape:", conv5_visual_features.shape)

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
        
        return conv5_visual_features, conv4_visual_features, conv3_visual_features  #, conv2_visual_features