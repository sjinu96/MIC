import os
import random
import numpy as np
import torch
import torch.utils.data as data
import lib.utils as utils
import pickle
from PIL import Image
from glob import glob

from torchvision import transforms as T

# att_feat 관련 코드들 모두 main_mimic.py로
class CocoDataset(data.Dataset):
    def __init__(
        self, 
        image_ids_path, 
        input_seq, 
        target_seq,
        gv_feat_path, 
        # att_feats_folder, 
        input_images_folder, # jsp
        normal_images_folder, # jsp
        seq_per_img,
        max_feat_num,
        all_file, 
        input_transform, # jsp
        normal_transform, # jsp
        leverage_normal, # jsp
        num_ca, # jsp
    ):
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        #print("image_ids_path:", image_ids_path)
        self.image_ids = utils.load_lines(image_ids_path)
        # self.att_feats_folder = att_feats_folder if len(att_feats_folder) > 0 else None
        self.input_images_folder = input_images_folder # jsp
        self.normal_images_folder = normal_images_folder # jsp
        self.num_normal = len(self.normal_images_folder) #jsp

        self.leverage_normal = leverage_normal # jsp
        
        if leverage_normal =='CA':
            self.normal_tensors_folder = normal_images_folder
            self.normal_tensors_name = os.listdir(self.normal_tensors_folder)
            self.num_ca = num_ca
            
        else :
            self.normal_images_name = os.listdir(self.normal_images_folder)
        
        
        self.input_images_name = os.listdir(self.input_images_folder)
        if 'jpg' in os.listdir(self.input_images_folder)[0]:
            self.image_type = '.jpg'
        else: # 아마 png.
            self.image_type = '.png'





        print("Training with normal image Using `{}` Method".format(self.leverage_normal))
        
        self.input_transform = input_transform
        self.normal_transform = normal_transform
        self.gv_feat = pickle.load(open(gv_feat_path, 'rb'), encoding='bytes') if len(gv_feat_path) > 0 else None
        
        self.all_ids = utils.load_lines(all_file) # changed
        # changed
        self.jpg_to_id = {}
        self.id_to_jpg = {}
        for i, jpg in enumerate(self.all_ids):
            self.jpg_to_id[jpg] = i
            self.id_to_jpg[i] = jpg
        
        if input_seq is not None and target_seq is not None:
            self.input_seq = pickle.load(open(input_seq, 'rb'), encoding='bytes')
            self.target_seq = pickle.load(open(target_seq, 'rb'), encoding='bytes')
            #print("len_self.image_ids:", len(self.image_ids))
            self.seq_len = len(self.input_seq[str(self.jpg_to_id[self.image_ids[0]])][0,:]) # changed 201
        else:
            self.seq_len = -1
            self.input_seq = None
            self.target_seq = None
         
    def set_seq_per_img(self, seq_per_img):
        self.seq_per_img = seq_per_img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # print('get item 시작--------------------')
        image_id = self.image_ids[index]
        indices = np.array([index]).astype('int')

        if self.gv_feat is not None:
            gv_feat = self.gv_feat[image_id]
            gv_feat = np.array(gv_feat).astype('float32')
        else:
            gv_feat = np.zeros((1,1))

        # get Image, Normal..


        num_normal_images=len(os.listdir(self.normal_images_folder))

        # only one normal
        if num_normal_images == 1: # or self.leverage_normal =='single'
            if not  self.leverage_normal =='single':
                raise NameError('You Should when normal images pool is only one')
            image_name = os.listdir(self.normal_images_folder)[0]
            normal_image = Image.open(os.path.join(self.normal_images_folder, image_name)).convert('RGB')
            normal_image = self.normal_transform(normal_image) # Tensor [3, 224, 224]

        elif self.leverage_normal =='random': # random하게 샘플링
            random_idx = np.random.randint(self.num_normal)
            normal_image = Image.open(os.path.join(self.normal_images_folder, self.normal_images_name[random_idx])).convert('RGB')
            normal_image = self.normal_transform(normal_image)
            # print(glob(os.path.join(self.normal_images_folder, '*'))[random_idx])
        # normal pair
        elif self.leverage_normal =='pair':
            normal_image = Image.open(os.path.join(self.normal_images_folder, str(image_id)+self.image_type)).convert('RGB')
            normal_image = self.normal_transform(normal_image)
        elif self.leverage_normal =='CA':
            random_idxes = np.random.randint(0, self.num_normal, self.num_ca) # 전체 normal image 중 num_ca개 활용
            normal_tensors = []
            for idx in random_idxes:
                # CA를 사용할 경우 transform 연산 시간이 오래 걸리기 때문에 resize -> tensor 까지만 미리 해놓음.
                normal_tensor = torch.load(os.path.join(self.normal_tensors_folder, self.normal_tensors_name[idx])) # [3, X, 512] tensor
                # 단, Augmentation 성능 향상 겸 Crop만 따로(Random)
                # 특히, 이 때 normal_transform은 crop만 적용.
                normal_tensors.append(self.normal_transform(normal_tensor).unsqueeze(0)) # [3, crop_size, crop_size]
                # normal_tensors.append(T.RandomCrop(448)(normal_tensor).unsqueeze(0)) # [3, 448, 448] tensor 

            # N(=10)개의 이미지지만, 코드 효율을 위해 normal_images가 아닌 normal_image로 명명 
            normal_image = torch.cat(normal_tensors, axis=0) # [N, 3, 448, 448] (N=10)
            # print('CoCo dataset - get_item()', normal_image.shape) 
            
        
        # print(os.path.join(self.input_images_folder, str(image_id)+'.jpg'))
        
        input_image = Image.open(os.path.join(self.input_images_folder, str(image_id)+self.image_type)).convert('RGB')
        
        # print('input_image', type(input_image), np.array(input_image).shape)
        
        # print('input_transform', type(self.input_transform), self.input_transform)
        # print('input image', type(input_image), np.array(input_image).shape)
        input_image = self.input_transform(input_image) # Tensor[3,224,224]
        
        # print('input_image : ', type(input_image), input_image.shape)
        # print('normal_image : ', type(normal_image), normal_image.shape)
        
        

        # 생략해도 무방
        # if self.att_feats_folder is not None:
        #     att_feats = np.load(os.path.join(self.att_feats_folder, str(image_id) + '.npz'))['feat']
        #     att_feats = np.array(att_feats).astype('float32')
        # else:
        #     att_feats = np.zeros((1,1))

        # att_feats :  <class 'numpy.ndarray'> (1029, 512)

        # 생략해도 무방
        # if self.max_feat_num > 0 and att_feats.shape[0] > self.max_feat_num:
        #    att_feats = att_feats[:self.max_feat_num, :]


        # val 진행시.
        if self.seq_len < 0:
            return indices, gv_feat, input_image, normal_image #att_feats

        input_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        target_seq = np.zeros((self.seq_per_img, self.seq_len), dtype='int')
        
        n = len(self.input_seq[str(self.jpg_to_id[image_id])])   # changed
        if n >= self.seq_per_img:
            sid = 0
            ixs = random.sample(range(n), self.seq_per_img)                
        else:
            sid = n
            ixs = random.sample(range(n), self.seq_per_img - n)
            input_seq[0:n, :] = self.input_seq[image_id]
            target_seq[0:n, :] = self.target_seq[image_id]
           
        for i, ix in enumerate(ixs):
            input_seq[sid + i] = self.input_seq[str(self.jpg_to_id[image_id])][ix,:]   # changed
            target_seq[sid + i] = self.target_seq[str(self.jpg_to_id[image_id])][ix,:] # changed
        return indices, input_seq, target_seq, gv_feat, input_image, normal_image#att_feats
