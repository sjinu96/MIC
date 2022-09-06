from signal import SIG_DFL
from VisualEncoder import VisualEncoder
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import argparse
import os
from PIL import Image
class FeatureExtractor:
    def __init__(self, args):
        
        self.args = args
        
        self.train_transform = self.init_train_transform()
        self.normal_transform = self.init_normal_transform()
        self.train_data_loader = self.init_data_loader(args.input_image_dir, args.batch_size, transform=self.train_transform, mode='train')
        self.normal_data_loader = self.init_data_loader(args.normal_image_dir, args.batch_size,  transform=self.normal_transform, mode='normal')
        
        if args.gpu_ids >= 0:
            # torch.cuda.set_device(args.gpu_ids)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else : 
            self.device = 'cpu'
        self.encoder = VisualEncoder(args.model_name, args.pretrained).to(self.device)
         

        print('device to use', self.device.type + f':{args.gpu_ids}')
      
    def init_data_loader(self, image_dir, batch_size, transform, mode='train'):

        dataset = ImageFolderWithPaths(image_dir, transform) # Custom Dataset

        if mode=='train':
            self.num_images = len(dataset)
        
        dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size = batch_size)

        return dataloader

    def init_train_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),

            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))])

        return transform

    def init_normal_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        return transform


    def get_final_features(self, input_images, normal_images, feat_fusion_mode):
        """배치단위 Feature 추출"""


        if self.device.type == 'cuda':
            input_images = input_images.cuda() 
            normal_images = normal_images.cuda()
        # input image features
        # [7x7, B, 512] , [14x14, B, 512], [28x28, B, 512]

        conv5_fc_features, conv4_fc_features, conv3_fc_features = self.encoder.forward(input_images)      # torch.Size([14*14, B, 512])) / torch.Size([B, 2048])
        
    
        # normal features
        # 위와 동일1
        conv5_norm_features, conv4_norm_features, conv3_norm_features = self.encoder.forward(normal_images)  # torch.Size([1, 2048, 7, 7]), torch.Size([2048])

        conv5_diff_features = conv5_fc_features * conv5_norm_features  
        conv4_diff_features = conv4_fc_features * conv4_norm_features   
        conv3_diff_features = conv3_fc_features * conv3_norm_features 

        # [7x7+14x14+28x28, B, 512] = [1029, B, 512]
        diff_features = torch.cat([conv5_diff_features, conv4_diff_features, conv3_diff_features])

        # [B, 1029, 512]
        total_features = diff_features.permute(1,0,2)
        
        return total_features 
    

    def get_features(self, mode='only_one_normal'):
        """output : {filename : features} - json
           
           file name을 기준으로 Pair """
        feature_json ={}

    
        # not pair
        if mode == 'only_one_normal':
            assert len(self.normal_data_loader) == 1

            normal_image, _ = next(iter(self.normal_data_loader))

            # [1, 3, 224, 224] -> [B, 3, 224, 224]
            normal_images = normal_image.expand(self.args.batch_size, -1, -1, -1)

            idx = 0 
            for (input_images, path) in self.train_data_loader:

                if len(input_images) != len(normal_images):
                    normal_images = normal_image.expand(len(input_images), -1, -1, -1)


                if self.device.type =='cuda':
                    input_images = input_images.cuda()
                    normal_images = normal_images.cuda()

                features = self.get_final_features(input_images, normal_images, self.args.feat_fusion_mode)
                feature_json.update(dict(zip(path, features.detach().cpu().numpy())))
                
                # 메모리 삭제
                # del features, input_images, normal_images
                # torch.cuda.empty_cache()
                
                idx += self.args.batch_size
                print(f'Compleate : {idx}/ {self.num_images}')
            

        # pair
        elif mode == 'pair':
            assert len(self.train_data_loader)==len(self.normal_data_loader)

            for (input_images, path), (normal_images, path_normal) in zip (self.train_data_loader, self.normal_data_loader):
                
                assert path == path_normal

                features = self.get_final_features(input_images, normal_images, self.args.feat_fusion_mode)
                # add 'image file : features' batch to dictionary
                feature_json.update(dict(zip(path, features.detach().cpu().numpy())))
                
                # 메모리 삭제
                # del features, input_images, normal_images
                # torch.cuda.empty_cache()

                idx += self.args.batch_size
                print(f'Compleate : {idx}/ {self.num_images}')

        else: 
            pass

        return feature_json

    def save_features(self, feature_json):
        """json to """
        pass

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


class ImageFolderWithPaths(Dataset):
    
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.image_path_list = sorted(glob(os.path.join(image_dir, '*')))
        self.transform=transform
        
    def __getitem__(self, index):

        image_path = self.image_path_list[index]
        image = Image.open(image_path).convert('RGB')
        image_name = image_path.split('/')[-1] # 80B49786-...-261155eb.jpg
        if self.transform is not None:
            image = self.transform(image)


        return image, image_name

    def __len__(self):
        return len(self.image_path_list)



# Folder 내에 클래스가 존재하ㅐㄹ 때.
# class ImageFolderWithPaths(datasets.ImageFolder):
#     """Custom dataset that includes image file paths.
#     Extends torchvision.datasets.ImageFolder
#     """

#     # override the __getitem__ method.
#     def __getitem__(self, index):
#         # this is what ImageFolder normally returns 
#         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
#         image = original_tuple[0]
#         # the image file path
#         path = self.imgs[index][0]
#         # make a new tuple that includes original and the path
#         tuple_with_path = (image + (path,))

#         return tuple_with_path




if __name__ == '__main__':

    parser =argparse.ArgumentParser()

    parser.add_argument('input_image_dir', default=None, type=str)
    parser.add_argument('normal_image_dir', default=None, type=str)
    parser.add_argument('model_name', default='resnet152', type=str)
    parser.add_argument('pretrained', default='ImageNet', type=str)
    parser.add_argument('batch_size', default=1, type=int)


    parser.add_argument('resize', default=256, type=int)
    parser.add_argument('crop_size', default=224, type=int)
    parser.add_argument('feat_fusion_mode', default='ewp', type=str)
    parser.add_argument('diff_mode', default='only_one_normal', type=str)

    parser.add_argument('gpu_ids', default='0', type=int)


    input_dir = '/home/mskang/jinsu/med/H_LSTM_Transformer/data/all_jpgs'
    normal_dir = '/home/mskang/jinsu/med/H_LSTM_Transformer/data/normal_image'
    

    # py 파일 테스트용
    args = parser.parse_args(args=[input_dir, normal_dir, 'resnet152', 'ImageNet', \
    '5', '256', '224', 'ewp', 'only_one_normal', '0'])
    # in cmd
    # args = parser.parse_args()

    feature_extractor = FeatureExtractor(args)

    print('Feature Extractor 할당 끝')

    feature_json = feature_extractor.get_features(args.diff_mode)
    
    print(feature_json)

