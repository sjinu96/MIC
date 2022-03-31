import torchvision.transforms as transforms

def init_train_transform(resize, crop_size):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])
        #transforms.Normalize((0.5, 0.5, 0.5),
        #                     (0.5, 0.5, 0.5))])
    return transform

def init_val_transform(resize, crop_size):
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)), # jsp, 조심
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])
        #transforms.Normalize((0.5, 0.5, 0.5),
        #                     (0.5, 0.5, 0.5))])
    return transform

def init_normal_transform(resize, crop_size):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))])
        #transforms.Normalize((0.5, 0.5, 0.5),
        #                     (0.5, 0.5, 0.5))])
    return transform

    # Image name 과 해당 image의 tag vector를 각각 list형태로 반환


# CA 실행 시 학습 시간 절약을 위해 crop 제외, Tensor로 변경해놓음
def init_normal_transform_ca(crop_size):
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size)])

    return transform

    # Image name 과 해당 image의 tag vector를 각각 list형태로 반환

# Label 이 필요할 수도
def __load_label_list(self, file_list):
    labels = []                                         # tag vector 들의 list
    filename_list = []                                  # image name 들의 list
    with open(file_list, 'r') as f:
        for line in f:
            items = line.split()
            image_name = items[0]                       # 해당 image name ex) CXR1972_IM-0633-1001
            label = items[1:]                           # 해당 image의 tag vector(0,1로 이루어진)
            label = [int(i) for i in label]             # string -> int
            image_name = '{}.jpg'.format(image_name)
            filename_list.append(image_name)
            labels.append(label)
    return filename_list, labels
# 해당 index 의 image, image_name, list(label / np.sum(label)), target, sentence_num, max_word_num 반환