from models.updown import UpDown
from models.xlan import XLAN
from models.xtransformer import XTransformer
from models.visual_extractor import VisualFeatureExtractor
from models.contra_att import ContraAtt

from XTransformer.lib.config import cfg

__factory = {
    'UpDown': UpDown,
    'XLAN': XLAN,
    'XTransformer': XTransformer
}

### jsp

__factory_encoder ={
    # 'densenet161+ImageNet' : VisualFeatureExtractor(model_name = 'densenet161', pretrained=True ),
    'resnet152+ImageNet' : VisualFeatureExtractor(model_name = 'resnet152', pretrained=True), 
    # 'densenet121+ImageNet' : VisualFeatureExtractor(model_name = 'densenet121', pretrained='ImageNet', encoder_cfg_path = 'example.json') # 일단 상수 고정
}
def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)



### jsp
def create_encoder(name, pretrained, cfg): #encoder_cfg_path=False, pretrained_path=False):

    if 'resnet' in name:
        if name+'+'+pretrained not in __factory_encoder:
            raise KeyError("Unknown encoder model:", name+'+'+pretrained)

        return __factory_encoder[name+'+'+pretrained]
    elif ('densenet' in name) and (pretrained=='ImageNet'):
        # print('-----------------Create encoder... : ', name+'+'+pretrained)
        return VisualFeatureExtractor(model_name = 'densenet121', pretrained='ImageNet', cfg = cfg)
    elif ('densenet' in name) and (pretrained=='Chexpert'):
        print('denseNet121 + Chexpert')
        return VisualFeatureExtractor(model_name = 'densenet121', pretrained='Chexpert', cfg = cfg)


def create_contra_att(cfg):

    return ContraAtt(cfg = cfg)