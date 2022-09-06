# MIC_team

## Workspace
- Captioning : Medical Image Captioning 
- Dataset : Chexpert 등 (공용)
- SSL : Self-Supervised Learning (혁종)
- SL : Supervised pre-training (진수)
- FeatureEval : Linear Evaluation / K-NN Evaluation(?) (유진)
- Visualization : NLP 관련 시각화 / 그 외 시각화 샘플들 (승용)

- ETC : 진짜 잡폴더


## CheXpert 관련 내용
- Official Paper : https://arxiv.org/pdf/1901.07031.pdf
- Official Site : https://stanfordmlgroup.github.io/competitions/chexpert/
- Other Github :  https://github.com/gaetandi/cheXpert
  - Dataset, Pre-Training, Evaluation 다 있어요 ! 



---

# End-to-End Medical Image Captioning(DeXTr)

Many research use only language model in image captioning.  
In other words, model's input is image feature from pre-trained CNN networks  

In contrast, my model take image as input, so that CNN networks can also learn information about image captioning task.  
Maybe CNN networks(visual encoder) will have potential to work better than (practically frozen)pre-trained CNN on X-ray datasets(e.g. CheXpert), in this way.     

Besides, my DeXTr also use several normal images as input(visual encoder part), extract mutual information between input and normal images (feature difference part), pass this information(feature) to [X-Transformer](https://arxiv.org/abs/2003.14080) (language model part).  

See below for details.

[Report(Written in Korean)](https://www.notion.so/sjinu/End-to-End-Medical-Image-Captioning-63f3fa7dc9be45dea3b8aa664890f123)

## Model

### Architecture
<img width="1138" alt="image" src="https://user-images.githubusercontent.com/71121461/160956787-d2bd9d8c-1ebf-41ea-bbe8-3488422d7b9d.png">

### Way to use normal image
<img width="492" alt="image" src="https://user-images.githubusercontent.com/71121461/161174793-0bc2e8d4-4aaf-4652-8aea-d68d08366d31.png">

see the function `__getitem__` in [*Dextr/coco_dataset.py*](https://github.com/sjinu96/End-to-End-Medical-Image-Captioning/blob/main/DeXTr/datasets/coco_dataset.py)

#### CA(Contrastive Attention)

> Wrote the [code of contrastive attention](https://github.com/sjinu96/End-to-End-Medical-Image-Captioning/blob/main/DeXTr/models/contra_att.py) based on theory of [Liu et al.(2022)](https://arxiv.org/abs/2106.06965)
<img width="961" alt="image" src="https://user-images.githubusercontent.com/71121461/161176159-7870a384-0d12-4723-b6c8-90f3ffb5d556.png">

### Evaluation

#### Quantitative
<img width="949" alt="image" src="https://user-images.githubusercontent.com/71121461/161175225-76bb9bd7-e00c-418f-adf5-a73f518b0753.png">

#### Qualitative

<img width="894" alt="image" src="https://user-images.githubusercontent.com/71121461/172307602-ff03178f-caa5-47c2-81b7-4015f9d84f26.png">

### Other results

#### Stability according to 'visual encoder & pre-training dataset'
<img width="669" alt="image" src="https://user-images.githubusercontent.com/71121461/161175757-b459530c-b3af-4f32-b9a4-c63008da1672.png">


#### 2d representation
<img width="969" alt="image" src="https://user-images.githubusercontent.com/71121461/161175574-3d282ed4-b54b-407b-8b8a-2aeb829b3916.png">

----
  
## About Code


**DeXTr**(Full architecture) :  [DeXTr/models/Detr.py](https://github.com/sjinu96/End-to-End-Medical-Image-Captioning/tree/main/DeXTr/models)  

**Visual Encoder** :  [DeXTr/models/visual_extractor.py](https://github.com/sjinu96/End-to-End-Medical-Image-Captioning/blob/main/DeXTr/models/visual_extractor.py)  
**Feature Difference** :  CA in [DeXTr/models/contra_att.py](https://github.com/sjinu96/End-to-End-Medical-Image-Captioning/blob/main/DeXTr/models/contra_att.py) & Others in [DeXTr/models/Detr.py](https://github.com/sjinu96/End-to-End-Medical-Image-Captioning/tree/main/DeXTr/models)  
**Language Model+Report Generation** :   [Code by Pan(Author of X-LAN)](https://github.com/JDAI-CV/image-captioning)  


---

**Training** : [DeXTr/main_mimic.py](https://github.com/sjinu96/End-to-End-Medical-Image-Captioning/blob/main/DeXTr/main_mimic.py)

`$ CUDA_VISIBLE_DEVICES=1 python3 main_mimic.py --folder ./experiments/name`

