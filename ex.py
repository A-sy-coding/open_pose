# import os, json
# from glob import glob
# # path = './data/mask/train2014/mask_COCO_train2014_000000528276.jpg'
# path = './data/mask/train2014/mask_COCO_train2014_000000054351.jpg'

# # path1 = './data/mask/train_2014'
# # print(glob(os.path.join(path1,'*.jpg')))
# imgs = glob(os.path.join('./data/mask/train2014', '*'))
# imgs_val = glob(os.path.join('./data/mask/val2014', '*'))
# # print(imgs)
# if path in imgs:
#     print('exist')
# else:
#     print('no exist')

# # print('train :' ,len(imgs))
# # print('val : ' ,len(imgs_val))
# # from utils import make_datapath_list
# def make_datapath_list(rootpath):
#     """
#     학습 및 검증의 화상 데이터와 어노테이션 데이터, 마스크 데이터의 파일 경로 리스트를 작성한다.
#     """

#     # rootpath : COCO.json 파일이 있는 절대경로
    
#     # annotaion json file read
#     json_path = os.path.join(rootpath, 'COCO.json')
#     with open(json_path) as f:
#         data = json.load(f)
#         data_json = data['root']  # keys : ['root']

#     # index 저장
#     num_samples = len(data_json) # 전체 파일 개수
#     train_indexes, val_indexes = [], []

#     for count in range(num_samples):
#         if data_json[count]['isValidation'] != 0:  # validation files
#             val_indexes.append(count)
#         else:
#             train_indexes.append(count)  # train files
    
    
#     train_img_list, val_img_list = [],[]  # img 경로 저장
#     train_mask_list, val_mask_list = [], [] # mask 경로 저장
#     train_meta_list, val_meta_list = [], [] # annotation 경로 저장
    
#     # COCO.json 파일에 있는 train 이미지가 mask/train2014에는 존재하지 않는 경우가 발생하기 때문에 둘의 교집합 이미지들만 출력하도록 한다. 
#     imgs = glob(os.path.join('./data/mask/train2014', '*')) # mask폴더 안에 있는 train 이미지 리스트
#     imgs_val = glob(os.path.join('./data/mask/val2014', '*')) # mask폴더 안에 있는 val 이미지 리스트
    
    
#     for idx in train_indexes:
#         img_path = os.path.join(rootpath, data_json[idx]['img_paths'])   # train_img 경로 저장
#         img_idx = data_json[idx]['img_paths'][-16:-4] # 이미지 파일 이름만 가져오기
#         anno_path = "./data/mask/train2014/mask_COCO_train2014_" + img_idx + '.jpg' # mask file 경로 저장

#         if anno_path in imgs:
#             train_img_list.append(img_path)
#             train_mask_list.append(anno_path)
#             train_meta_list.append(data_json[idx]) # data_json의 모든 value들을 저장
        
#     for idx in val_indexes:
#         img_path = os.path.join(rootpath, data_json[idx]['img_paths'])  # val_img path store
#         img_idx = data_json[idx]['img_paths'][-16:-4] 
#         anno_path = './data/mask/val2014/mask_COCO_val2014_' + img_idx + '.jpg'

#         if anno_path in imgs_val:
#             val_img_list.append(img_path)
#             val_mask_list.append(anno_path)
#             val_meta_list.append(data_json[idx]) # data_json의 모든 value들을 저장
    
#     print('-----------------')
#     print('COCO train : ', len(train_indexes))
#     print('COCO val : ', len(val_indexes))
#     print('mask/train2014 :', len(imgs))
#     print('mask/val2014 :', len(imgs_val))
#     print('train_img :', len(train_img_list))
#     print('train_mask :', len(train_mask_list))
#     print('val_img :', len(val_img_list))
#     print('val_mask :', len(val_mask_list))

#     return train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list
# # 파일 경로 리스트
# train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list(rootpath = './data')
# # print(len(train_img_list))


import random
import math
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import make_datapath_list, DataTransform, COCOkeypointsDataset

# 파일 경로 리스트
train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list(rootpath = './data')
train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list = train_img_list[:500], val_img_list[:500], train_mask_list[:500], val_mask_list[:500], train_meta_list[:500], val_meta_list[:500]

# 훈련, 검증 데이터 가져오기 --> 전처리도 같이 수행
train_dataset = COCOkeypointsDataset(train_img_list, train_mask_list, train_meta_list, phase='train', transform=DataTransform())
val_dataset = COCOkeypointsDataset(val_img_list, val_mask_list, val_meta_list, phase='val', transform=DataTransform())


# dataloader
batch_size = 32
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}
print(dataloaders_dict['train'].dataset)
print(dataloaders_dict['val'].dataset)
print(train_dataset)
print(val_dataset)