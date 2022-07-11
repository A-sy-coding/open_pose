'''
uils/data_loader의 ground_truth 함수를 사용하여 신체 좌표 annotaion과 pafs를 구한다.
신체좌표 annotaion을 가우스 분포를 이용하여 좀 더 확률을 높이도록 한다. (분산을 키운다..)
pafs는 부위 사이의 방향정보를 나타내게 되는데, 부위 사이의 직선상에 있으면 1, 없으면 0으로 표기한다.
'''
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import get_ground_truth
from utils import make_datapath_list # json 파일 가져오는 클래스
from utils import DataTransform # 전처리 수행해주는 클래스
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 데이터 가져오기
train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list('./data')
print(len(train_img_list))
print(len(train_mask_list))
print(len(train_meta_list))

# 이미지 읽기
index = 24
img = cv2.imread(val_img_list[index])
mask_miss = cv2.imread(val_mask_list[index])
meta_data = val_meta_list[index]

# 이미지 전처리
transform = DataTransform()
meta_data, img, mask_miss = transform("train", meta_data, img, mask_miss)

img = img.numpy().transpose((1,2,0))
mask_miss = mask_miss.numpy().transpose((1,2,0))

# openpose용 annotaion 데이터 생성
heat_mask, heatmaps, paf_mask, pafs = get_ground_truth(meta_data, mask_miss)
# print(heatmaps)

# 왼쪽 팔굼치 히트맵
from matplotlib import cm
img = Image.fromarray(np.uint8(img*255))
img = np.asarray(img.convert('RGB'))

heat_map = heatmaps[:,:,6] # 6이 왼쪽 팔꿈치를 의미한다.
heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)* 255))
heat_map = np.asarray(heat_map.convert('RGB'))
heat_map = cv2.resize(heat_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

blend_img = cv2.addWeighted(img, 0.5, heat_map, 0.5, 0)
plt.imshow(blend_img)
plt.savefig('./sample_img/annotation_blend.jpg')


#########################
# split dataset & dataloader
#########################
from utils import COCOkeypointsDataset
import torch.utils.data as data
train_dataset = COCOkeypointsDataset(train_img_list, train_mask_list, train_meta_list, phase='train', transform = DataTransform())
val_dataset = COCOkeypointsDataset(val_img_list, val_mask_list, val_meta_list, phase='val', transform = DataTransform())

# img, heatmaps, heat_mask, pafs, paf_mask return
# item = train_dataset.__getitem__(0)
# print(item[0].shape)
# print(item[1].shape)
# print(item[2].shape)
# print(item[3].shape)
# print(item[4].shape)

# use dataloader 
batch_size = 8

train_dataloader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True)

# dict에 저장
dataloaders_dict = {'train' : train_dataloader, 'val':val_dataloader}

# 작동 확인
# batch_iterator = iter(dataloaders_dict['train'])


# item = next(batch_iterator) # 첫번째 원소 꺼내기    # 작동할 때마다 next가 되는것 같다....
#                             # img, heatmaps, heat_mask, pafs, paf_mask return
# print(item[0].shape)
# print(item[1].shape)
# print(item[2].shape)
# print(item[3].shape)
# print(item[4].shape)

# print(cv2.imread(train_img_list[91062]))
# print(cv2.imread(train_mask_list[91062]))  # None
# print(train_meta_list[91062])