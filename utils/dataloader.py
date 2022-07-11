# import os, json
# from PIL import Image
# import cv2
# import numpy as np
# from scipy import misc, ndimage
# import torch
# import torch.utils.data as data

# from utils import Compose, get_anno, add_neck, aug_scale, aug_rotate, aug_croppad, aug_flip, remove_illegal_joint, Normalize_Tensor, no_Normalize_Tensor

# def putGaussianMaps(center, accumulate_confid_map, params_transform):
#     '''가우스 맵으로 변환'''
#     crop_size_y = params_transform['crop_size_y']
#     crop_size_x = params_transform['crop_size_x']
#     stride = params_transform['stride']
#     sigma = params_transform['sigFma']

#     grid_y = crop_size_y / stride
#     grid_x = crop_size_x / stride
#     start = stride / 2.0 - 0.5
#     y_range = [i for i in range(int(grid_y))]
#     x_range = [i for i in range(int(grid_x))]
#     xx, yy = np.meshgrid(x_range, y_range)
#     xx = xx * stride + start
#     yy = yy * stride + start
#     d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
#     exponent = d2 / 2.0 / sigma / sigma
#     mask = exponent <= 4.6052
#     cofid_map = np.exp(-exponent)
#     cofid_map = np.multiply(mask, cofid_map)
#     accumulate_confid_map += cofid_map
#     accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

#     return accumulate_confid_map


# def putVecMaps(centerA, centerB, accumulate_vec_map, count, params_transform):
#     '''Parts A Field의 벡터를 구한다'''

#     centerA = centerA.astype(float)
#     centerB = centerB.astype(float)

#     stride = params_transform['stride']
#     crop_size_y = params_transform['crop_size_y']
#     crop_size_x = params_transform['crop_size_x']
#     grid_y = crop_size_y / stride
#     grid_x = crop_size_x / stride
#     thre = params_transform['limb_width']   # limb width
#     centerB = centerB / stride
#     centerA = centerA / stride

#     limb_vec = centerB - centerA
#     norm = np.linalg.norm(limb_vec)
#     if (norm == 0.0):
#         # print 'limb is too short, ignore it...'
#         return accumulate_vec_map, count
#     limb_vec_unit = limb_vec / norm
#     # print 'limb unit vector: {}'.format(limb_vec_unit)

#     # To make sure not beyond the border of this two points
#     min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
#     max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
#     min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
#     max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

 
#     range_x = list(range(int(min_x), int(max_x), 1))
#     range_y = list(range(int(min_y), int(max_y), 1))
#     xx, yy = np.meshgrid(range_x, range_y)
#     ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
#     ba_y = yy - centerA[1]
#     limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
#     mask = limb_width < thre  # mask is 2D

#     vec_map = np.copy(accumulate_vec_map) * 0.0
#     vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
#     vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

#     mask = np.logical_or.reduce(
#         (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

#     accumulate_vec_map = np.multiply(
#         accumulate_vec_map, count[:, :, np.newaxis])
#     accumulate_vec_map += vec_map
#     count[mask == True] += 1

#     mask = count == 0

#     count[mask == True] = 1

#     accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
#     count[mask == True] = 0

#     return accumulate_vec_map, count

# def get_ground_truth(meta, mask_miss):
#     '''
#     annotation과 mask data에서 정답값을 구하도록 한다.
#     '''
    
#     # 초기 설정
#     params_transform = dict()
#     params_transform['stride'] = 8
#     params_transform['mode'] = 5
#     params_transform['crop_size_x'] = 368
#     params_transform['crop_size_y'] = 368
#     params_transform['np'] = 56
#     params_transform['sigma'] = 7.0
#     params_transform['lib_width'] = 1.0

#     stride = params_transform['stride']
#     mode = params_transform['mode']
#     crop_size_y = params_transform['crop_size_y']
#     crop_size_x = params_transform['crop_size_x']
#     num_parts = params_transform['np']
#     nop = meta['numOtherPeople']

#     # 이미지 크기
#     grid_x = crop_size_x / stride
#     grid_y = crop_size_y / stride
#     channels = (num_parts + 1) * 2

#     # 저장할 변수
#     heatmaps = np.zeros((int(grid_y), int(grid_x), 19))
#     pafs = np.zeros((int(grid_y), int(grid_x), 38))

#     mask_miss = cv2.resize(mask_miss, (0, 0), fx=1.0 / stride, fy=1.0 /
#                            stride, interpolation=cv2.INTER_CUBIC).astype(
#         np.float32)
#     mask_miss = mask_miss / 255.
#     mask_miss = np.expand_dims(mask_miss, axis=2)

#     # mask 변수
#     heat_mask = np.repeat(mask_miss, 19, axis = 2)
#     paf_mask = np.repeat(mask_miss, 38, axis=2)

#     # 정확한 좌표 정보를 가우스 분포로 흐릿하게 한다
#     for i in range(18):
#         if (meta['joint_self'][i, 2] <= 1):
#             center = meta['joint_self'][i, :2]
#             gaussian_map = heatmaps[:, :, i]
#             heatmaps[:, :, i] = putGaussianMaps(
#                 center, gaussian_map, params_transform)
#         for j in range(nop):
#             if (meta['joint_others'][j, i, 2] <= 1):
#                 center = meta['joint_others'][j, i, :2]
#                 gaussian_map = heatmaps[:, :, i]
#                 heatmaps[:, :, i] = putGaussianMaps(
#                     center, gaussian_map, params_transform)
#     # pafs
#     mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4,
#              3, 2, 6, 7, 6, 2, 1, 1, 15, 16]

#     mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5,
#              17, 6, 7, 8, 18, 1, 15, 16, 17, 18]

#     thre = 1
#     for i in range(19):
#         # limb

#         count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
#         if (meta['joint_self'][mid_1[i] - 1, 2] <= 1 and meta['joint_self'][mid_2[i] - 1, 2] <= 1):
#             centerA = meta['joint_self'][mid_1[i] - 1, :2]
#             centerB = meta['joint_self'][mid_2[i] - 1, :2]
#             vec_map = pafs[:, :, 2 * i:2 * i + 2]
#             # print vec_map.shape

#             pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
#                                                             centerB=centerB,
#                                                             accumulate_vec_map=vec_map,
#                                                             count=count, params_transform=params_transform)
#         for j in range(nop):
#             if (meta['joint_others'][j, mid_1[i] - 1, 2] <= 1 and meta['joint_others'][j, mid_2[i] - 1, 2] <= 1):
#                 centerA = meta['joint_others'][j, mid_1[i] - 1, :2]
#                 centerB = meta['joint_others'][j, mid_2[i] - 1, :2]
#                 vec_map = pafs[:, :, 2 * i:2 * i + 2]
#                 pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
#                                                                 centerB=centerB,
#                                                                 accumulate_vec_map=vec_map,
#                                                                 count=count, params_transform=params_transform)
#     # background
#     heatmaps[:, :, -
#              1] = np.maximum(1 - np.max(heatmaps[:, :, :18], axis=2), 0.)

#     # Tensorに
#     heat_mask = torch.from_numpy(heat_mask)
#     heatmaps = torch.from_numpy(heatmaps)
#     paf_mask = torch.from_numpy(paf_mask)
#     pafs = torch.from_numpy(pafs)

#     return heat_mask, heatmaps, paf_mask, pafs

# def make_datapath_list(rootpath):
#     """
#     학습 및 검증의 화상 데이터와 어노테이션 데이터, 마스크 데이터의 파일 경로 리스트를 작성한다.
#     """

#     # 어노테이션의 JSON 파일을 읽기
#     json_path = os.path.join(rootpath, 'COCO.json')
#     with open(json_path) as data_file:
#         data_this = json.load(data_file)
#         data_json = data_this['root']

#     # index를 저장
#     num_samples = len(data_json)
#     train_indexes = []
#     val_indexes = []
#     for count in range(num_samples):
#         if data_json[count]['isValidation'] != 0.:
#             val_indexes.append(count)
#         else:
#             train_indexes.append(count)

#     # 화상 파일 경로를 저장
#     train_img_list = list()
#     val_img_list = list()

#     for idx in train_indexes:
#         img_path = os.path.join(rootpath, data_json[idx]['img_paths'])
#         train_img_list.append(img_path)

#     for idx in val_indexes:
#         img_path = os.path.join(rootpath, data_json[idx]['img_paths'])
#         val_img_list.append(img_path)

#     # 마스크 데이터의 경로를 저장
#     train_mask_list = []
#     val_mask_list = []

#     for idx in train_indexes:
#         img_idx = data_json[idx]['img_paths'][-16:-4]
#         anno_path = "./data/mask/train2014/mask_COCO_train2014_" + img_idx+'.jpg'
#         train_mask_list.append(anno_path)

#     for idx in val_indexes:
#         img_idx = data_json[idx]['img_paths'][-16:-4]
#         anno_path = "./data/mask/val2014/mask_COCO_val2014_" + img_idx+'.jpg'
#         val_mask_list.append(anno_path)

#     # 어노테이션 데이터를 저장
#     train_meta_list = list()
#     val_meta_list = list()

#     for idx in train_indexes:
#         train_meta_list.append(data_json[idx])

#     for idx in val_indexes:
#         val_meta_list.append(data_json[idx])

#     return train_img_list, train_mask_list, val_img_list, val_mask_list, train_meta_list, val_meta_list

# class DataTransform():
#     """
#     화상과 마스크, 어노테이션의 전처리 클래스.
#     학습시와 추론시에 서로 다르게 동작한다.
#     학습시에는 데이터 확장을 수행한다.
#     """

#     def __init__(self):

#         self.data_transform = {
#             'train': Compose([
#                 get_anno(),  # JSON에서 어노테이션을 사전에 저장
#                 add_neck(),  # 어노테이션 데이터의 순서를 변경하고, 목의 어노테이션 데이터를 추가
#                 aug_scale(),  # 확대 축소
#                 aug_rotate(),  # 회전
#                 aug_croppad(),  # 자르기
#                 aug_flip(),  # 좌우 반전
#                 remove_illegal_joint(),  # 화상에서 밀려나온 어노테이션을 제거
#                 Normalize_Tensor()  # 색상 정보의 표준화 및 텐서화
#                 # no_Normalize_Tensor()  # 여기서는 색상 정보의 표준화를 생략
#             ]),
#             'val': Compose([
#                 # 검증을 생략
#             ])
#         }

#     def __call__(self, phase, meta_data, img, mask_miss):
#         """
#         Parameters
#         ----------
#         phase : 'train' or 'val'
#             전처리의 모드를 지정.
#         """
#         meta_data, img, mask_miss = self.data_transform[phase](meta_data, img, mask_miss)

#         return meta_data, img, mask_miss

# class COCOkeypointsDataset(data.Dataset):
#     """
#     MSCOCO Cocokeypoints의 Dataset를 작성하는 클래스. PyTorch Dataset 클래스를 상속.

#     Attributes
#     ----------
#     img_list : 리스트
#         화상 경로를 저장한 리스트
#     anno_list : 리스트
#         어노테이션 경로를 저장한 리스트
#     phase : 'train' or 'test'
#         학습 또는 훈련을 설정.
#     transform : object
#         전처리 클래스의 인스턴스
#     """

#     def __init__(self, img_list, mask_list, meta_list, phase, transform):
#         self.img_list = img_list
#         self.mask_list = mask_list
#         self.meta_list = meta_list
#         self.phase = phase
#         self.transform = transform

#     def __len__(self):
#         '''화상 매수를 반환한다'''
#         return len(self.img_list)

#     def __getitem__(self, index):
#         img, heatmaps, heat_mask, pafs, paf_mask = self.pull_item(index)
#         return img, heatmaps, heat_mask, pafs, paf_mask

#     def pull_item(self, index):
#         '''화상의 Tensor 형식 데이터, 어노테이션, 마스크를 취득한다'''

#         # 1. 화상 읽기
#         image_file_path = self.img_list[index]
#         img = cv2.imread(image_file_path)  # [높이][폭][색BGR]

#         # 2. 마스크와 어노테이션 읽기
#         mask_miss = cv2.imread(self.mask_list[index])
#         meat_data = self.meta_list[index]

#         # 3. 화상 전처리
#         meta_data, img, mask_miss = self.transform(
#             self.phase, meat_data, img, mask_miss)

#         # 4. 정답 어노테이션 데이터 취득
#         mask_miss_numpy = mask_miss.numpy().transpose((1, 2, 0))
#         heat_mask, heatmaps, paf_mask, pafs = get_ground_truth(
#             meta_data, mask_miss_numpy)

#         # 5. 마스크 데이터는 RGB가 (1,1,1) 또는 (0,0,0)이므로, 차원을 낮춘다
#         heat_mask = heat_mask[:, :, :, 0]
#         paf_mask = paf_mask[:, :, :, 0]

#         # 6. 채널이 맨 끝에 있으므로, 순서를 변경
#         # 예: paf_mask: torch.Size([46, 46, 38])
#         # → torch.Size([38, 46, 46])
#         paf_mask = paf_mask.permute(2, 0, 1)
#         heat_mask = heat_mask.permute(2, 0, 1)
#         pafs = pafs.permute(2, 0, 1)
#         heatmaps = heatmaps.permute(2, 0, 1)

#         return img, heatmaps, heat_mask, pafs, paf_mask

# 4장 자세 추정의 데이터 확장
# 구현에 일부 참고함
# https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/
# Released under the MIT license

import os
import os.path as osp
import json
from PIL import Image
import cv2
import numpy as np
from scipy import misc, ndimage
import torch
import torch.utils.data as data

from utils import Compose, get_anno, add_neck, aug_scale, aug_rotate, aug_croppad, aug_flip, remove_illegal_joint, Normalize_Tensor, no_Normalize_Tensor


def putGaussianMaps(center, accumulate_confid_map, params_transform):
    '''가우스 맵으로 변환'''
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    stride = params_transform['stride']
    sigma = params_transform['sigma']

    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

    return accumulate_confid_map


def putVecMaps(centerA, centerB, accumulate_vec_map, count, params_transform):
    '''Parts A Field의 벡터를 구한다'''

    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    thre = params_transform['limb_width']   # limb width
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if (norm == 0.0):
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

 
    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[:, :, np.newaxis])
    accumulate_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
    count[mask == True] = 0

    return accumulate_vec_map, count


def get_ground_truth(meta, mask_miss):
    """어노테이션과 마스크 데이터에서 정답을 구한다"""

    # 초기 설정
    params_transform = dict()
    params_transform['stride'] = 8  # 화상 크기를 변경하고 싶지 않다면 1로 한다
    params_transform['mode'] = 5
    params_transform['crop_size_x'] = 368
    params_transform['crop_size_y'] = 368
    params_transform['np'] = 56
    params_transform['sigma'] = 7.0
    params_transform['limb_width'] = 1.0

    stride = params_transform['stride']
    mode = params_transform['mode']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    num_parts = params_transform['np']
    nop = meta['numOtherPeople']

    # 화상 크기
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    channels = (num_parts + 1) * 2

    # 저장할 변수
    heatmaps = np.zeros((int(grid_y), int(grid_x), 19))
    pafs = np.zeros((int(grid_y), int(grid_x), 38))

    mask_miss = cv2.resize(mask_miss, (0, 0), fx=1.0 / stride, fy=1.0 /stride, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mask_miss = mask_miss / 255.
    mask_miss = np.expand_dims(mask_miss, axis=2)

    # 마스크 변수
    heat_mask = np.repeat(mask_miss, 19, axis=2)
    paf_mask = np.repeat(mask_miss, 38, axis=2)

    # 정확한 좌표 정보를 가우스 분포로 흐릿하게 한다
    for i in range(18):
        if (meta['joint_self'][i, 2] <= 1):
            center = meta['joint_self'][i, :2]
            gaussian_map = heatmaps[:, :, i]
            heatmaps[:, :, i] = putGaussianMaps(
                center, gaussian_map, params_transform)
        for j in range(nop):
            if (meta['joint_others'][j, i, 2] <= 1):
                center = meta['joint_others'][j, i, :2]
                gaussian_map = heatmaps[:, :, i]
                heatmaps[:, :, i] = putGaussianMaps(
                    center, gaussian_map, params_transform)
    # pafs
    mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4,
             3, 2, 6, 7, 6, 2, 1, 1, 15, 16]

    mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5,
             17, 6, 7, 8, 18, 1, 15, 16, 17, 18]

    thre = 1
    for i in range(19):
        # limb

        count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
        if (meta['joint_self'][mid_1[i] - 1, 2] <= 1 and meta['joint_self'][mid_2[i] - 1, 2] <= 1):
            centerA = meta['joint_self'][mid_1[i] - 1, :2]
            centerB = meta['joint_self'][mid_2[i] - 1, :2]
            vec_map = pafs[:, :, 2 * i:2 * i + 2]
            # print vec_map.shape

            pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                            centerB=centerB,
                                                            accumulate_vec_map=vec_map,
                                                            count=count, params_transform=params_transform)
        for j in range(nop):
            if (meta['joint_others'][j, mid_1[i] - 1, 2] <= 1 and meta['joint_others'][j, mid_2[i] - 1, 2] <= 1):
                centerA = meta['joint_others'][j, mid_1[i] - 1, :2]
                centerB = meta['joint_others'][j, mid_2[i] - 1, :2]
                vec_map = pafs[:, :, 2 * i:2 * i + 2]
                pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                                centerB=centerB,
                                                                accumulate_vec_map=vec_map,
                                                                count=count, params_transform=params_transform)
    # background
    heatmaps[:, :, -
             1] = np.maximum(1 - np.max(heatmaps[:, :, :18], axis=2), 0.)

    # Tensorに
    heat_mask = torch.from_numpy(heat_mask)
    heatmaps = torch.from_numpy(heatmaps)
    paf_mask = torch.from_numpy(paf_mask)
    pafs = torch.from_numpy(pafs)

    return heat_mask, heatmaps, paf_mask, pafs


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
    
#     # img 경로 저장
#     train_img_list, val_img_list = [],[]

#     for idx in train_indexes:
#         img_path = os.path.join(rootpath, data_json[idx]['img_paths'])   # train_img path store
#         train_img_list.append(img_path)

#     for idx in val_indexes:
#         img_path = os.path.join(rootpath, data_json[idx]['img_paths'])  # val_img path store
#         val_img_list.append(img_path)

#     # mask 경로 저장
#     train_mask_list, val_mask_list = [], []

#     for idx in train_indexes:
#         img_idx = data_json[idx]['img_paths'][-16:-4] # 이미지 파일 이름만 가져오기
#         anno_path = "./data/mask/train2014/mask_COCO_train2014_" + img_idx + '.jpg' # mask file 경로 저장
#         train_mask_list.append(anno_path)

#     for idx in val_indexes:
#         img_idx = data_json[idx]['img_paths'][-16:-4] 
#         anno_path = './data/mask/val2014/mask_COCO_val2014_' + img_idx + '.jpg'
#         val_mask_list.append(anno_path)
    
    
#     # annotation 경로 저장
#     train_meta_list, val_meta_list = [], []

#     for idx in train_indexes:
#         train_meta_list.append(data_json[idx])  # data_json의 모든 value들을 저장
        
#     for idx in val_indexes:
#         val_meta_list.append(data_json[idx])

#     return train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list
from glob import glob
def make_datapath_list(rootpath):
    """
    학습 및 검증의 화상 데이터와 어노테이션 데이터, 마스크 데이터의 파일 경로 리스트를 작성한다.
    """

    # rootpath : COCO.json 파일이 있는 절대경로
    
    # annotaion json file read
    json_path = os.path.join(rootpath, 'COCO.json')
    with open(json_path) as f:
        data = json.load(f)
        data_json = data['root']  # keys : ['root']

    # index 저장
    num_samples = len(data_json) # 전체 파일 개수
    train_indexes, val_indexes = [], []

    for count in range(num_samples):
        if data_json[count]['isValidation'] != 0:  # validation files
            val_indexes.append(count)
        else:
            train_indexes.append(count)  # train files
    
    
    train_img_list, val_img_list = [],[]  # img 경로 저장
    train_mask_list, val_mask_list = [], [] # mask 경로 저장
    train_meta_list, val_meta_list = [], [] # annotation 경로 저장
    
    # COCO.json 파일에 있는 train 이미지가 mask/train2014에는 존재하지 않는 경우가 발생하기 때문에 둘의 교집합 이미지들만 출력하도록 한다. 
    imgs = glob(os.path.join('./data/mask/train2014', '*')) # mask폴더 안에 있는 train 이미지 리스트
    imgs_val = glob(os.path.join('./data/mask/val2014', '*')) # mask폴더 안에 있는 val 이미지 리스트
    
    
    for idx in train_indexes:
        img_path = os.path.join(rootpath, data_json[idx]['img_paths'])   # train_img 경로 저장
        img_idx = data_json[idx]['img_paths'][-16:-4] # 이미지 파일 이름만 가져오기
        anno_path = "./data/mask/train2014/mask_COCO_train2014_" + img_idx + '.jpg' # mask file 경로 저장

        if anno_path in imgs:
            train_img_list.append(img_path)
            train_mask_list.append(anno_path)
            train_meta_list.append(data_json[idx]) # data_json의 모든 value들을 저장
        
    for idx in val_indexes:
        img_path = os.path.join(rootpath, data_json[idx]['img_paths'])  # val_img path store
        img_idx = data_json[idx]['img_paths'][-16:-4] 
        anno_path = './data/mask/val2014/mask_COCO_val2014_' + img_idx + '.jpg'

        if anno_path in imgs_val:
            val_img_list.append(img_path)
            val_mask_list.append(anno_path)
            val_meta_list.append(data_json[idx]) # data_json의 모든 value들을 저장
    
    print('-----------------')
    print('COCO train : ', len(train_indexes))
    print('COCO val : ', len(val_indexes))
    print('mask/train2014 :', len(imgs))
    print('mask/val2014 :', len(imgs_val))
    print('train_img :', len(train_img_list))
    print('train_mask :', len(train_mask_list))
    print('val_img :', len(val_img_list))
    print('val_mask :', len(val_mask_list))

    return train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list


class DataTransform():
    """
    화상과 마스크, 어노테이션의 전처리 클래스.
    학습시와 추론시에 서로 다르게 동작한다.
    학습시에는 데이터 확장을 수행한다.
    """

    def __init__(self):

        self.data_transform = {
            'train': Compose([
                get_anno(),  # JSON에서 어노테이션을 사전에 저장
                add_neck(),  # 어노테이션 데이터의 순서를 변경하고, 목의 어노테이션 데이터를 추가
                aug_scale(),  # 확대 축소
                aug_rotate(),  # 회전
                aug_croppad(),  # 자르기
                aug_flip(),  # 좌우 반전
                remove_illegal_joint(),  # 화상에서 밀려나온 어노테이션을 제거
                Normalize_Tensor()  # 색상 정보의 표준화 및 텐서화
                # no_Normalize_Tensor()  # 여기서는 색상 정보의 표준화를 생략
            ]),
            'val': Compose([
                get_anno(),  # JSON에서 어노테이션을 사전에 저장
                add_neck(),  # 어노테이션 데이터의 순서를 변경하고, 목의 어노테이션 데이터를 추가
                aug_scale(),  # 확대 축소
                aug_rotate(),  # 회전
                aug_croppad(),  # 자르기
                aug_flip(),  # 좌우 반전
                remove_illegal_joint(),  # 화상에서 밀려나온 어노테이션을 제거
                Normalize_Tensor()  # 색상 정보의 표준화 및 텐서화
            ])
        }

    def __call__(self, phase, meta_data, img, mask_miss):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            전처리의 모드를 지정.
        """
        meta_data, img, mask_miss = self.data_transform[phase](
            meta_data, img, mask_miss)

        return meta_data, img, mask_miss


class COCOkeypointsDataset(data.Dataset):
    """
    MSCOCO Cocokeypoints의 Dataset를 작성하는 클래스. PyTorch Dataset 클래스를 상속.

    Attributes
    ----------
    img_list : 리스트
        화상 경로를 저장한 리스트
    anno_list : 리스트
        어노테이션 경로를 저장한 리스트
    phase : 'train' or 'test'
        학습 또는 훈련을 설정.
    transform : object
        전처리 클래스의 인스턴스
    """

    def __init__(self, img_list, mask_list, meta_list, phase, transform):
        self.img_list = img_list
        self.mask_list = mask_list
        self.meta_list = meta_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''화상 매수를 반환한다'''
        return len(self.img_list)

    def __getitem__(self, index):
        
        img, heatmaps, heat_mask, pafs, paf_mask = self.pull_item(index)
        return img, heatmaps, heat_mask, pafs, paf_mask

    def pull_item(self, index):
        '''화상의 Tensor 형식 데이터, 어노테이션, 마스크를 취득한다'''
        
        # print('-------------------')
        # print(index)
        # print(len(self.img_list))
        # print(len(self.mask_list))

        # 1. 화상 읽기
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [높이][폭][색BGR]


        # 2. 마스크와 어노테이션 읽기
        mask_miss = cv2.imread(self.mask_list[index])
        meat_data = self.meta_list[index]

            
        # 3. 화상 전처리
        meta_data, img, mask_miss = self.transform(self.phase, meat_data, img, mask_miss)
        
        
        
        # 4. 정답 어노테이션 데이터 취득
        if isinstance(mask_miss, np.ndarray):
            mask_miss_numpy = mask_miss.transpose((1, 2, 0))
        else:
            mask_miss_numpy = mask_miss.numpy().transpose((1, 2, 0))
        
        # print(mask_miss_numpy)
        # print(mask_miss_numpy.shape)
        # print('img shape: ',img.shape)
        # print('mask_miss shape : ',mask_miss.shape)
        heat_mask, heatmaps, paf_mask, pafs = get_ground_truth(meta_data, mask_miss_numpy)

        # 5. 마스크 데이터는 RGB가 (1,1,1) 또는 (0,0,0)이므로, 차원을 낮춘다
        heat_mask = heat_mask[:, :, :, 0]
        paf_mask = paf_mask[:, :, :, 0]

        # 6. 채널이 맨 끝에 있으므로, 순서를 변경
        # 예: paf_mask: torch.Size([46, 46, 38])
        # → torch.Size([38, 46, 46])
        paf_mask = paf_mask.permute(2, 0, 1)
        heat_mask = heat_mask.permute(2, 0, 1)
        pafs = pafs.permute(2, 0, 1)
        heatmaps = heatmaps.permute(2, 0, 1)

        return img, heatmaps, heat_mask, pafs, paf_mask
