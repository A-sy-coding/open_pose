# COCO.json 파일은 현재 task에 맞춰 값들을 모아놓은 파일이다.
# 실제 COCO annotation 데이터 셋들을 종합하여 만든 파일이다.
import json
import os
import numpy as np

# path  = os.path.join('data', 'annotations', 'person_keypoints_val2014.json')
path  = os.path.join('data', 'COCO.json')  # COCO.json파일 확인
with open(path) as f:
    data = json.load(f)
    
print(data.keys())
print(data['root'][0])
# print(data['info'])
# print(data['annotations'][0])
# print(data['categories'][0])
# print(data['images'][0])
# print(data.values())


########################
# COCO.json 파일 형식 
########################

'''
COCO.json 파일의 keys : 'root'만 존재
{'dataset': 'COCO', 
 'isValidation': 0.0, 
 'img_paths': 'train2014/COCO_train2014_000000000036.jpg', 
 'img_width': 481.0, 'img_height': 640.0, 
 'objpos': [322.885, 395.485], 
 'image_id': 36.0, 
 'bbox': [167.58, 162.89, 310.61, 465.19], 
 'segment_area': 86145.297, 
 'num_keypoints': 13.0, 
 'joint_self': [[250.0, 244.0, 1.0], [265.0, 223.0, 1.0], [235.0, 235.0, 1.0], [309.0, 227.0, 1.0], [235.0, 253.0, 1.0], [355.0, 337.0, 1.0], [215.0, 342.0, 1.0], [407.0, 494.0, 1.0], [213.0, 520.0, 1.0], [445.0, 617.0, 1.0], [244.0, 447.0, 1.0], [338.0, 603.0, 1.0], [267.0, 608.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]], 
 'scale_provided': 1.264, 
 'joint_others': [], 
 'annolist_index': 1.0, 
 'people_index': 1.0, 
 'numOtherPeople': 0.0, 
 'scale_provided_other': {'_ArrayType_': 'double', '_ArraySize_': [0, 0], '_ArrayData_': None}, 
 'objpos_other': {'_ArrayType_': 'double', '_ArraySize_': [0, 0], '_ArrayData_': None},
 'bbox_other': {'_ArrayType_': 'double', '_ArraySize_': [0, 0], '_ArrayData_': None}, 
 'segment_area_other': {'_ArrayType_': 'double', '_ArraySize_': [0, 0], '_ArrayData_': None}, 
 'num_keypoints_other': {'_ArrayType_': 'double', '_ArraySize_': [0, 0], '_ArrayData_': None}}
'''
