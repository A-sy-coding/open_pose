'''
data_load.py 파일은 COCO.json 파일을 이용하여 train 데이터와 valid 데이터를 분리하고,
이미지 데이터, 마스크 데이터, anotation 데이터들을 각각 리스트로 저장하도록 한다.
'''
import json, os
# make_data_list의 코드를 변경해야 될것 같다. --> train_img_list에 존재하는 파일 이름이 train_mask_list에는 존재하지 않는 경우가 발생
# 따라서, train_img_list가 train_mask_list와 같지 않으면 저장하지 않도록 한다.

def make_data_list(rootpath):
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

    # img 경로 저장
    train_img_list, val_img_list = [],[]

    for idx in train_indexes:
        img_path = os.path.join(rootpath, data_json[idx]['img_paths'])   # train_img path store
        train_img_list.append(img_path)

    for idx in val_indexes:
        img_path = os.path.join(rootpath, data_json[idx]['img_paths'])  # val_img path store
        val_img_list.append(img_path)

    # mask 경로 저장
    train_mask_list, val_mask_list = [], []

    for idx in train_indexes:
        img_idx = data_json[idx]['img_paths'][-16:-4] # 이미지 파일 이름만 가져오기
        anno_path = "./data/mask/train2014/mask_COCO_train2014_" + img_idx + '.jpg' # mask file 경로 저장
        train_mask_list.append(anno_path)

    for idx in val_indexes:
        img_idx = data_json[idx]['img_paths'][-16:-4] 
        anno_path = './data/mask/val2014/mask_COCO_val2014_' + img_idx + '.jpg'
        val_mask_list.append(anno_path)

    # annotation 경로 저장
    train_meta_list, val_meta_list = [], []

    for idx in train_indexes:
        train_meta_list.append(data_json[idx])  # data_json의 모든 value들을 저장
        
    for idx in val_indexes:
        val_meta_list.append(data_json[idx])

    return train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list

train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list = make_data_list('./data')
print(len(train_img_list))
print(len(train_mask_list))
print(len(train_meta_list))

###########################
# 데이터 동작 확인
##########################
import cv2
import matplotlib.pyplot as plt
from PIL import Image

index = 24

img = cv2.imread(val_img_list[index])  # 이미지 읽기
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB로 이미지 바꾸기

plt.imshow(img)
plt.savefig('sample_img.jpg')

mask_miss = cv2.imread(val_mask_list[index])  # mask 이미지 읽기
mask_miss = cv2.cvtColor(mask_miss, cv2.COLOR_BGR2RGB)

plt.imshow(mask_miss)
plt.savefig('sample_mask_img.jpg')

blend_img = cv2.addWeighted(img, 0.4, mask_miss, 0.6, 0) # 0.4, 0.6은 투명도 가중치를 의미한다. 클수록 진해진다.
                                                         # blend = alpha * img1 + beta * img2 + gamma
plt.imshow(blend_img)                                                         
plt.savefig('sample_blend_img.jpg')

