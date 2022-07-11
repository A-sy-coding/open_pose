from utils import DataTransform # 전처리 수행해주는 클래스
from utils import make_datapath_list # json 파일 가져오는 클래스
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

'''
이미지, 마스크, annotaion 전처리 수행 클래스
학습시에 데이터 확장 수행 (data augument)
'''
# 데이터 가져오기
train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list('./data')

# 샘플 이미지 전처리 수행
index = 24
img = cv2.imread(val_img_list[index])
mask_miss = cv2.imread(val_mask_list[index])
meta_data = val_meta_list[index]

# 이미지 전처리 --> 미리 만들어 놓은 DataTransform 사용
transform = DataTransform()
meta_data, img, mask_miss = transform("train", meta_data, img, mask_miss)

# 이미지 표시
img = img.numpy().transpose((1,2,0)) # RGB로 변경
# plt.imshow(img)
# plt.savefig('./sample_img/preprocessing_img.jpg')

# warning : Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
# plt.imshow((img * 255).astype(np.uint8))
# plt.savefig('./sample_img/preprocessing_img.jpg')
# cv2.imwrite('./sample_img/preprocessing_img.jpg', img)

# 마스크 표시
mask_miss = mask_miss.numpy().transpose((1,2,0))
# plt.imshow((mask_miss * 255).astype(np.uint8))
# plt.savefig('./sample_img/preprocessing_mask.jpg')
# cv2.imwrite('./sample_img/preprocessing_mask.jpg', mask_miss)

# 이미지 + 마스크
img = Image.fromarray(np.uint8(img*255))
img = np.asarray(img.convert('RGB'))

mask_miss = Image.fromarray(np.uint8(mask_miss))
mask_miss = np.asarray(mask_miss.convert('RGB'))

blend_img = cv2.addWeighted(img, 0.4, mask_miss, 0.6, 0)
# plt.imshow(blend_img)
# plt.savefig('./sample_img/preprocessing_blend.jpg')
# cv2.imwrite('./sample_img/preprocessing_blend.jpg', blend_img)
