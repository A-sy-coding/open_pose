import random
import math
import time, os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import make_datapath_list, DataTransform, COCOkeypointsDataset

# 디렉토리 만들기
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error : Creating directory' + directory)

# 파일 경로 리스트
train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list(rootpath = './data')
# train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list = train_img_list[:500], val_img_list[:500], train_mask_list[:500], val_mask_list[:500], train_meta_list[:500], val_meta_list[:500]

# 훈련, 검증 데이터 가져오기 --> 전처리도 같이 수행
train_dataset = COCOkeypointsDataset(train_img_list, train_mask_list, train_meta_list, phase='train', transform=DataTransform())
val_dataset = COCOkeypointsDataset(val_img_list, val_mask_list, val_meta_list, phase='val', transform=DataTransform())


# dataloader
batch_size = 32
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}

# print(dataloaders_dict)

from utils.openpose_net import OpenPoseNet
# model
net = OpenPoseNet()

# loss calss 설정
class OpenPoseLoss(nn.Module):
    def __init__(self):
        super(OpenPoseLoss, self).__init__()

    def forward(self, saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask):
        '''
        손실함수 계산
        heatmap_target : [batch, 19,46,46]
        heatmap_mask : [batch, 19, 46, 46]
        paf_target : [batch, 38, 46, 46]
        paf_mask : [batch, 38, 46, 46]
        '''
        total_loss = 0

        # stage마다 계산
        for i in range(6): # stage는 1~6
            # heatmap에서 mask된 부분은 무시
            
            # pafs
            pred1 = saved_for_loss[2*i] * paf_mask
            gt1 = paf_target.float() * paf_mask

            # heatmap
            pred2 = saved_for_loss[2*i+1] * heat_mask
            gt2 = heatmap_target.float() * heat_mask

            total_loss += F.mse_loss(pred1, gt1, reduction='mean') + \
                          F.mse_loss(pred2, gt2, reduction='mean')

        return total_loss

criterion = OpenPoseLoss() # loss 설정

# 최적화
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0001)

# 모델 학습
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    #gpu 확인
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('사용 장치 :', device)

    
    net.to(device)
    torch.backends.cudnn.benchmark = True  # 네트원크가 어느 정도 고정되면 고속화시킨다.

    # 이미지 정보
    num_train_imgs = len(dataloaders_dict['train'].dataset)
    num_val_imgs = len(dataloaders_dict['val'].dataset)
    batch_size = dataloaders_dict['train'].batch_size

    iteration = 1 # 반복 횟수
    
    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epochs_train_loss = 0.0 # 손실 값
        epochs_val_loss = 0.0

        print('--------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('--------------')

        # 에포크별 훈련 및 검증
        for phase in ['train','val']:
            if phase == 'train':
                net.train() # 훈련 모드
                optimizer.zero_grad()
                print(' (train) ')
            else:
                net.eval()  # 검증 모드
                print('-------------')
                print(' (val) ')

            # 배치 사이즈 만큼 데이터 가져오기
            for images, heatmap_target, heat_mask, paf_target, paf_mask in tqdm(dataloaders_dict[phase]):
                if images.size()[0] == 1: # 1이면 배치 정규화에서 오류가 발생
                    continue
                
                # gpu로 데이터 보내기
                images = images.to(device, dtype=torch.float64)
                images = images.type(torch.FloatTensor).to(device)
                
                haetmap_target = heatmap_target.to(device,  dtype=torch.float64)
                heatmap_target = heatmap_target.type(torch.FloatTensor).to(device)
                
                heat_mask = heat_mask.to(device, dtype=torch.float64)
                heat_mask = heat_mask.type(torch.FloatTensor).to(device)
                
                paf_target = paf_target.to(device,  dtype=torch.float64)
                paf_target = paf_target.type(torch.FloatTensor).to(device)
                
                paf_mask = paf_mask.to(device, dtype=torch.float64)
                paf_mask = paf_mask.type(torch.FloatTensor).to(device)
                # print(images.dtype)
                # print(heatmap_target.dtype)
                # print(heat_mask.dtype)
                # print(paf_target.dtype)
                # print(paf_mask.dtype)
                # print(images.type())
                # print(heatmap_target.type())
                # print(heat_mask.type())
                # print(paf_target.type())
                # print(paf_mask.type())
                optimizer.zero_grad() # 기울기 초기화

                # 순전파 수행
                with torch.set_grad_enabled(phase=='train'):  # 범위 지정
                    _, saved_for_loss = net(images)

                    loss = criterion(saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask)
                    del saved_for_loss 

                    # 훈련 시에는 역전파 수행
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10번 반복당 loss값 표기
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start  # 반복당 걸린 시간
                            # print('반복 {} || Loss: {:.4f} || 10iter : {:.4f} sec.'.format(iteration, loss.item()/batch_size, duration))
                    
                        epochs_train_loss += loss.item()
                        iteration += 1
                    else:
                        epochs_val_loss += loss.item()
        
        # epoch의 phase별 손실과 정답률
        t_epoch_finish = time.time()
        print('--------------')
        print('epoch {} || Epoch_TRAIN_Loss : {:.4f} || Epoch_VAL_Loss : {:.4f}'.format(epoch+1, epochs_train_loss/num_train_imgs, epochs_val_loss/num_val_imgs))
        print('timer : {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    # 학습된 모델 저장
    createFolder('weights')
    torch.save(net.state_dict(), 'weights/openpose_net_' + str(epoch+1) + '.pth')


# 학습 진행하기
num_epochs = 2  # 2번 반복
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs = num_epochs)