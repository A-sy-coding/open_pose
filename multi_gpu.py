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
from utils.openpose_net import OpenPoseNet
from utils.openpose_loss import OpenPoseLoss
import os, sys, argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

# 디렉토리 만들기
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error : Creating directory' + directory)

def data_loader(train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list, num_worker, batch_size):
    # 훈련, 검증 데이터 가져오기 --> 전처리도 같이 수행
    train_dataset = COCOkeypointsDataset(train_img_list, train_mask_list, train_meta_list, phase='train', transform=DataTransform())
    val_dataset = COCOkeypointsDataset(val_img_list, val_mask_list, val_meta_list, phase='val', transform=DataTransform())

    train_sampler = data.distributed.DistributedSampler(train_dataset) 
    # val_sampler = data.distributed.DistributedSampler(val_dataset) 
    
    shuffle = False
    pin_memory = True
    
    # dataloader
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=pin_memory,num_workers=num_worker, shuffle=shuffle, sampler=train_sampler)
    # val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, pin_memory=pin_memory,num_workers=num_worker, shuffle=shuffle, sampler=val_sampler)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}
    return dataloaders_dict


def main_worker(gpu, ngpus_per_node):
 
    train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list(rootpath = './data')
    
    batch_size = 64
    num_worker = ngpus_per_node * 4
    num_epochs = 2
 
    batch_size = int(batch_size / ngpus_per_node)
    num_worker = int(num_worker / ngpus_per_node)
    
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)
    
    net = OpenPoseNet()
    torch.cuda.set_device(gpu)
    net = net.cuda(gpu)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu])
 
    dataloaders_dict = data_loader(train_img_list, val_img_list, train_mask_list, val_mask_list, train_meta_list, val_meta_list, num_worker, batch_size)
    print('------------------')
    print('gpu : ', gpu)
    print('------------------')
    # 최적화
    optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0001)
    criterion = OpenPoseLoss().to(gpu) # loss 설정
 
 
    # 이미지 정보
    num_train_imgs = len(dataloaders_dict['train'].dataset)
    num_val_imgs = len(dataloaders_dict['val'].dataset)
    # batch_size = dataloaders_dict['train'].batch_size

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
                images = images.to(gpu, dtype=torch.float64)
                images = images.type(torch.FloatTensor).to(gpu)
                
                haetmap_target = heatmap_target.to(gpu,  dtype=torch.float64)
                heatmap_target = heatmap_target.type(torch.FloatTensor).to(gpu)
                
                heat_mask = heat_mask.to(gpu, dtype=torch.float64)
                heat_mask = heat_mask.type(torch.FloatTensor).to(gpu)
                
                paf_target = paf_target.to(gpu,  dtype=torch.float64)
                paf_target = paf_target.type(torch.FloatTensor).to(gpu)
                
                paf_mask = paf_mask.to(gpu, dtype=torch.float64)
                paf_mask = paf_mask.type(torch.FloatTensor).to(gpu)
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
    torch.save(net.state_dict(), 'weights/multigpu_openpose_net_' + str(epoch+1) + '.pth')


# option에는 캔인지 페트병인지가 들어간다. (can or pet)
def run():
    # 필요한 데이터 경로 수집 
    '''
    COCO.json에 존재하는 파일들이 mask/train2014 or mask/val2014에 존재하지 않을 수도 있기 때문에 존재하는 파일들만 가져오기
    '''
    

    # gpu 사용 설정
    ngpus_per_node = torch.cuda.device_count()  # 사용 가능한 gpu 개수
    world_size = ngpus_per_node
 
    
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, ))

    



def parse_opt():
    parser = argparse.ArgumentParser(description="Openpose train - multi-gpu")
    # parser.add_argument('--path', type=str, required=True, help='data 폴더 위치 -> json파일이 있는 위치')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)