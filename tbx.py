from utils.openpose_net import OpenPoseNet
from tensorboardX import SummaryWriter
import torch

# 모델
net = OpenPoseNet()
net.train() # 학습 모드

# writer 객체 생성
writer = SummaryWriter('./tensorboard')  # tensorboard 폴더에 값들 저장

batch_size = 2
dummy_img = torch.rand(batch_size, 3, 368, 368)

# openpose 신경망에 더미 데이터를 전달할 때 계산 그래프를 writer에 저장
writer.add_graph(net, (dummy_img,))
writer.close()