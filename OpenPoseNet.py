'''
OpenPose의 신경망은 Feature 모듈과 Stage 모듈로 구분된다.
Feature 모듈은 이미지의 특징점들을 추출하고, Stage 모듈은 Pafs와 heatmaps를 추출하도록 한다.
이후 Stage단계를 계속 반복하여 정확도를 높이도록 한다.
'''

import torch
import torch.nn as nn
from torch.nn import init
import torchvision


class OpenPoseNet(nn.Module):
    def __init__(self):
        super(OpenPoseNet, self).__init__()

        # Feature모듈
        self.model0 = OpenPose_Feature()

        # Stage모듈
        # PAFs(Part Affinity Fields)측
        self.model1_1 = make_OpenPose_block('block1_1')
        self.model2_1 = make_OpenPose_block('block2_1')
        self.model3_1 = make_OpenPose_block('block3_1')
        self.model4_1 = make_OpenPose_block('block4_1')
        self.model5_1 = make_OpenPose_block('block5_1')
        self.model6_1 = make_OpenPose_block('block6_1')

        # confidence heatmap측
        self.model1_2 = make_OpenPose_block('block1_2')
        self.model2_2 = make_OpenPose_block('block2_2')
        self.model3_2 = make_OpenPose_block('block3_2')
        self.model4_2 = make_OpenPose_block('block4_2')
        self.model5_2 = make_OpenPose_block('block5_2')
        self.model6_2 = make_OpenPose_block('block6_2')

    def forward(self, x):
        """순전파 정의"""

        # Feature모듈
        out1 = self.model0(x)

        # Stage1
        out1_1 = self.model1_1(out1)  # PAFs측
        out1_2 = self.model1_2(out1)  # confidence heatmap측

        # CStage2
        out2 = torch.cat([out1_1, out1_2, out1], 1)  # 1차원의 채널 결합
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)

        # Stage3
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)

        # Stage4
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)

        # Stage5
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)

        # Stage6
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        # 손실의 계산을 위해 각 Stage 결과를 저장
        saved_for_loss = []
        saved_for_loss.append(out1_1)  # PAFs측
        saved_for_loss.append(out1_2)  # confidence heatmap측
        saved_for_loss.append(out2_1)
        saved_for_loss.append(out2_2)
        saved_for_loss.append(out3_1)
        saved_for_loss.append(out3_2)
        saved_for_loss.append(out4_1)
        saved_for_loss.append(out4_2)
        saved_for_loss.append(out5_1)
        saved_for_loss.append(out5_2)
        saved_for_loss.append(out6_1)
        saved_for_loss.append(out6_2)

        # 최종적인 PAFs의 out6_1과 confidence heatmap의 out6_2, 그리고
        # 손실 계산용으로 각 단계에서 PAFs와 heatmap을 저장한 saved_for_loss를 출력
        # out6_1: torch.Size([minibatch, 38, 46, 46])
        # out6_2: torch.Size([minibatch, 19, 46, 46])
        # saved_for_loss:[out1_1, out_1_2, ・・・, out6_2]

        return (out6_1, out6_2), saved_for_loss


class OpenPose_Feature(nn.Module):
    def __init__(self):
        super(OpenPose_Feature, self).__init__()

        # VGG-19의 최초 10개의 합성곱을 사용
        # 처음 실행할 때에는 학습된 파라미터를 다운로드하므로 실행에 시간이 걸립니다
        vgg19 = torchvision.models.vgg19(pretrained=True)
        model = {}
        model['block0'] = vgg19.features[0:23]  # VGG-19의 최초 10개의 합성곱 층까지

        # 나머지는 새로운 합성곱 층을 2개 준비
        model['block0'].add_module("23", torch.nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1))
        model['block0'].add_module("24", torch.nn.ReLU(inplace=True))
        model['block0'].add_module("25", torch.nn.Conv2d(
            256, 128, kernel_size=3, stride=1, padding=1))
        model['block0'].add_module("26", torch.nn.ReLU(inplace=True))

        self.model = model['block0']

    def forward(self, x):
        outputs = self.model(x)
        return outputs


def make_OpenPose_block(block_name):
    """
    구성 변수에서 OpenPose의 Stage모듈의 block을 작성
    nn.Module이 아니라, nn.Sequential로 한다
    """

    # 1. 구성 사전 변수 blocks을 작성하여, 네트워크를 생성시킨다
    # 먼저 전 패턴의 사전을 준비하여, block_name 인수만을 생성한다
    blocks = {}
    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]

    # block_name 인수의 구성 사전을 꺼낸다
    cfg_dict = blocks[block_name]

    # 2. 구성 내용을 리스트 변수 layers에 저장
    layers = []

    # 0번째부터 최후의 층까지 작성
    for i in range(len(cfg_dict)):
        for k, v in cfg_dict[i].items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]

    '''
    make_OpenPose_block('block1_1')를 수행하면 밑과 같이 나온다.    
    [Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), ReLU(inplace=True), Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), ReLU(inplace=True), Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), ReLU(inplace=True), Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1)), ReLU(inplace=True), Conv2d(512, 38, kernel_size=(1, 1), stride=(1, 1)), ReLU(inplace=True)]
    '''

    # 3. layers를 Sequential로 한다
    # 단, 최후에 ReLU는 필요 없으므로 직전까지를 사용한다
    net = nn.Sequential(*layers[:-1])

    # 4. 초기화 함수를 설정하여, 합성곱 층을 초기화한다
    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    net.apply(_initialize_weights_norm)

    return net

#########################
# 실행 확인
########################
net = OpenPoseNet()
net.train()

batch_size = 2
dummy_img = torch.rand(batch_size, 3, 368, 368)
# print(dummy_img)
outputs = net(dummy_img)
print(outputs)