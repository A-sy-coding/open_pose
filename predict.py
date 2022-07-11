from utils.openpose_net import OpenPoseNet
import argparse
import torch
from PIL import Image
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from utils.decode_pose import decode_pose

def img_preprocess(img):
    oriImg = cv2.imread(img)  # B,G,R의 순서

    # BGR을 RGB로 하여 표시
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    # plt.imshow(oriImg)
    # plt.show()

    # img resize
    size = (368, 368)
    img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)

    # 이미지 전처리
    img = img.astype(np.float32) / 255.

    # 색상 정보의 표준화
    color_mean = [0.485, 0.456, 0.406]
    color_std = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]  # BGR→RGB

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

    # (높이, 폭, 색) → (색, 높이, 폭)
    img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

    # 이미지를 Tensor로 변경 후 미니 배치화: torch.Size([1, 3, 368, 368])
    img = torch.from_numpy(img)
    x = img.unsqueeze(0)

    return x

def get_pafs_heatmaps(model, preprocess_img, img):  # net, x, img
    
    # 원본 이미지
    oriImg = cv2.imread(img)  # B,G,R의 순서
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    
    model.eval()
    predicted_outputs, _ = model(preprocess_img)  # (out6_1, out6_2), saved_for_loss --> pafs와 heatmaps 반환

    # 이미지를 텐서에서 NumPy로 변환해, 크기를 반환합니다
    pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
    heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)
    
    size = (368, 368)
    pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)

    pafs = cv2.resize(pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    return pafs, heatmaps
    
def test(img, pafs, heatmaps):
    # 원본 이미지
    oriImg = cv2.imread(img)  # B,G,R의 순서
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    
    # 왼쪽 팔꿈치와 왼쪽 손목의 heatmap, 그리고 왼쪽 팔꿈치와 왼쪽 손목을 잇는 PAF의 x 벡터를 시각화한다
    # 왼쪽 팔꿈치
    heat_map = heatmaps[:, :, 6]  # 6은 왼쪽 팔꿈치
    heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
    heat_map = np.asarray(heat_map.convert('RGB'))

    # 합성해서 표시
    blend_img = cv2.addWeighted(oriImg, 0.5, heat_map, 0.5, 0)
    plt.imshow(blend_img)
    plt.savefig('sample_img/left_arm.jpg')


    # 왼쪽 손목
    heat_map = heatmaps[:, :, 7]  # 7은 왼쪽 손목
    heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
    heat_map = np.asarray(heat_map.convert('RGB'))

    # 합성해서 표시
    blend_img = cv2.addWeighted(oriImg, 0.5, heat_map, 0.5, 0)
    plt.imshow(blend_img)
    plt.savefig('sample_img/left_hands.jpg')


    # 왼쪽 팔꿈치와 왼쪽 손목을 잇는 PAF의 x 벡터
    paf = pafs[:, :, 24]
    paf = Image.fromarray(np.uint8(cm.jet(paf)*255))
    paf = np.asarray(paf.convert('RGB'))

    # 합성해서 표시
    blend_img = cv2.addWeighted(oriImg, 0.5, paf, 0.5, 0)
    plt.imshow(blend_img)
    plt.savefig('sample_img/left_arm_hands.jpg')
    
def pose_test(img, pafs, heatmaps):
    # 원본 이미지
    oriImg = cv2.imread(img)  # B,G,R의 순서
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

    # pose 연결
    _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)  # to_plot, canvas, joint_list, person_to_joint_assoc
    
    plt.imshow(result_img)
    plt.savefig('sample_img/pose_result.jpg')

def run(weights):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 모델 정의 
    net = OpenPoseNet()
    
    if device == 'cuda:0':
        net_weights = torch.load(weights)
    else:
        net_weights = torch.load(weights, map_location=device)
    
    keys = list(net_weights.keys())

    weights_load = {}

    # 로드한 내용을 이 책에서 구축한 모델의
    # 파라미터명 net.state_dict().keys()로 복사
    for i in range(len(keys)):
        weights_load[list(net.state_dict().keys())[i]] = net_weights[list(keys)[i]]

    # 복사한 내용을 모델에 할당
    state = net.state_dict()
    state.update(weights_load)
    net.load_state_dict(state)

    path = './data/hit-1407826_640.jpg'
    
    # 이미지 전처리
    preprocess_img = img_preprocess(path)
    # pafs, heatmaps 구하기
    pafs, heatmaps = get_pafs_heatmaps(net, preprocess_img, path)
    # test 해보기
    test(path, pafs, heatmaps)
    # pose test 해보기
    pose_test(path, pafs, heatmaps)


def parse_opt():
    parser = argparse.ArgumentParser(description='새로운 이미지 테스트')
    parser.add_argument('--weights', type=str, required=True, help = '학습시킨 pth 파일 경로')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
