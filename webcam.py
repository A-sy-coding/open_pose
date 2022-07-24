import cv2
from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose
import numpy as np
import torch
import math
def img_preprocess(img):
    # oriImg = cv2.imread(img)  # B,G,R의 순서

    # BGR을 RGB로 하여 표시
    oriImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    # oriImg = cv2.imread(img)  # B,G,R의 순서
    oriImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
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
    

    
def pose_test(img, pafs, heatmaps):
    # 원본 이미지
    # oriImg = cv2.imread(img)  # B,G,R의 순서
    oriImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pose 연결
    _, result_img, _, _ ,joint_lmList= decode_pose(oriImg, heatmaps, pafs)  # to_plot, canvas, joint_list, person_to_joint_assoc, joint_coords
    
    return result_img, joint_lmList


# angle 출력 --> img는 pose_estimator출력한 이미지, ,(p1,p2,p3) : 관절 위치
def findAngle(img,joint_dict, p1, p2, p3, draw=True):   
        #Get the landmarks
        x1,y1 = map(int,joint_dict[p1][0])
        x2,y2 = map(int,joint_dict[p2][0])
        x3,y3 = map(int,joint_dict[p3][0])        
        
        # print(x1, x2, x3)
        # print(y1, y2, y3)
        # radians를 degree로 변환
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

        # print('변환 전 :', angle)
        if angle < 0 :
            angle +=  360  # 값이 360도라는 범위 안에 들게 한다.
            if angle > 180:
                angle = 360 - angle # 값의 범위를 180도 안으로 설정하도록 한다.
        elif angle > 180:
            angle = 360 - angle
        # print('변환 후 :', angle)
        
        #Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)

            
            cv2.circle(img, (x1, y1), 5, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0,0,255), 2)
            cv2.circle(img, (x2, y2), 5, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0,0,255), 2)
            cv2.circle(img, (x3, y3), 5, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0,0,255), 2)
            
            ############# 확인 -> 관절 위치 + x좌표 + y좌표 ###############
            # cv2.putText(img, str(int(p1))+ ' ' + str(x1) + ' , ' + str(y1), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            # cv2.putText(img, str(int(p2))+ ' ' + str(x2) + ' , ' + str(y2), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            # cv2.putText(img, str(int(p3))+ ' ' + str(x3) + ' , ' + str(y3), (x3, y3), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
            ##################################
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        return angle



def push_up(net):
    count = 0
    direction = 0
    form = 0
    feedback = "Fix Form"

    # webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, oriImg = video_capture.read() 
        
        # 이미지 전처리
        preprocess_img = img_preprocess(oriImg)
        # pafs, heatmaps 구하기
        pafs, heatmaps = get_pafs_heatmaps(net, preprocess_img, oriImg)
        # pose test 해보기
        out, joint_lmList = pose_test(oriImg, pafs, heatmaps)
        
        # joint_lmList 전처리 --> 중복된 값 없게 설정
        unique_joint = []  # 중복되지 않는 값들을 다시 정의
        joint_lmList_to_list = [list(i) for i in joint_lmList]
        for joint in joint_lmList_to_list:
            if joint in unique_joint:
                continue
            else:
                unique_joint.append(joint)

        joint_dict = {} # 중복되지 않는 값들을 다시 정의 --> dict형태로 --> 관절 위치를 key값으로 설정
        for joint in unique_joint:
            if int(joint[-1]) in joint_dict:
                joint_dict[int(joint[-1])].append(joint[0:2])
            else:
                joint_dict[int(joint[-1])] = [joint[0:2]]
        
        # elbow, shoulder, hip
        '''
        (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
        5-'left_shoulder' 6-'left_elbow'	7-'left_wrist'  8-'right_hip'
        9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
        13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
        17-'left_ear' )
        '''
        
        '''
        img -> pose 연결 한 후 반환한 이미지 -> pose_test 반환 값
        '''
        
        need_pose = [5,6,7,11,12] # push-up을 detect하기 위해 필요한 pose 위치
        
        flag = 1
        for pose in need_pose:  # joint_dict.keys()에 필요한 pose 위치가 없으면 수행 x
            if pose not in joint_dict.keys():
                flag=0
                break
        
        if len(joint_dict) != 0 and flag == 1:

            elbow = findAngle(out,joint_dict, 5, 6, 7)
            shoulder = findAngle(out,joint_dict, 6, 5, 11)
            hip = findAngle(out,joint_dict, 5, 11,12)
            
            # elbow = findAngle(out,joint_dict, 2, 3, 4)
            # shoulder = findAngle(out,joint_dict, 3, 2, 8)
            # hip = findAngle(out,joint_dict, 2, 8, 9)


            # push up 성공 퍼센트 
            per = np.interp(elbow, (90, 160), (0, 100))

            # push up 진행정도를 보여주는 bar
            bar = np.interp(elbow, (0,160), (380,50))

            # push up 시작 자세 check
            if elbow > 160 and shoulder > 40 and hip > 160:
                form = 1
            
            # push up 전체 모션을 check
            if form == 1:
                if per == 0:
                    if elbow <=90 and hip > 160:
                        feadback = 'Up'
                        if direction == 0:
                            count += 0.5
                            direction = 1 # 방향 전환
                    else:
                        feedback = 'Fix Form'
                
                if per == 100:
                    if elbow > 160 and shoulder > 40 and hip > 160:
                        feedback = 'Down'
                        if direction == 1:
                            count += 0.5
                            direction = 0
                    else:
                        feedback = 'Fix Form'

            # draw bar
            if form == 1:
                cv2.rectangle(out, (580, 50), (600, 380), (0, 255, 0), 3)
                cv2.rectangle(out, (580, int(bar)), (600, 380), (0, 255, 0), cv2.FILLED)
                cv2.putText(out, f'{int(per)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 0), 2)
            
            # push up count
            cv2.rectangle(out, (0,380), (100, 480), (0, 255, 0), cv2.FILLED)
            cv2.putText(out, str(int(count)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

            # feedback print
            cv2.rectangle(out, (500,0), (640, 40), (255,255,255), cv2.FILLED)
            cv2.putText(out, feedback, (500, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    

        cv2.imshow('Pose estimator', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
 

def squat(net):
    count = 0
    direction = 0
    form = 0
    feedback = "Fix Form"
    
    print('-------------- original Form 상태 -----------------')
    print(form)
    print('-' * 30)

    # webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, oriImg = video_capture.read() 
        
        # 이미지 전처리
        preprocess_img = img_preprocess(oriImg)
        # pafs, heatmaps 구하기
        pafs, heatmaps = get_pafs_heatmaps(net, preprocess_img, oriImg)
        # pose test 해보기
        out, joint_lmList = pose_test(oriImg, pafs, heatmaps)
        
        # joint_lmList 전처리 --> 중복된 값 없게 설정
        unique_joint = []  # 중복되지 않는 값들을 다시 정의
        joint_lmList_to_list = [list(i) for i in joint_lmList]
        for joint in joint_lmList_to_list:
            if joint in unique_joint:
                continue
            else:
                unique_joint.append(joint)

        joint_dict = {} # 중복되지 않는 값들을 다시 정의 --> dict형태로 --> 관절 위치를 key값으로 설정
        for joint in unique_joint:
            if int(joint[-1]) in joint_dict:
                joint_dict[int(joint[-1])].append(joint[0:2])
            else:
                joint_dict[int(joint[-1])] = [joint[0:2]]
        '''
        (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
        5-'left_shoulder' 6-'left_elbow'	7-'left_wrist'  8-'right_hip'
        9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
        13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
        17-'left_ear' )
        '''
        # left leg --> left hip(11), left knee(12), left ankle(13)
        # right leg --> right hip(8), right knee(9), right angle(10)

        need_pose = [8,9,10,11,12,13] # squat를 detect하기 위해 필요한 pose 위치
        
        flag = 1
        for pose in need_pose:  # joint_dict.keys()에 필요한 pose 위치가 없으면 수행 x
            if pose not in joint_dict.keys():
                flag=0
                break
        
        if len(joint_dict) != 0 and flag == 1:
            
            left_knee_angle = findAngle(out,joint_dict, 11, 12, 13)
            right_knee_angle = findAngle(out,joint_dict, 8, 9, 10)
        

            # squat 성공 퍼센트 
            per_left = np.interp(left_knee_angle, (90, 160), (0, 100))
            per_right = np.interp(right_knee_angle, (90, 160), (0, 100))
            print('percnet left :',per_left)
            print('percent right :',per_right)

            # squat 진행정도를 보여주는 bar
            bar_left = np.interp(left_knee_angle, (0,160), (380,50))
            bar_right = np.interp(right_knee_angle, (0,160), (380,50))

            # squat 시작 자세 check
            if left_knee_angle > 160 and right_knee_angle > 160:
                form = 1

            print('---------- form 상태 ---------------')
            print(form)
            print('------------------------------------')
            
            # squat 전체 모션을 check
            if form == 1:
                if per_left == 0 and per_right ==0 :
                    print('왼쪽 무릎 각도 :', left_knee_angle)
                    print('오른쪽 무릎 각도 :', right_knee_angle)
                    if left_knee_angle < 90 and right_knee_angle < 90:
                        feadback = 'Up'
                        if direction == 0:
                            count += 0.5
                            direction = 1 # 방향 전환
                    else:
                        feedback = 'Fix Form'
                
                if per_left == 100 and per_right == 100:
                    if left_knee_angle > 160 and right_knee_angle > 160:
                        feedback = 'Down'
                        if direction == 1:
                            count += 0.5
                            direction = 0
                    else:
                        feedback = 'Fix Form'
            print('--------------------- count --------------------------' )
            print(count)
            print('-------------------------------------------------------')

            # draw bar
            if form == 1:
                # cv2.rectangle(out, (580, 50), (600, 380), (0, 255, 0), 3)
                # cv2.rectangle(out, (580, int(bar_left)), (600, 380), (0, 255, 0), cv2.FILLED)
                
                # cv2.rectangle(out, (30, 50), (50, 380), (0, 255, 0), 3)
                # cv2.rectangle(out, (30, int(bar_right)), (50, 380), (0, 255, 0), cv2.FILLED)
                
                # cv2.putText(out, f'{int(per_left)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 0), 2)
                # cv2.putText(out, f'{int(per_right)}%', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 0), 2)

                cv2.putText(out, f'{int(per_left)}%', (565, 240), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255), 2)
                cv2.putText(out, f'{int(per_right)}%', (10, 240), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255), 2)
            
            # squat count
            cv2.rectangle(out, (0,400), (100, 480), (0, 255, 0), cv2.FILLED)
            cv2.putText(out, str(int(count)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

            # feedback print
            cv2.rectangle(out, (500,0), (640, 40), (255,255,255), cv2.FILLED)
            cv2.putText(out, feedback, (500, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)


        cv2.imshow('Pose estimator', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights = '../pose_model_scratch.pth'

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

squat(net)
# push_up(net)