import math
import matplotlib.pyplot as plt
import numpy as np
#Get the landmarks
# x1,y1 = 281, 163
# x2,y2 = 229, 248
# x3,y3 = 234, 378

# x1,y1 = 206, 259
# x2,y2 = 277, 269
# x3,y3 = 233, 376

# x1,y1 = 421, 240
# x2,y2 = 347, 267
# x3,y3 = 644 , 371

# # x2,y2가 기준 --> 해당 좌표를 (0,0)으로 이동
# new_x1, new_y1 = (x1 - x2), (y1 - y2) 
# new_x3, new_y3 = (x3 - x2), (y3 - y2)
# print('원점 이동 후 x1, y1, 좌표 : ', new_x1, new_y1)
# print('원점 이동 후 x3, y3, 좌표 : ', new_x3, new_y3)

# if angle < 0 :
#     angle +=  360  # 값이 360도라는 범위 안에 들게 한다.
#     if angle > 180:
#         angle = 360 - 180 # 값의 범위를 180도 안으로 설정하도록 한다.
#     elif angle > 180:
#         angle = 360 - angle
# # plt.plot(new_x1, new_y1, 'o')
# # plt.plot(new_x3, new_y3, 'o')
# # plt.show()

# print('원점과 (x1,y1)의 기울기 : ', new_y1 / new_x1)
# print('원점과 (x3,y3)의 기울기 : ', new_y3 / new_x3)

# # 원점 이동 후 두 기울기 사이의 각도 구하기
# temp1 = math.degrees(math.atan2(new_y1 , new_x1))
# temp2 = math.degrees(math.atan2(new_y3 , new_x3))
# print('(x1,y1)과 x축 사이의 각도 :', temp1)
# print('(x3,y3)과 x축 사이의 각도 :', temp2)

# angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
# print('원래 angle : ', angle)
# print('변환 angle : ', 360 - angle)

elbow = 50
bar = np.interp(elbow, (0,160), (380,50))
print(bar)
    # per_right = np.interp(right_knee_angle, (90, 160), (0, 100))