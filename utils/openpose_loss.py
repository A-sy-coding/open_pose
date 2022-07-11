import torch.nn as nn
import torch.nn.functional as F

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
