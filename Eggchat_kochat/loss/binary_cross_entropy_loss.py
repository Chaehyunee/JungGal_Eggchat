import torch
from torch.nn import functional as F
from torch import Tensor
from torch import nn
from kochat.loss.base_loss import BaseLoss
from foodfeeling.decorators import loss


@loss
class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, label_dict: dict):
        """
        binary cross entropy loss를 계산합니다.

        :param label_dict: 라벨 딕셔너리
        """

        super(BinaryCrossEntropyLoss, self).__init__()

        self.label_dict = label_dict   
        self.loss = nn.BCEWithLogitsLoss().to(self.device)


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss(input, target)

    def compute_loss(self, labels: Tensor, logits: Tensor, feats: Tensor = None, mask: nn.Module = None) -> Tensor:
        """
        학습을 위한 total loss를 계산합니다.

        :param label: label
        :param logits: logits
        :param feats: feature
        :param mask: mask vector
        :return: total loss
        """

        if mask is None:
            # 마스크 없는 경우 torch의 cross entropy 이용
            return self(logits, labels)

        else:
            # 마스크 있는 경우 마스크를 처리하고 cross entropy 계산
            logits = logits.to(self.device).clone()
            labels = labels.to(self.device).clone()

            labels = labels.float().to(self.device).clone()

            return self(input=logits, target=labels)







            # 아마 forward로 가서 알아서 작동하지 않을까 (self(~,~) 문법 공부해보기)
            # logits = logits.permute(1, 0) #[4, 512] -> [512, 4]
            # logits_flat = logits.view(-1, logits.size(-1)) # 텐서 reshape 해준 다음 (?xlogits.size(-1) 크기로 변경)
            # log_probs_flat = F.log_softmax(logits_flat, dim=1) # 모델에서 나온 logits을 flat화 시킨 것을 softmax에 넣어준다
            # target_flat = label.view(-1, 1) # label을 ?x1 형태로 flat화 시켜준다
            # losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat) # n차원 행렬에서 원하는 부분만 모아주는 것
            # # log_probs_flat : 소스 Tensor , dim = 인덱스 생성할 축, index=수집할 요소의 인덱스
            # losses = losses_flat.view(mask.size()) # 마스크 크기로 shape를 다시 바꾸고
            # losses = losses * mask.float() # loss에 마스크 정수 값을 곱한다 (0이면 사라지고, 나머지는 살아남음)
            # return losses.mean() # 평균을 구하면 이 과정이 cross entropy인듯 (마스크 제거 포함)
