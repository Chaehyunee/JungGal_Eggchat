from abc import ABCMeta, abstractmethod
from time import time
from typing import List

import torch
from torch import nn
from torch.nn import Parameter
from torch import Tensor


from foodfeeling.decorators import food_feeling
from kochat.loss.base_loss import BaseLoss
from foodfeeling.loss.masking import Masking
from foodfeeling.proc.torch_processor import TorchProcessor

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import kobert

@food_feeling
class SadClassifier(TorchProcessor):

    def __init__(self, model: nn.Module, loss: BaseLoss):
        """
        Food 에 따른 Feeling 별 추천 여부 분류 모델을 학습시키고 테스트 및 추론합니다.

        :param model: FoodFeelingClassification 모델
        :param loss: Loss 함수 종류
        """
        self.label_dict = model.label_dict
        self.loss = loss.to(self.device)
        self.mask = Masking() if self.masking else None
        self.parameters = list(model.parameters())

        if len(list(loss.parameters())) != 0:
            self.parameters += list(loss.parameters())

        self.device = torch.device("cuda:0")
        self.model = self.__add_classifier(model).to(self.device)

        super().__init__(self.model, self.parameters)


    def predict(self, sequence: Tensor) -> list:
        """
        사용자의 입력에 inference합니다.
        
        :param sequence: 입력 시퀀스
        :return: 분류 결과 (엔티티 시퀀스) 리턴
        """

        self._load_model(self.model, feeling='sad')
        self.model.eval()

        # 만약 i가 pad라면 i값에서 PAD를 빼면 전부 0이 되고
        # 그 상태에서 입력 전체에 1을 더하면, pad토큰은 [1, 1, 1, ...]이 됩
        # all()로 체크하면 pad만 True가 나옴. (모두 1이여야 True)
        # 이 때 False 갯수를 세면 pad가 아닌 부분의 길이가 됨.
        length = [all(map(int, (i - self.PAD + 1).tolist()))
                  for i in sequence.squeeze()].count(False)

        predicts = self._forward(self.model, self.loss, sequence).squeeze().t()

        predicts = [list(self.label_dict.keys())[i.item()]  # 라벨 딕셔너리에서 i번째 원소를 담음
                    for i in predicts]  # 분류 예측결과에서 i를 하나씩 꺼내서

        return predicts[:length]


    def _train_epoch(self, epoch: int) -> tuple:
        """
        학습시 1회 에폭에 대한 행동을 정의합니다.

        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """
        loss_list, predict_list, feats_list, label_list = [], [], [], []

        self.model.train()

        for feats, labels_g, labels_s, labels_t, labels_r, lengths in self.train_data:
            feats, labels = feats.to(self.device), labels_s.to(self.device)

            # forward
            predicts, feats, losses = self._forward(self.model, self.loss, feats, labels, lengths)

            # backward
            losses = self._backward(losses)

            # feats list에 append
            feats_list.append(feats)

            # loss list에 append
            loss_list.append(losses)

            # predict list에 append
            predict_list.append(predicts)

            # label list에 append
            label_list.append(labels) 

        # loss 계산
        losses = sum(loss_list) / len(loss_list)
        feats = torch.cat(feats_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        return losses, labels, predicts

    def _test_epoch(self, epoch: int) -> tuple:
        """
        테스트시 1회 에폭에 대한 행동을 정의합니다.
        
        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """
        loss_list, predict_list, feats_list, label_list = [], [], [], []

        self.model.eval()

        for feats, labels_g, labels_s, labels_t, labels_r, lengths in self.train_data:
            feats, labels = feats.to(self.device), labels_s.to(self.device)

            # forward
            predicts, feats, losses = self._forward(self.model, self.loss, feats, labels, lengths)

            # feats list에 append
            feats_list.append(feats)

            # loss list에 append
            loss_list.append(losses)

            # predict list에 append
            predict_list.append(predicts)

            # label list에 append
            label_list.append(labels) 

        # loss 계산
        losses = sum(loss_list) / len(loss_list)
        feats = torch.cat(feats_list, dim=0)
        predicts = torch.cat(predict_list, dim=0)
        labels = torch.cat(label_list, dim=0)

        return losses, labels, predicts

    def _forward(self, model, loss, feats: Tensor, labels: Tensor = None, length: Tensor = None) -> tuple:
        """
        모델의 feed forward에 대한 행동을 정의합니다.

        :param feats: 입력 feature
        :param labels: label 리스트
        :param length: 패딩을 제외한 입력의 길이 리스트
        :return: 모델의 예측, loss
        """

        feats = model(feats)
        logits = model.classifier(feats)

        predicts = torch.max(logits, dim=1)[1]

        if labels is None:
            return predicts
        else:
            mask = self.mask(length) if self.mask else None
            loss = loss.compute_loss(labels, logits, feats, mask)
            return predicts, feats, loss

    def __add_classifier(self, model):
        sample = torch.randn(1, self.max_len, self.vector_size)
        sample = sample.to(self.device)
        output_size = model.to(self.device)(sample)
        # output_size를 정해줌
        
        classifier = nn.Linear(output_size.shape[1], self.classes) 
        # nn.Linear : input features size랑 output feature size 를 넣어준다
        setattr(model, 'classifier', classifier.to(self.device))
        return model
