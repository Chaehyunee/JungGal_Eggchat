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

from kochat.proc.feeling_classifier.good_classifier import GoodClassifier
from kochat.proc.feeling_classifier.sad_classifier import SadClassifier
from kochat.proc.feeling_classifier.tired_classifier import TiredClassifier
from kochat.proc.feeling_classifier.stress_classifier import StressClassifier

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import kobert

@food_feeling
class FoodFeelingClassifier(TorchProcessor): # TODO 상속 없애기 고민

    def __init__(self, 
                    model_g: nn.Module, 
                    model_s: nn.Module, 
                    model_t: nn.Module, 
                    model_r: nn.Module,
                    loss_g: BaseLoss,
                    loss_s: BaseLoss,
                    loss_t: BaseLoss,
                    loss_r: BaseLoss
                    ):
        """
        Food 에 따른 Feeling 별 추천 여부 분류 모델을 학습시키고 테스트 및 추론합니다.

        :param model_g: FoodFeelingClassification 모델, Feeling : good
        :param loss_g: Loss 함수 종류, Feeling : good  
        """
        self.good_classifier = GoodClassifier(model_g, loss_g)
        self.sad_classifier = SadClassifier(model_s, loss_s)
        self.tired_classifier = TiredClassifier(model_t, loss_t)
        self.stress_classifier = StressClassifier(model_r, loss_r)

        self.device = torch.device("cuda:0")

    def predict(self, sequence: Tensor) -> list:
        """
        사용자의 입력에 inference합니다.
        
        :param sequence: 입력 시퀀스
        :return: 분류 결과 (엔티티 시퀀스) 리턴
        """

        predict_g = self.good_classifier.predict(sequence)
        predict_s = self.sad_classifier.predict(sequence)
        predict_t = self.tired_classifier.predict(sequence)
        predict_r = self.stress_classifier.predict(sequence)

        return [predicts_g, predicts_s, predicts_t, predicts_r] #list로 return 할지, dictionary로 return할지 추후 결정

    def _train_epoch(self, epoch: int) -> tuple:
        """
        학습시 1회 에폭에 대한 행동을 정의합니다.

        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """

        losses_g, labels_g, predicts_g = self.good_classifier.train_epoch(epoch)
        losses_s, labels_s, predicts_s = self.sad_classifier.train_epoch(epoch)
        losses_t, labels_t, predicts_t = self.tired_classifier.train_epoch(epoch)
        losses_r, labels_r, predicts_r = self.stress_classifier.train_epoch(epoch)

        losses = [losses_g, losses_s, losses_t, losses_r]
        labels = [labels_g, labels_s, labels_t, labels_r]
        predicts = [predicts_g, predicts_s, predicts_t, predicts_r]

        return losses, labels, predicts

    def _test_epoch(self, epoch: int) -> tuple:
        """
        테스트시 1회 에폭에 대한 행동을 정의합니다.
        
        :param epoch: 현재 에폭
        :return: 평균 loss, 예측 리스트, 라벨 리스트
        """

        losses_g, labels_g, predicts_g = self.good_classifier.test_epoch(epoch)
        losses_s, labels_s, predicts_s = self.sad_classifier.test_epoch(epoch)
        losses_t, labels_t, predicts_t = self.tired_classifier.test_epoch(epoch)
        losses_r, labels_r, predicts_r = self.stress_classifier.test_epoch(epoch)

        losses = [losses_g, losses_s, losses_t, losses_r]
        labels = [labels_g, labels_s, labels_t, labels_r]
        predicts = [predicts_g, predicts_s, predicts_t, predicts_r]

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
