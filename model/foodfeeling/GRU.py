from torch import Tensor
from torch import nn
import torch

from foodfeeling.decorators import food_feeling
from kochat.model.layers.convolution import Convolution

from torch.autograd import Variable


@food_feeling
class BasicGRU(nn.Module):
    def __init__(self, label_dict: dict, residual: bool = True):
        """
        food-feeling Classification을 위한 GRU 클래스입니다.

        :param label_dict: 라벨 딕셔너리
        :param residual: skip connection 여부
        """
        super(BasicGRU, self).__init__()
        self.label_dict = label_dict
        print('GRU init label_dict len : ', len(label_dict))

       #앞에서 정의한 하이퍼 파라미터를 넣어 GRU 정의
        self.gru = nn.GRU(input_size=self.embed_dim, 
                          hidden_size=self.hidden_dim,
                          num_layers=self.n_layers,
                          batch_first=True,
                          dropout=self.dropout_p) 

        # self.fc_1 = nn.Linear(self.hidden_dim, 128)
        # self.fc = nn.Linear(128, self.n_classes)
        # self.relu = nn.ReLU(inplace=False)

        #Input: GRU의 hidden vector(context), Output : Class probability vector
        self.out = nn.Linear(self.hidden_dim, self.n_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        # Input data : 한 batch 내 모든 음식명-감정 데이터

        # 초기 hidden state vector를 zero vector로 생성
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim)).to(self.device)
        output, (h_n) = self.gru(x, h_0)

        # h_t : Batch 내 모든 sequential hidden state vector의 제일 마지막 토큰을 내포한 (batch_size, 1, hidden_dim)형태의 텐서 추출
        # 다른 의미로 음식명-감정 배열들을 압축한 hidden state vector
        h_t = h_n.view(-1, self.hidden_dim) # 이거 하니까 512로 바뀜 self.hidden_dim이 512여서 그런가봄

        # linear layer의 입력으로 주고, 각 클래스 별 결과 logit을 생성.
        logit = self.out(h_t)  # [b, h] -> [b, o]
        return logit