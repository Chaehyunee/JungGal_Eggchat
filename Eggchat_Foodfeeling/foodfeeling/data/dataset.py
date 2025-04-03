import os
import random

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader


import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/torch/utils/data')
from dataset import TensorDataset
from typing import Tuple
from torch import default_generator



from foodfeeling.data.organizer import Organizer
from foodfeeling.data.preprocessor import Preprocessor
from foodfeeling.decorators import data
from foodfeeling.proc.base_processor import BaseProcessor


@data
class FoodDataset:

    def __init__(self, ood: bool, naver_fix: bool = False):
        """
        학습과 추론에 사용할 데이터셋을 생성하는 클래스입니다.
        ood는 Out of distribution 데이터셋 사용 여부입니다.
        ood 데이터를 쓰면 Threshold 설정 없이 Automatic Fallback Detection이 가능합니다.

        :param naver_fix: 네이버 맞춤법 검사기 사용 여부 (상업적 이용시 꺼주세요)
        :param ood: Out of distribution dataset 사용 여부입니다.
        """

        self.ood = ood
        self.org = Organizer()
        self.prep = Preprocessor(naver_fix=naver_fix)

        self.feeling_tag_dict = self.org.organize_feeling_tag()
        self.entity_dict = self.org.organize_entity()

    def load_tagging_embed(self) -> list:
        """
        임베딩 프로세서 학습용 데이터를 생성합니다.
        임베딩 데이터셋은 라벨(인텐트, 엔티티)가 필요하지 않기 때문에
        라벨을 자르고 문장 (데이터) 부분만 반환합니다.

        :return: 라벨이 없는 임베딩 학습용 데이터셋입니다.
        """

        embed_dataset = pd.read_csv(self.entity_data_dir)

        if self.ood:  # ood = True이면 ood도 로드해서 합침
            embed_dataset = pd.concat([embed_dataset, self.__load_ood()])

        embed_dataset = embed_dataset.values.tolist()
        embed_dataset = self.__tokenize_dataset(embed_dataset)
        question_index, label_index = 0, 1  # 매직넘버 사용하지 않기 !

        return np.array(embed_dataset)[:, question_index].tolist()
        # label말고 question 부분만 리턴하기 (gensim 임베딩은 라벨 필요 X)

# good, sad, tired, stress 의 multi-label classifier 제작 시 feeling_tag
    def load_feeling_tag(self, emb_processor: BaseProcessor) -> tuple:
        """
        feeling_tag 프로세서 학습용 데이터를 생성합니다.

        :param emb_processor: 임베딩 과정이 들어가므로 임베딩 프로세서를 입력해야합니다.
        :return: 인텐트 프로세서 학습용 데이터셋입니다.
        """

        feeling_tag_dataset = pd.read_csv(self.feeling_tag_data_dir)

        feeling_tag_train, feeling_tag_test = self.__make_feeling_tag(feeling_tag_dataset, emb_processor)

        feeling_tag_train, feeling_tag_test = self.__mini_batch(feeling_tag_train), self.__mini_batch(feeling_tag_test)

        return feeling_tag_train, feeling_tag_test

    def load_tagging_entity(self, emb_processor: BaseProcessor) -> tuple:
        """
        엔티티 프로세서 학습용 데이터를 생성합니다.

        :param emb_processor: 임베딩 과정이 들어가므로 임베딩 프로세서를 입력해야합니다.
        :return: 엔티티 프로세서 학습용 데이터셋입니다.
        """

        entity_dataset = pd.read_csv(self.entity_data_dir)
        entity_train, entity_test = self.__make_entity(entity_dataset, emb_processor)
        return self.__mini_batch(entity_train), self.__mini_batch(entity_test)

    def load_predict(self, text: str, emb_processor: BaseProcessor) -> Tensor:
        """
        실제 애플리케이션 등에서 유저 입력에 대한 인퍼런스를 수행할 때
        사용자가 입력한 Raw 텍스트(str)를 텐서로 변환합니다.

        :param text: 사용자의 텍스트 입력입니다.
        :param emb_processor: 임베딩 과정이 들어가므로 임베딩 프로세서를 입력해야합니다.
        :return: 유저 입력 추론용 텐서를 리턴합니다.
        """

        text = self.prep.tokenize(text, train=False)  # 토크나이징 (띄어쓰기 기준 split)

        if len(text) == 0:
            raise Exception("문장 길이가 0입니다.")

        text = emb_processor.predict(text)  # 임베딩
        text, _ = self.prep.pad_sequencing(text)  # 패드 시퀀싱
        return text.unsqueeze(0).to(self.device)  # 차원 증가 (batch_size = 1)

    def __load_ood(self) -> DataFrame:
        """
        메모리에서 OOD 데이터를 로드합니다.
        OOD 데이터는 폴백 디텍션 모델의 Threshold를 자동으로 설정하고,
        인텐트 검색기와 폴백 디텍션 모델의 성능을 검증하기 위해 사용됩니다.

        :return: 여러개의 OOD 데이터를 한 파일로 모아서 반환합니다.
        """

        ood_dataset = []

        for ood in os.listdir(self.ood_data_dir):
            if ood != '__init__.py':  # __init__ 파일 제외
                ood = pd.read_csv(self.ood_data_dir + ood)
                ood_dataset.append(ood)

        return pd.concat(ood_dataset)

    def __make_feeling_tag(self, feeling_tag_dataset: DataFrame, emb_processor: BaseProcessor) -> tuple:# TODO intent를 multi-label classification 으로 변경
        """
        food-feeling tag 데이터셋을 만드는 세부 과정입니다.

        - 라벨을 숫자로 맵핑합니다.
        - 데이터를 토큰화 합니다 (네이버 맞춤법 검사기 + Konlpy 사용)
        - 데이터를 학습 / 검증용으로 나눕니다.
        - 데이터의 길이를 맞추기 위해 임베딩합니다.
        - 리스트로 출력된 데이터들을 concatenation하여 텐서로 변환합니다.

        :param feeling_tag_dataset: 저장공간에서 로드한 인텐트 데이터 파일입니다.
        :param emb_processor: 임베딩을 위한 임베딩 프로세서를 입력해야합니다.
        :return: 텐서로 변환된 인텐트 데이터입니다.
        """

        feeling_tag_dataset = self.__map_label(feeling_tag_dataset, 'feeling_tag') # 라벨을 숫자로 맵핑
        feeling_tag_dataset = self.__tokenize_dataset(feeling_tag_dataset) # 데이터를 토큰화
        train, test = self.__split_data(feeling_tag_dataset)

        train_question, \
        train_label_g, train_label_s, train_label_t, train_label_r, \
        train_length = self.__embedding(train, emb_processor)

        test_question, \
        test_label_g, test_label_s, test_label_t, test_label_r, \
        test_length = self.__embedding(test, emb_processor)

        train_tensors = self.__list2tensor(train_question, train_label_g, train_label_s, train_label_t, train_label_r, train_length)
        test_tensors = self.__list2tensor(test_question, test_label_g, test_label_s, test_label_t, test_label_r, test_length)
        return train_tensors, test_tensors

    def __make_entity(self, entity_dataset: DataFrame, emb_processor: BaseProcessor) -> tuple:
        """
        엔티티 데이터셋을 만드는 세부 과정입니다.

        - 라벨을 숫자로 맵핑합니다.
        - 데이터를 토큰화 합니다 (네이버 맞춤법 검사기 + Konlpy 사용)
        - 데이터를 학습 / 검증용으로 나눕니다.
        - 데이터의 길이를 맞추기 위해 패드시퀀싱 후 임베딩합니다.
        - 엔티티 데이터는 라벨도 각각 길이가 달라서 패드시퀀싱 해야합니다.
        - 리스트로 출력된 데이터들을 concatenation하여 텐서로 변환합니다.

        :param entity_dataset: 저장공간에서 로드한 엔티티 데이터 파일입니다.
        :param emb_processor: 임베딩을 위한 임베딩 프로세서를 입력해야합니다.
        :return: 텐서로 변환된 엔티티 데이터입니다.
        """

        entity_dataset = self.__map_label(entity_dataset, 'entity')
        entity_dataset = self.__tokenize_dataset(entity_dataset)
        train, test = self.__split_data(entity_dataset)

        train_question, train_label, train_length = self.__embedding(train, emb_processor)
        test_question, test_label, test_length = self.__embedding(train, emb_processor)

        train_label = [self.prep.label_sequencing(label, self.entity_dict) for label in train_label]
        test_label = [self.prep.label_sequencing(label, self.entity_dict) for label in test_label]
        # 1차원 라벨 리스트들을 하나하나 꺼내서 패드 시퀀싱한 뒤 다시 리스트에 넣어 2차원으로 만듬

        train_tensors = self.__list2tensor(train_question, train_label, train_length)
        test_tensors = self.__list2tensor(test_question, test_label, test_length)
        return train_tensors, test_tensors

    def __map_label(self, dataset: DataFrame, kinds: str) -> list:
        """
        라벨을 맵핑합니다.
        데이터를 불러오고 나서 라벨을 컴퓨터가 이해 가능한 숫자의 형태로 맵핑합니다.

        :param dataset: 메모리로 부터 불러온 데이터셋입니다.
        :param kinds: 어떤 종류의 데이터인지(feeling_tag or entity) 나타냅니다.
        :return: 맵핑이 완료된 리스트 데이터 셋
        """

        questions, labels = dataset['FOODNAME'], None 

        if kinds == 'feeling_tag':
            labels = []
            labels.append(dataset['good'].map(self.feeling_tag_dict))
            labels.append(dataset['sad'].map(self.feeling_tag_dict))
            labels.append(dataset['tired'].map(self.feeling_tag_dict))
            labels.append(dataset['stress'].map(self.feeling_tag_dict))
            labels[0].fillna(-1, inplace=True)
            labels[1].fillna(-1, inplace=True)
            labels[2].fillna(-1, inplace=True)
            labels[3].fillna(-1, inplace=True)
            # ood는 feeling_tag dict에 라벨이 없기 때문에 nan가 되는데 -1로 대체함.
            labels[0] = labels[0].astype(int).tolist()
            labels[1] = labels[1].astype(int).tolist()
            labels[2] = labels[2].astype(int).tolist()
            labels[3] = labels[3].astype(int).tolist()
            # fillna하면 float이 되기 때문에 이를 int로 바꿔줌

            return list(zip(questions, labels[0], labels[1], labels[2], labels[3]))

        elif kinds == 'entity':
            # 라벨 태그(COMMENT, O 등을 하나씩 꺼내와서 1, 2와 같은 숫자로 맵핑)

            labels = [[self.entity_dict[t] for t in lable_tag.split()]
                      for lable_tag in dataset['label']]

        return list(zip(questions, labels))

    def __tokenize_dataset(self, dataset: list) -> list:
        """
        데이터셋(2차원)을 토큰화 합니다.

        :param dataset: 라벨 맵핑이 완료된 리스트 데이터셋입니다.
        :return: 토큰화가 완료된 리스트 데이터셋입니다.
        """

        # 엔티티 라벨과 문장 데이터는 1차원(벡터)가 여러개 있으므로 2차원인데
        # 인텐트 라벨의 경우, 0차원(스칼라)이 여러개 있으므로 1차원입니다.
        # 때문에 만약 타입이 list가 아닌 스칼라는 (즉, 인텐트 라벨의 경우는)
        # 강제로 길이가 1인 리스트로 만들어서 (unsqueeze) 차원을 맞춥니다.
        if len(dataset) > 2:
            return [[self.prep.tokenize(question, train=True),  # question부분은 토크나이징
                    [label_g] if not isinstance(label_g, list) else label_g,
                    [label_s] if not isinstance(label_s, list) else label_s,
                    [label_t] if not isinstance(label_t, list) else label_t,
                    [label_r] if not isinstance(label_r, list) else label_r]  # feeling_tag 라벨 unsqueeze해주기
                    for (question, label_g, label_s, label_t, label_r) in dataset]  # 데이터 셋에서 하나씩 꺼내와서

        else:
            return [[self.prep.tokenize(question, train=True),  # question부분은 토크나이징
                    [label] if not isinstance(label, list) else label]  # intent 라벨 unsqueeze해주기
                    for (question, label) in dataset]  # 데이터 셋에서 하나씩 꺼내와서

    def __split_data(self, dataset: list) -> tuple:
        """
        데이터셋을 학습용 / 검증용으로 나눕니다.
        Configuration에 적힌 split ratio를 기준으로 데이터를 쪼갭니다.

        :param dataset: 토큰화가 완료된 리스트 데이터셋
        :return: 분리가 완료된 (학습용 데이터, 검증용 데이터)
        """
        random.shuffle(dataset)  # 데이터 섞어주기
        split_point = int(len(dataset) * self.data_ratio)

        train_dataset = dataset[:split_point]
        test_dataset = dataset[split_point:]
        return train_dataset, test_dataset

    def __embedding(self, dataset: list, emb_processor: BaseProcessor) -> tuple:
        """
        자연어 데이터를 연산하기 위해 임베딩합니다.
        임베딩을 위해서 각각 다른 데이터들의 길이를 일정하게 맞추고,
        (Pad Sequencing 진행) 임베딩 프로세서로 임베딩합니다.

        :param dataset: 이전단계에서 학습/검증용으로 나뉜 데이터 중 하나
        :param emb_processor: 임베딩을 위한 임베딩 프로세서
        :return: 임베딩된 자연어 데이터, 라벨 데이터, 길이 데이터
        """

        if len(dataset) > 2:
            question_list, label_list_g, label_list_s, label_list_t, label_list_r, length_list = [], [], [], [], [], []
            for i, (question, label_g, label_s, label_t, label_r) in enumerate(dataset):
                question = emb_processor.predict(question)
                question, length = self.prep.pad_sequencing(question)

                question_list.append(question.unsqueeze(0))
                label_list_g.append(torch.tensor(label_g).unsqueeze(0))
                label_list_s.append(torch.tensor(label_s).unsqueeze(0))
                label_list_t.append(torch.tensor(label_t).unsqueeze(0))
                label_list_r.append(torch.tensor(label_r).unsqueeze(0))
                length_list.append(torch.tensor(length).unsqueeze(0))
       
            return question_list, label_list_g, label_list_s, label_list_t, label_list_r, length_list


        else:
            question_list, label_list, length_list = [], [], []
            for i, (question, label) in enumerate(dataset):
                question = emb_processor.predict(question)
                question, length = self.prep.pad_sequencing(question)

                question_list.append(question.unsqueeze(0))
                label_list.append(torch.tensor(label))
                length_list.append(torch.tensor(length).unsqueeze(0))

            return question_list, label_list, length_list

    def __list2tensor(self, *lists: list) -> list:
        """
        리스트 데이터들을 Pytorch의 Tensor 형태로 변환하여 병합(concatenation)합니다.
        각 리스트는 0번 Axis로 모두 Unsqueeze 되어 있어야 합니다.
        lists 길이 : 6

        :param lists: 텐서로 만들 리스트 데이터셋들 (shape = [1, XXX, XXX])
        :return: 텐서 데이터들이 담긴 리스트 (타입은 list지만 텐서로 unpacking 됩니다)
        """

        return [torch.cat(a_list, dim=0) for a_list in lists]

    def __mini_batch(self, tensors: tuple) -> DataLoader:
        """
        데이터를 미니배치 형태로 쪼개서 로딩할 수 있게 하는
        Pytorch DataLoader로 만듭니다.

        :param tensors: 텐서로 병합한 데이터셋들
        :return: 미니배치 트레이닝용 데이터로더 객체
        """

        return DataLoader(
            dataset=MyTensorDataset(*tensors),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )



class MyTensorDataset(TensorDataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        
        # tensors[0].size(0), tensors[0][0].size(0), tensors[0].size()
        # 1945                  8                   torch.Size([1945, 8, 128])

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)