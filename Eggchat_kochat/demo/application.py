"""
@auther Chaehyunee
@since 25/9/2022
"""
import sys
sys.path.append('/data/EggChat/kochat')
sys.path.append('/data/EggChat')
sys.path.append('/data/kocrawl')
sys.path.append('/data/EggChat/foodfeeling/')

from app.kochat_api import EggChatApi
import ssl
from kochat.data import Dataset
from loss import CRFLoss, CosFace, CenterLoss, COCOLoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from model import intent, embed, entity, foodfeeling
from proc import DistanceClassifier, GensimEmbedder, EntityRecognizer, FoodFeelingClassifier, FoodFeelingEntityRecognizer

from foodfeeling.data import FoodDataset
from foodfeeling.proc import FoodFeelingGensimEmbedder

from scenario import dust, weather, travel, restaurant, feeling, foodname, food_feeling
# 에러 나면 이걸로 실행해보세요!


dataset = Dataset(ood=True) #Eggchat dataset
emb = GensimEmbedder(model=embed.FastText()) #Eggchat Embedder

clf = DistanceClassifier( #Eggchat Intent 분류를 위한 Distance Classifier
    model=intent.CNN(dataset.intent_dict),
    loss=CenterLoss(dataset.intent_dict),
)

rcn = EntityRecognizer( #Eggchat Intent에 따른 Entity 인식을 위한 Recognizer
    model=entity.LSTM(dataset.entity_dict),
    loss=CRFLoss(dataset.entity_dict)
)

foodfeeling_dataset = FoodDataset(ood=False) #게시글 제목에서 음식 명 Entity 를 찾기 위한 Dataset 클래스
foodfeeling_emb = FoodFeelingGensimEmbedder(model=embed.FastText()) #찾은 음식명 Entity에 기분별 True, False 태그를 위해 임베딩

foodfeeling_rcn = FoodFeelingEntityRecognizer(
    model=entity.LSTM(foodfeeling_dataset.entity_dict),
    loss=CRFLoss(foodfeeling_dataset.entity_dict)
)

foodfeeling_clf = FoodFeelingClassifier( # TODO 내부 모듈 개발해야함 (Classifier, GRU, BCELoss)
    model_g=foodfeeling.BasicGRU(foodfeeling_dataset.feeling_tag_dict), # BasicGRU 파라미터 label_dict 뭐 넣어줄지 고민 후 수정
    model_s=foodfeeling.BasicGRU(foodfeeling_dataset.feeling_tag_dict), # BasicGRU 파라미터 label_dict 뭐 넣어줄지 고민 후 수정
    model_t=foodfeeling.BasicGRU(foodfeeling_dataset.feeling_tag_dict), # BasicGRU 파라미터 label_dict 뭐 넣어줄지 고민 후 수정
    model_r=foodfeeling.BasicGRU(foodfeeling_dataset.feeling_tag_dict), # BasicGRU 파라미터 label_dict 뭐 넣어줄지 고민 후 수정
    loss_g=BinaryCrossEntropyLoss(foodfeeling_dataset.feeling_tag_dict),
    loss_s=BinaryCrossEntropyLoss(foodfeeling_dataset.feeling_tag_dict),
    loss_t=BinaryCrossEntropyLoss(foodfeeling_dataset.feeling_tag_dict),
    loss_r=BinaryCrossEntropyLoss(foodfeeling_dataset.feeling_tag_dict)
)

eggchat = EggChatApi(
    dataset=dataset,
    embed_processor=(emb, False),
    intent_classifier=(clf, False),
    entity_recognizer=(rcn, False),
    foodfeeling_dataset=foodfeeling_dataset,
    foodfeeling_embed_processor=(foodfeeling_emb, False),
    foodfeeling_entity_recognizer=(foodfeeling_rcn, False),
    foodfeeling_classifier=(foodfeeling_clf, False),
    scenarios=[
        weather, dust, travel, restaurant, feeling, foodname, food_feeling 
    ]
)
# feeling: 기분에 따른 음식 추천, foodname: 음식명으로 게시물 검색, food_feeling: 게시물명에 따른 feeling 태그

eggchat.app.run(host='0.0.0.0', port= 8891)
# kochat.app.run(host='0.0.0.0', port= 8891, ssl_context='adhoc')