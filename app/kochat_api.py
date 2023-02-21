# Copyright 2020 EggChat. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List

#flask app 통신
from flask import Flask, render_template, request, jsonify

import sys
sys.path.append('/data/EggChat/foodfeeling')
sys.path.append('/data/EggChat/kochat')
sys.path.append('/data/EggChat/kochat/data')

from app.scenario import Scenario
from app.scenario_manager import ScenarioManager
from dataset import Dataset
from decorators import api
from foodfeeling.data import FoodDataset


@api
class EggChatApi:

    def __init__(self,
                 dataset: Dataset,
                 embed_processor,
                 intent_classifier,
                 entity_recognizer,
                 foodfeeling_dataset: FoodDataset,
                 foodfeeling_embed_processor,
                 foodfeeling_entity_recognizer,
                 foodfeeling_classifier,
                 scenarios: List[Scenario] = None):

        """
        Flask를 이용해 구현한 RESTFul API 클래스입니다.

        :param dataset: 데이터셋 객체
        :param embed_processor: 임베딩 프로세서 객체 or (, 학습여부)
        :param intent_classifier: 인텐트 분류기 객체 or (, 학습여부)
        :param entity_recognizer: 개체명 인식기 객체 or (, 학습여부)
        :param scenarios : 시나리오 리스트 (list 타입)
        """

        self.app = Flask(__name__)
        self.app.config['JSON_AS_ASCII'] = False

        self.dataset = dataset
        self.scenario_manager = ScenarioManager()
        self.dialogue_cache = {}

        self.foodfeeling_dataset = foodfeeling_dataset

        self.embed_processor = embed_processor[0] \
            if isinstance(embed_processor, tuple) \
            else embed_processor

        self.intent_classifier = intent_classifier[0] \
            if isinstance(intent_classifier, tuple) \
            else intent_classifier

        self.entity_recognizer = entity_recognizer[0] \
            if isinstance(entity_recognizer, tuple) \
            else entity_recognizer

        # food feeling
        self.foodfeeling_embed_processor = foodfeeling_embed_processor[0] \
            if isinstance(foodfeeling_embed_processor, tuple) \
            else foodfeeling_embed_processor

        self.foodfeeling_entity_recognizer = foodfeeling_entity_recognizer[0] \
            if isinstance(foodfeeling_entity_recognizer, tuple) \
            else foodfeeling_entity_recognizer

        self.foodfeeling_classifier = foodfeeling_classifier[0] \
            if isinstance(foodfeeling_classifier, tuple) \
            else foodfeeling_classifier

        # chatbot
        if isinstance(embed_processor, tuple) \
                and len(embed_processor) == 2 and embed_processor[1] is True:
            self.__fit_embed()

        if isinstance(intent_classifier, tuple) \
                and len(intent_classifier) == 2 and intent_classifier[1] is True:
            self.__fit_intent()

        if isinstance(entity_recognizer, tuple) \
                and len(entity_recognizer) == 2 and entity_recognizer[1] is True:
            self.__fit_entity()

        # food feeling
        if isinstance(foodfeeling_embed_processor, tuple) \
                and len(foodfeeling_embed_processor) == 2 and foodfeeling_embed_processor[1] is True:
            self.__fit_foodfeeling_embed()

        if isinstance(foodfeeling_entity_recognizer, tuple) \
                and len(foodfeeling_entity_recognizer) == 2 and foodfeeling_entity_recognizer[1] is True:
            self.__fit_foodfeeling_entity()

        if isinstance(foodfeeling_classifier, tuple) \
                and len(foodfeeling_classifier) == 2 and foodfeeling_classifier[1] is True:
            self.__fit_foodfeeling_classifier()


        for scenario in scenarios:
            self.scenario_manager.add_scenario(scenario)

        self.__build()

    def __build(self):
        """
        flask 함수들을 build합니다.
        """

        @self.app.route('/egg/request_chat', methods=['POST'])
        def request_chat() -> dict:
            """
            문자열을 입력하면 intent, entity, state, answer 등을 포함한
            딕셔너리를 json 형태로 반환합니다.

            :param uid: 유저 아이디 (고유값)
            :param input_text: 유저 입력 문자열
            :param x: 사용자 현재 위치 longitude, 경도
            :param y: 사용자 현재 위치 latitude, 위도
            :return: json 딕셔너리
            """
            params = request.get_json()
            text = params['input_text']
            uid = params['uid']
            x = params['x_coordinate']
            y = params['y_coordinate']

            prep = self.dataset.load_predict(text, self.embed_processor)
            intent = self.intent_classifier.predict(prep, calibrate=False) ##사용자에게 받은 문장-> intent 예측
            entity = self.entity_recognizer.predict(prep)
            text = self.dataset.prep.tokenize(text, train=False)
            self.dialogue_cache[uid] = self.scenario_manager.apply_scenario(intent, entity, text, x, y)


            """
            dialogue_cache[uid] ={
                'answer': self.api(*result.values())
                'entity': entity,
                'intent': self.intent,
                'state': 'SUCCESS',
            }
            """
            return jsonify(self.dialogue_cache[uid])

        @self.app.route('/egg/food_feeling', methods=['POST']) # 미완성 
        def request_foodfeeling():
            """
            문자열을 입력하면 good, sad, tired, stress 에 대한 tag 값을 포함한
            딕셔너리를 json 형태로 반환합니다.

            :param uid: 유저 아이디 (고유값)
            :param share_post_name: 게시물 제목 유저 입력 문자열
            :return: json 딕셔너리
            """
            params = request.get_json()
            print('android to kochat > '+ params['uid'], params['share_post_name'])
            share_post_name = params['share_post_name']
            uid = params['uid']
            intent = 'food_feeling'

            prep = self.foodfeeling_dataset.load_predict(share_post_name, self.foodfeeling_embed_processor)
            entity = self.foodfeeling_entity_recognizer.predict(prep)
            text = self.foodfeeling_dataset.prep.tokenize(share_post_name, train=False)
            self.dialogue_cache[uid] = self.scenario_manager.apply_scenario(intent, entity, share_post_name, user_id=uid) #x, y = 0 

            return jsonify(self.dialogue_cache[uid])


    def __fit_intent(self):
        """
        Intent Classifier를 학습합니다.
        """

        self.intent_classifier.fit(self.dataset.load_intent(self.embed_processor))

    def __fit_entity(self):
        """
        Entity Recognizr를 학습합니다.
        """

        self.entity_recognizer.fit(self.dataset.load_entity(self.embed_processor))

    def __fit_embed(self):
        """
        Embedding을 학습합니다.
        """

        self.embed_processor.fit(self.dataset.load_embed())


    # food feeling tagging
    def __fit_foodfeeling_classifier(self):
        """
        Food-Feeling multi-label classifier를 학습합니다.
        """

        self.foodfeeling_classifier.fit(self.foodfeeling_dataset.load_feeling_tag(self.foodfeeling_embed_processor))

    def __fit_foodfeeling_entity(self):
        """
        Entity Recognizr를 학습합니다.
        """

        self.foodfeeling_entity_recognizer.fit(self.foodfeeling_dataset.load_foodfeeling_entity(self.foodfeeling_embed_processor))

    def __fit_foodfeeling_embed(self):
        """
        Embedding을 학습합니다.
        """

        self.foodfeeling_embed_processor.fit(self.foodfeeling_dataset.load_foodfeeling_embed())
