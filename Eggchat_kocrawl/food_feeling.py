"""
@auther ChaeHyunee
@since {4/10/2022}
@see :
"""
import sys
sys.path.append('/data')
sys.path.append('/data/kocrawl')
sys.path.append('/data/EggChat/kochat/proc')

from kocrawl.answerer.base_answerer import BaseAnswerer
from kocrawl.base import BaseCrawler
from kocrawl.searcher.food_feeling_searcher import FoodFeelingSearcher

from food_feeling_classifier import FoodFeelingClassifier


"""
searcher:
classification model에 entity 
DB 접근 / data set 받아옴

editor: 
data set을 저장한 dictionary를 slot을 채울 수 있는 string으로 변경

answerer:
dict를 바탕으로 출력메시지를 만들어 msg라는 하나의 문장 생성 후 반환


foodname.py 순서
분류된 entity를 받아서 searcher에 전달한다
searcher의 return 값인 (dictionary 형)게시물 DB 받음
    if 게시물 없으면 MapCrawler로 연결, return
게시물 있으면 
editor에 게시물 DB dict 입력 -> 모든 key값의 value가 slot에 채워지기 좋은 string으로 반환된 dict 받음
answerer로 달걀이 멘트 정리 후 하나의 string으로 반환 
"""

class FoodFeelingTagger():

    def request(self, foodname: str, user_id:str) -> str:
        """
        음식명에 해당하는 게시물을 찾습니다.
        (try-catch로 에러가 나지 않는 함수)

        :param foodname: 음식명으로 뽑힌 Entity
        :param user_id: 게시물 입력한 사용자의 user_id
        :param feeling: good, sad, tired, stress에 해당하는 boolean(1 or 0) 값을 담은 list
        :return: 성공 시 'success' 반환
        """

        try:
            print(foodname)
            return self.request_debug(foodname)
        except Exception:
            return BaseAnswerer().sorry(
                "근처에 {}에 해당하는 게시물이 없습니다.".format(foodname)
            )

    def request_dict(self, foodname: str, user_id:str):
        """
        음식명에 해당하는 게시물을 찾습니다.
        (try-catch로 에러가 나지 않는 함수)

        :param foodname: 음식명으로 뽑힌 Entity
        :param user_id: 게시물 입력한 사용자의 user_id
        :param feeling: good, sad, tired, stress에 해당하는 boolean(1 or 0) 값을 담은 list
        :return: 성공 시 'success' 반환
        """

        try:
            return self.request_debug(foodname)
        except Exception:
            return BaseAnswerer().sorry(
                "근처에 {}에 해당하는 게시물이 없습니다.".format(foodname)
            )

    def request_debug(self, foodnames: list, user_id:str) -> str:
        """
        음식명에 해당하는 게시물을 찾습니다.
        (에러가 나는 디버깅용 함수)

        :param foodnames: 음식명으로 뽑힌 Entity 의 list (str 형식)
        :param user_id: 게시물 입력한 사용자의 user_id
        :param feeling: good, sad, tired, stress에 해당하는 boolean(1 or 0) 값을 담은 list
        :return: 성공 시 'success' 반환
        """
        print("request_debug> foodname: ", foodname)
        # foodname = ""
        # foodname.append(f for f in foodnames) # 음식명으로 분류된 entity (str) 들의 list를 하나의 str으로 합쳐야 함
        # # 합쳐진 foodname을 Tensor로 바꾸기 위해서
        # # 임베딩해줘야 한다고 생각, sequence: Tensor
        # feeling = FoodFeelingClassifier().predict(foodname) #TODO Classifier predict 연결 필요
        # # feeling = [1, 1, 0, 0] (예시)
        # print("request_debug> feeling Classifier result: ", feeling)

        # is_success = FoodFeelingSearcher().set_share_post_feeling(feeling, user_id)
        is_success = "success"
        return is_success
