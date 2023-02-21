"""
@auther ChaeHyunee
@since {4/10/2022}
@see :
"""
import sys
sys.path.append('/data')

from kocrawl.answerer.feeling_answerer import FeelingAnswerer
from kocrawl.base import BaseCrawler
from kocrawl.editor.feeling_editor import FeelingEditor
from kocrawl.searcher.feeling_searcher import FeelingSearcher #classification

from kocrawl.predict import Predictor

"""
searcher:
classification model에 entity 
DB 접근 / data set 받아옴

editor: 
data set을 저장한 dictionary를 slot을 채울 수 있는 string으로 변경

answerer:
dict를 바탕으로 출력메시지를 만들어 msg라는 하나의 문장 생성 후 반환


feeling.py 순서
분류된 entity를 받아서 searcher에 전달한다
searcher의 return 값인 (dictionary 형)게시물 DB 받음
    if 게시물 없으면 MapCrawler로 연결, return
게시물 있으면 
editor에 게시물 DB dict 입력 -> 모든 key값의 value가 slot에 채워지기 좋은 string으로 반환된 dict 받음
answerer로 달걀이 멘트 정리 후 하나의 string으로 반환 
"""

class FeelingCrawler():

    def request(self, feeling: str, x: float, y: float) -> str:
        """
        기분을 분류합니다.
        (try-catch로 에러가 나지 않는 함수)

        :param location: 지역
        :param place: 장소
        :return: 해당지역 장소
        """

        try:
            print(feeling, "x : {}, y : {}".format(x, y))
            return self.request_debug(feeling, x, y)[0]
        except Exception:
            return FeelingAnswerer().sorry(
                "해당 기분은 알 수 없습니다." + feeling
            )

    def request_dict(self, feeling: str, x: float, y: float):
        """
        기분을 분류합니다.
        (try-catch로 에러가 나지 않는 함수)

        :param location: 지역
        :param place: 장소
        :return: 해당지역 장소
        """

        try:
            return self.request_debug(feeling, x, y)[1]
        except Exception:
            return FeelingAnswerer().sorry(
                "해당 기분은 알 수 없습니다." + feeling
            )

    def request_debug(self, feeling: str, x: float, y: float) -> tuple:
        """
        기분을 분류합니다.
        (에러가 나는 디버깅용 함수)

        :param feeling: 사용자의 감정
        :param x: 장소의 경도
        :param y: 장소의 위도
        :return: 음식 게시물 추천
        """
        feel_result = Predictor().predict(feeling)
        print('feel_result : ', feel_result)
        result_dict = FeelingSearcher().search_JungGal_post(feel_result, x, y) #Searcher: feel_classification
        result = FeelingEditor().edit_feeling(feel_result, result_dict)
        result = FeelingAnswerer().feeling_form(result)
        return result, result_dict
