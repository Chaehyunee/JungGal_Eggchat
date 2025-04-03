"""
@auther ChaeHyunee
@since {4/10/2022}
@see :
"""
import sys
sys.path.append('/data')

from kocrawl.answerer.foodname_answerer import FoodnameAnswerer
from kocrawl.base import BaseCrawler
from kocrawl.editor.foodname_editor import FoodnameEditor
from kocrawl.searcher.foodname_searcher import FoodnameSearcher


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

class FoodNameCrawler():

    def request(self, foodname: str, x: float, y: float) -> str:
        """
        음식명에 해당하는 게시물을 찾습니다.
        (try-catch로 에러가 나지 않는 함수)

        :param foodname: 검색할 음식명
        :param x: 장소의 경도
        :param y: 장소의 위도
        :return: 음식 게시물 추천
        """

        try:
            print(foodname, "x : {}, y : {}".format(x, y))
            return self.request_debug(foodname, x, y)[0]
        except Exception:
            return FoodnameAnswerer().sorry(
                "근처에 {}에 해당하는 게시물이 없습니다.".format(foodname)
            )

    def request_dict(self, foodname: str, x: float, y: float):
        """
        음식명에 해당하는 게시물을 찾습니다.
        (try-catch로 에러가 나지 않는 함수)

        :param foodname: 검색할 음식명
        :param x: 장소의 경도
        :param y: 장소의 위도
        :return: 음식 게시물 추천
        """

        try:
            return self.request_debug(foodname, x, y)[1]
        except Exception:
            return FoodnameAnswerer().sorry(
                "근처에 {}에 해당하는 게시물이 없습니다.".format(foodname)
            )

    def request_debug(self, foodname: str, x: float, y: float) -> tuple:
        """
        음식명에 해당하는 게시물을 찾습니다.
        (에러가 나는 디버깅용 함수)

        :param foodname: 검색할 음식명
        :param x: 장소의 경도
        :param y: 장소의 위도
        :return: 음식 게시물 추천
        """
        result_dict = FoodnameSearcher().search_foodname(foodname, x, y) #Searcher: feel_classification
        result = FoodnameEditor().edit_foodname(foodname, result_dict)
        result = FoodnameAnswerer().foodname_form(result)
        return result, result_dict
