"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from kocrawl.editor.base_editor import BaseEditor
import re


class MapEditor(BaseEditor):

    def edit_map(self, location: str, place: str, result: dict) -> dict:
        """
        join_dict를 사용하여 딕셔너리에 있는 string 배열들을
        하나의 string으로 join합니다.

        :param location: 지역
        :param place: 장소
        :param result: 데이터 딕셔너리
        :return: 수정된 딕셔너리
        """

        result = self.join_dict(result, 'name') #result의 'name'이 key인 string 타입 value들을 join해주어 저장함
        result = self.join_dict(result, 'context')#'context': ["데이트" "행복해"]를 "데이트 행복해"로 바꾸어주고
        result = self.join_dict(result, 'category')
        result = self.join_dict(result, 'address')
        result = self.join_dict(result, 'thumUrl')

        if isinstance(result['context'], str):
            result['context'] = re.sub(' ', ', ', result['context'])
            #' '를 쉼표로 바꾸는 작업 (result['context']내에 있는 것을)
            #다시 "데이트, 행복해" 로 변경해주는 작업

        return result
