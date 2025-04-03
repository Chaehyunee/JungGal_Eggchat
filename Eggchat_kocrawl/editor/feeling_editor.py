"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from kocrawl.editor.base_editor import BaseEditor
import re


class FeelingEditor(BaseEditor):

    def edit_feeling(self, feel_result: str, result: dict) -> dict:
        """
        join_dict를 사용하여 딕셔너리에 있는 string 배열들을
        하나의 string으로 join합니다.

        :param share_post_id: 게시물 연결을 위한 고유값
        :param share_post_name: 게시물 제목 및 음식 명
        :param share_story: 게시물 내용
        :param share_post_region: 나눔 장소
        :param share_time: 나눔 시각
        :param result: 데이터 딕셔너리
        :return: 수정된 딕셔너리
        """

        feel_result = self.feeling[feel_result] # 대체 문구로 수정
        share_post_id = result['share_post_id'][0]
        share_post_name = result['share_post_name'][0]
        share_story = result['share_story'][0]
        share_post_region = result['share_post_region'][0]
        share_time = result['share_time'][0]

        result = {
            'feel_result': feel_result,
            'share_post_id': share_post_id, 
            'share_post_name': share_post_name,
            'share_story': share_story, 
            'share_post_region': share_post_region,
            'share_time': share_time
        }


        return result
