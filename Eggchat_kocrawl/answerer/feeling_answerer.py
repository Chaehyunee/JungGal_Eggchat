"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
import sys
sys.path.append('/data/kocrawl')
from kocrawl.answerer.base_answerer import BaseAnswerer


class FeelingAnswerer(BaseAnswerer):

    def feeling_form(self, result: dict) -> str:
        """
        Feeling 출력 포맷

        :param share_post_id: 게시물 연결을 위한 고유값
        :param share_post_name: 게시물 제목 및 음식 명
        :param share_story: 게시물 내용
        :param share_post_region: 나눔 장소
        :param share_time: 나눔 시각
        :param result: 데이터 딕셔너리
        :return: 출력 메시지
        """
        print("\n\n여기는 Answerer : ", result)
        msg = self.feeling_init.format(feel_result = str(result['feel_result']))
        msg = self._add_msg_from_dict(result, 'share_post_name', msg, '{share_post_name}에 도전해보시는 건 어떤가요?')
        msg = self._add_msg_from_dict(result, 'share_story', msg, '나눔 하시는 분이 \n"{share_story}"\n라고 하시네요')
        msg = self._add_msg_from_dict(result, 'share_post_region', msg, '나눔은 {share_post_region}에서')
        msg = self._add_msg_from_dict(result, 'share_time', msg, '{share_time} 까지 진행한다고 해요. 도움이 되었으면 좋겠어요')
        msg = msg.format(share_post_name=result['share_post_name'], share_story=result['share_story'],
                         share_post_region=result['share_post_region'], share_time=result['share_time'])

        return msg
