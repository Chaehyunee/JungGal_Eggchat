"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from random import randint

from kocrawl.searcher.base_searcher import BaseSearcher

# mysql 연결
import pymysql

class FoodFeelingSearcher():

    def __init__(self):
        self.data_dict = {
            # 데이터를 담을 딕셔너리 구조를 정의합니다.
            'share_post_id': [], 'share_post_name':[],
            'share_story': [], 'share_post_region': [], 'share_time': []
        }
        

    def set_share_post_feeling(self, feeling: list, user_id: str):
        """
        DB랑 통신하는 방법 찾아서
        data_dict에 담고 반환

        :param share_post_id: 게시물 연결을 위한 고유값
        :param share_post_name: 게시물 제목 및 음식 명
        :param share_story: 게시물 내용
        :param share_post_region: 나눔 장소
        :param share_time: 나눔 시각
        """
        try:
            conn = pymysql.connect(
                host='203.255.3.237',
                user='chaehyun',
                password='123456',
                db='junggal_v2_database',
                charset='utf8')

            # Connection 으로부터 Cursor 생성
            curs = conn.cursor()

            #SQL문 실행 및 Fetch
            user_id = 'admin@gmail.com'
            feeling = [1, 1, 0, 0]
            sql = "update share_post set good={}, sad={}, tired={}, stress={} \
                where user_id = '{}' order by post_time desc limit 1;"\
                    .format(feeling[0], feeling[1], feeling[2], feeling[3], user_id)

            #주의할 점/ x가 3자리, y가 2자리

            curs.execute(sql)
            print("success!")

            conn.close()

            

        except:
            print("error!")
