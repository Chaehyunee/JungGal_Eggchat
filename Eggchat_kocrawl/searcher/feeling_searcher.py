"""
@auther Hyunwoong
@since {6/21/2020}
@see : https://github.com/gusdnd852
"""
from random import randint

from kocrawl.searcher.base_searcher import BaseSearcher

# mysql 연결
import pymysql

class FeelingSearcher():

    def __init__(self):
        self.data_dict = {
            # 데이터를 담을 딕셔너리 구조를 정의합니다.
            'share_post_id': [], 'share_post_name':[],
            'share_story': [], 'share_post_region': [], 'share_time': []
        }
        

    def search_JungGal_post(self, feel_result: str, x: float, y: float) -> dict:
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
            sql = "select share_post_id, share_post_name, share_story, share_post_region, share_time  \
                from share_post where ST_Distance_Sphere(share_post_point, ST_GeomFromText('point({} {} )')) <=20000 and {}=1;".format(x, y, feel_result)

            #주의할 점/ x가 3자리, y가 2자리

            curs.execute(sql)

            # 데이터 fetch
            rows = curs.fetchone()
            print("rows : ",rows) # 전체 rows

            conn.close()

            

        except:
            print("error!")

        self.data_dict['share_post_id'].append(rows[0]) # 고유값 (게시물로 이동 위해)
        self.data_dict['share_post_name'].append(rows[1]) #나눔 게시물 제목 (음식 이름)
        self.data_dict['share_story'].append(rows[2])
        self.data_dict['share_post_region'].append(rows[3])
        self.data_dict['share_time'].append(str(rows[4])) # 나눔 종료 시각
        return self.data_dict