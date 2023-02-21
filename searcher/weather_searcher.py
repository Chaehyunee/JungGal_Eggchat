from kocrawl.searcher.base_searcher import BaseSearcher
import re


class WeatherSearcher(BaseSearcher):

    def __init__(self):
        self.CSS = {
            # 검색에 사용할 CSS 셀렉터들을 정의합니다.
            'naver_weather': '.weather_main > i > .blind',
            'naver_temperature': '.temperature_text > strong',
            'naver_today_comparison' : '.status_wrap > .temperature_info > .summary',
            'google_weather': '#wob_dcp > #wob_dc',
            'google_temperature': '#wob_tm'
        }

        self.data_dict = {
            # 데이터를 담을 딕셔너리 구조를 정의합니다.
            'today_weather': None,
            'tomorrow_morning_weather': None,
            'tomorrow_afternoon_weather': None,
            'after_morning_weather': None,
            'after_afternoon_weather': None,
            'specific_weather': None,
            'today_temperature': None,
            'tomorrow_morning_temperature': None,
            'tomorrow_afternoon_temperature': None,
            'after_morning_temperature': None,
            'after_afternoon_temperature': None,
            'specific_temperature': None,
            'today_comparison' : None,
        }


    def _make_query(self, location: str) -> str:
        """
        검색할 쿼리를 만듭니다.
        
        :param location: 지역
        :param date: 날짜
        :return: "지역 날짜 날씨"로 만들어진 쿼리
        """

        return ' '.join([location, '날씨'])

    def _make_query_google(self, location: str, date: str) -> str:
        """
        google 전용
        """
        return ' '.join([location, date, '날씨'])

    def naver_search(self, location: str) -> dict:
        """
        네이버를 이용해 날씨를 검색합니다.

        :param location: 지역
        :return: 크롤링된 내용
        """
        query = self._make_query(location)  # 한번 서치에 전부 가져옴
        result = self._bs4_contents(self.url['naver'],
                                    selectors=[self.CSS['naver_weather'],
                                               self.CSS['naver_temperature'],
                                               self.CSS['naver_today_comparison']
                                               ],
                                    query=query)

        """result 평활화 시키는 작업이 필요할수도 있음"""

        i = 0
        for k in self.data_dict.keys():
            if 'specific' not in k:
                # specific 빼고 전부 담음
                self.data_dict[k] = re.sub(' ', '', result[i][0]) #re.sub는 문자열 치환. ' '없애줌
                i += 1

        return self.data_dict

    def google_search(self, location: str, date: str) -> dict:
        """
        구글을 이용해 날씨를 검색합니다.

        :param location: 지역
        :param date: 날짜
        :return: 크롤링된 내용
        """

        query = self._make_query_google(location, date)  # 날짜마다 따로 가져와야함
        result = self._bs4_contents(self.url['google'],
                                    selectors=[self.CSS['google_weather'],
                                               self.CSS['google_temperature']],
                                    query=query)

        self.data_dict['specific_weather'] = re.sub(' ', '', result[0][0])
        self.data_dict['specific_temperature'] = re.sub(' ', '', result[1][0])
        # specific만 담음
        print(self.data_dict)
        return self.data_dict


weather = WeatherSearcher()
