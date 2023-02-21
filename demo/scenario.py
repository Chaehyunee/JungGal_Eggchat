"""
@auther Hyunwoong
@since 7/1/2020
@see https://github.com/gusdnd852
"""
import sys
sys.path.append('/data/EggChat/kochat')
sys.path.append('/data/kocrawl')
from dust import DustCrawler
from weather import WeatherCrawler
from app import Scenario
from map import MapCrawler
from feeling import FeelingCrawler
from foodname import FoodNameCrawler
from food_feeling import FoodFeelingTagger

weather = Scenario(
    intent='weather',
    api=WeatherCrawler().request,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

dust = Scenario(
    intent='dust',
    api=DustCrawler().request,
    scenario={
        'LOCATION': [],
        'DATE': ['오늘']
    }
)

restaurant = Scenario(
    intent='restaurant',
    api=MapCrawler().request,
    scenario={
        'LOCATION': [],
        'PLACE': ['맛집']
    }
)

travel = Scenario(
    intent='travel',
    api=MapCrawler().request,
    scenario={
        'LOCATION': [],
        'PLACE': ['관광지']
    }
)

feeling = Scenario(
    intent='feeling',
    api=FeelingCrawler().request,
    scenario={
        'FEELING': []
    }
)

foodname = Scenario(
    intent='foodname',
    api=FoodNameCrawler().request,
    scenario={
        'FOODNAME': []
    }
)

food_feeling = Scenario(
    intent='food_feeling',
    api=FoodFeelingTagger().request,
    scenario={
        'FOODNAME': []
    }
)