############ Final oddsportal scraper

# ATP, baseball, basket, darts, eSports, football, nfl, nhl, rugby
''' Create 4 main functions : scrape_historical, scrape_specific_season, scrape current_season, scrape_next_games
NB : You need to be in the right repository to import functions...'''
import os

from functions import *

print('Data will be saved in the following directory:', os.getcwd())


scrape_oddsportal_historical(sport = 'baseball', country = 'USA', league = 'mlb', start_season = '2010-2011', nseasons = 5, current_season = 'yes', max_page = 25)
# scrape_oddsportal_current_season(sport = 'baseball', country = 'finland', league = 'veikkausliiga', season = '2020', max_page = 25)
# scrape_oddsportal_specific_season(sport = 'baseball', country = 'finland', league = 'veikkausliiga', season = '2019', max_page = 25)
# scrape_oddsportal_next_games(sport = 'tennis', country = 'germany', league = 'exhibition-bett1-aces-berlin-women', season = '2020')





