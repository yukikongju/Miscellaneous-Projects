
""" 
Script that fetch all team game stats from https://theaudl.com/stats/team-game-stats
and store them into db

Examples
--------
>>> python3 FETCH_TEAM_GAME_STATS.py 

TODO:
>>> python3 FETCH_TEAM_GAME_STATS.py AUDL-exploration/audl.cfg
"""

from utils import create_connection, load_config
from audl.stats.endpoints.teamgamestats import AllTeamGameStats

# Step 0: Load config file
config_file = 'AUDL-exploration/audl.cfg'
configs = load_config(config_file)
db_file = configs['database']['path']

# Step 1: Create SQL Connection
connection = create_connection(db_file)

# Step 2: Fetch Data
team_game_stats = AllTeamGameStats().get_game_stats()

# Step 3: save table to db
team_game_stats.to_sql(name='TeamGameStats', con=connection, if_exists='replace',
        index=False)


