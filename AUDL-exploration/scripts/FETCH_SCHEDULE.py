
""" 
Script that fetch all schedule and update to db

Examples
--------
>>> python3 FETCH_SCHEDULE.py 

"""


from audl.stats.endpoints.seasonschedule import AllSchedule, SeasonSchedule
from utils import create_connection

# Step 0: Load config file
config_file = 'AUDL-exploration/audl.cfg'
configs = load_config(config_file)
db_file = configs['database']['path']

# Step 1: Create SQL Connection
connection = create_connection(db_file)

# Step 2: Fetch Data => fetching schedule
schedules = AllSchedule().get_schedule()

# Step 3: save table to db
schedules.to_sql(name='Schedule', con=connection, if_exists='replace', index=False)
connection.close()



