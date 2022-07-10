
""" 
Script that fetch all schedule and update to db
"""


from audl.stats.endpoints.seasonschedule import AllSchedule, SeasonSchedule
from utils import create_connection

# create connection
db_file = 'AUDL-exploration/databases/audl.db'
connection = create_connection(db_file)
print(connection)

# fetch all schedules
schedules = AllSchedule().get_schedule()


# store in db
schedules.to_sql(name='Schedule', con=connection, if_exists='replace', index=False)
connection.close()



