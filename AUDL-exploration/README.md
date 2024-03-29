# AUDL Exploration

EDA for AUDL stats

[AUDL Analytics Pipeline](https://docs.google.com/drawings/d/1IdWRcp2mRWDZX7EwqnIUUZ3jFeHZ4ynZm599IL8uahc/edit)


# Requirements

```bash
pip install -r requirements.txt
```

# Structure

```markdown
├── data: all stats
│   ├── player_stats: data fetched from https://theaudl.com/stats/player-stats
│   └── team_stats: data fetched from https://theaudl.com/stats/team
├── exploration: jupyter notebook
├── reports: markdown and pdf reports 
├── README.md
```

# Exploration


**Player Stats**

- [ ] 


**Team Stats**

- [ ] Roaster offense/defense composition: who are the offense/defense player
- [ ] 


## TODOs

- [X] Create config files
- [ ] create templates directories

Scripts:
- [X] Schedule: Fetch all schedule
- [X] Team Game Stats
- [ ] Player Stats: get player regular and playoffs career
- [ ] Team Stats (by season, career) (need to code agg team + opponent first)
- [ ] Player Game Stats: get_roster_stats() for all games or get_game_stats() for 
      each player (how to update efficiently)


Workflows:
- [ ] Game Finder: Update Schedule every season
- [ ] Team Game Stats: Update game stats every game in season
- [ ] Player Game Stats: Update every game in season
- [ ] Player Season and Playoffs Stats
- [ ] 

## Technologies to be used

- [ ] Data Orchestration: `Dagster` and `Github Actions`
- [ ] Database: 
	* Document-Based for JSON file: `MongoDB`
	* Graph Database: `GraphQL` or `Neo4j` or `graphene`
	* Caching: `Redis` ; `SnowFlake`
	* `Databricks`
- [ ] CI/CD: `Github Actions` or `Terraform`
- [ ] Retraining ML Models: `SageMaker`
- [ ] Containerization: `Docker`
- [ ] MLOps: `MLFlow` or `Airflow`

**Optional**

- [ ] Drift Detection
- [ ] Hyperparameters Tuning


# How to

**Read from Sqlite3**

```bash
sqlite3 audl.db 'select * from schedule' >> tmp.csv
```



## Ressources
