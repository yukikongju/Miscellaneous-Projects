# ELT Process for Game Stats, Player Stats and Season Stats

## Stating the Problem

We want to fetch all the data from the AUDL website for RL and network 
analysis.

- The endpoint for **game statistic** is a json: `https://www.backend.audlstats.com/stats-pages/game/2022-07-31-DET-MIN`
- The endpoint for **game stats leader categories**: `https://www.backend.audlstats.com/web-api/game-stats?gameID=2023-07-22-DAL-HTX` [ NOT IMPLEMENTED YET ]
- Stats for player: `https://www.backend.audlstats.com/web-api/roster-stats-for-player?playerID=abarcio`
    * In a given year gives stats per game per game ID: `https://www.backend.audlstats.com/web-api/roster-game-stats-for-player?playerID=jfroude&year=2023`

## Data to download

For RL:
- json files for all games for play-by-play events

For Network:
- Do better team share the disc or is it the same few players that always have the disc?
- Clusters of players that works well together
- Which players are the most important?
- How does the team evolve during the season? Throwing distribution? Node degrees?

For ML:
- Find all games id for that season
- teams season stats by game
- player season stats by game
- create jobs to fetch game stats

Problem with their API:
- player stats are being stored player level rather than by game, so for 
  every new game, we have to loop through all players to get players stats 
  instead of fetching that game


