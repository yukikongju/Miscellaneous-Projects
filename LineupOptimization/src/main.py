import pandas as pd
import numpy as np

from team import TeamAdjancy, TeamILP
    

def print_player_lineups(csv_path, lineups):
    df = pd.read_csv(csv_path)

    lineup_names = []
    for lineup in lineups:
        names = []
        for player_id in lineup:
            player_name = df.loc[df['player_id'] == player_id, 'player_name'].item()
            names.append(player_name)
        lineup_names.append(names)

    for lineup in lineup_names:
        print(lineup)

    return lineup_names


def main():
    csv_path = 'LineupOptimization/data/dummy_situation1.csv'
    ilp_csv_path = 'LineupOptimization/data/dummy_ilp1.csv'
    valk1_path = 'LineupOptimization/data/valk1.csv'

    #  team_kmeans = TeamAdjancy(valk1_path, player_method='teammates', graph_method='adjancy', lineup_method='spectral_clustering', cluster_method='kmeans')
    #  print(f"Lineup with KMeans Spectral Clustering Method: {team_kmeans.lineup}")

    # --- spectral clustering with fielder method
    team_fiedler = TeamAdjancy(csv_path, player_method='teammates', graph_method='adjancy', lineup_method='spectral_clustering', cluster_method='fiedler')
    print('------------------------------------------------')
    print(f"Lineup with Fielder Spectral Clustering Method:")
    print_player_lineups(valk1_path, team_fiedler.lineup)
    print('------------------------------------------------')



    #  team_ilp = TeamILP(ilp_csv_path,  'teammates+willingness+score', 1.0, 2.5)
    #  for p in team_ilp.players:
    #      print(p)



if __name__ == "__main__":
    main()

