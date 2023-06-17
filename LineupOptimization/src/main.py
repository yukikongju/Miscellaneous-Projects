import pandas as pd
import numpy as np

from team import Team
    


def main():
    csv_path = 'LineupOptimization/data/dummy_situation1.csv'
    team_kmeans = Team(csv_path, player_method='teamates', graph_method='adjancy', 
                lineup_method='spectral_clustering', cluster_method='kmeans')
    team_fiedler = Team(csv_path, player_method='teamates', graph_method='adjancy', 
                lineup_method='spectral_clustering', cluster_method='fiedler')

    print(f"Lineup with Fielder Spectral Clustering Method: {team_fiedler.lineup}")
    print(f"Lineup with KMeans Spectral Clustering Method: {team_kmeans.lineup}")



if __name__ == "__main__":
    main()

