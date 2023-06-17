import pandas as pd
import numpy as np
#  from abc import ABC

from team import Team
    


def main():
    csv_path = 'LineupOptimization/data/dummy_situation1.csv'
    team = Team(csv_path, player_method='teamates', graph_method='adjancy', 
                lineup_method='spectral_clustering')
    print(team.df)


if __name__ == "__main__":
    main()

