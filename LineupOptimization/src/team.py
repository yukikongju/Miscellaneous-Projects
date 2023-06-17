import pandas as pd
import numpy as np

from player import SimplePlayer


class Team:


    def __init__(self, csv_path, player_method, graph_method, lineup_method):
        self.csv_path = csv_path
        self.player_method = player_method
        self.graph_method = graph_method
        self.lineup_method = lineup_method
        self.players = []

        self.df = pd.read_csv(csv_path)
        self._set_players()


    def _build_graph(self):
        """ 
        Given a pandas dataframe with columns 'player_id' (str) and 
        'prefered_teamates' ([str]), 
        return adjancy graph as numpy array
        """
        if self.graph_method == 'adjancy':
            pass
        else:
            raise ValueError(f"{self.graph_method} not supported. Select between ['adjancy', ]")



    def _set_players(self):
        if self.player_method == 'teamates':
            for _, row in self.df.iterrows():
                player = SimplePlayer(row['player_id'], row['prefered_teamates'])
                self.players.append(player)
        else:
            raise ValueError(f"{self.player_method} not supported. Select between [teamates, ]")

        

    def get_lineups(self):
        if self.lineup_method == 'spectral_clustering':
            pass
        else: 
            raise ValueError(f"{self.lineup_method} not supported. Select between [spectral_clustering]")


        
