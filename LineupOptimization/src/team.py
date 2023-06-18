import pandas as pd
import numpy as np

from player import SimplePlayer
from sklearn.cluster import KMeans
from abc import ABC

class Team(ABC):


    def __init__(self, csv_path):
        self.csv_path = csv_path

        self.players = []
        self.df = pd.read_csv(csv_path)


    def _set_players(self):
        pass

    def _set_lineup(self):
        pass


class TeamAdjancy(Team):

    """Determining lineup using adjancy graph"""

    def __init__(self, csv_path, player_method, graph_method, lineup_method, cluster_method):
        super(TeamAdjancy, self).__init__(csv_path)
        self.player_method = player_method
        self.graph_method = graph_method
        self.lineup_method = lineup_method
        self.cluster_method = cluster_method

        self._set_players()
        self._set_max_num_prefered_teammates()

        self.players_index_dict = {player.player_id: i for i, player in enumerate(self.players)}
        self._build_graph()
        self.set_lineups()

    def _set_max_num_prefered_teammates(self):
        self.max_num_prefered_teammates = 1
        for player in self.players:
            n = len(player.prefered_teamates)
            self.max_num_prefered_teammates = max(self.max_num_prefered_teammates, n)
        

    def _build_graph(self):
        if self.graph_method == 'adjancy':
            self.graph = self._build_uniform_adjancy_graph()
        else:
            raise ValueError(f"{self.graph_method} not supported. Select between ['adjancy', 'weighted_adjancy']")


    def _build_uniform_adjancy_graph(self):
        """ 
        Return numpy array representing weighted adjancy graph 
        """
        n = len(self.players)

        # adjancy graph
        A = np.zeros((n, n), dtype=int)

        for i, player in enumerate(self.players):
            for j, teammate_id in enumerate(player.prefered_teamates):
                teammate_idx = self.players_index_dict[teammate_id]
                A[i][teammate_idx] = self.max_num_prefered_teammates - j # weight

        return A
        

    def _set_players(self):
        if self.player_method == 'teamates':
            for _, row in self.df.iterrows():
                prefered_teamates = list(map(int, eval(row['prefered_teamates'])))
                player = SimplePlayer(row['player_id'], prefered_teamates)
                self.players.append(player)
        else:
            raise ValueError(f"{self.player_method} not supported. Select between [teamates, ]")

        

    def set_lineups(self):
        if self.lineup_method == 'spectral_clustering':
            self.lineup = self._get_spectral_clustering_lineup()
        else: 
            raise ValueError(f"{self.lineup_method} not supported. Select between [spectral_clustering]")


    def _get_spectral_clustering_lineup(self):
        """ 
        Find Lineup using spectral clustering algorithm:
        1. Build adjancy matrix
        2. Find degree matrix
        3. Calculate Graph Laplacian
        4. Calculate Eigenvalues of Graph Laplacian
        5. Sort based on Eigenvalues
        6. KMeans Clustering

        Problem with this algorithm: lines are not balanced ie some lines have 
        more players than others

        """
        # computing adjancy matrix (A), degree matrix (D), graph laplacian (L)
        A = self.graph
        D = np.diag(A.sum(axis=1))
        L = D-A

        # compute eigenvalues and eigenvectors - keep only real number
        vals, vecs = np.linalg.eig(L)
        vals, vecs = vals.real, vecs.real


        # sort based on eigenvalues
        vecs = vecs[:, np.argsort(vals)]
        vals = vals[np.argsort(vals)]

        NUM_LINES = 2
        if self.cluster_method == 'kmeans':
            kmeans = KMeans(n_clusters=NUM_LINES) # n_clusters = 2 because we want two lines
            kmeans.fit(vecs[:, 1:8])
            clusters = kmeans.labels_ # [0 1 0 1 1 1 0 0 0]
        elif self.cluster_method == 'fiedler':
            clusters = vecs[:, 1] > 0
            clusters = np.zeros((len(self.players), ), dtype=int) + clusters


        # get lineups
        lineups_dict = {line: [] for line in range(NUM_LINES)}
        for i, player in enumerate(clusters):
            lineups_dict[player].append(self.players[i].player_id)

        lineups = [lineups_dict[i] for i in range(NUM_LINES) ]
        return lineups

        
