import pandas as pd
import numpy as np
import networkx as nx

from player import SimplePlayer, PlayerPreferencesWillingnessScore, CompletePlayer
from sklearn.cluster import KMeans
from abc import ABC

from pulp import *

class Team(ABC):


    def __init__(self, csv_path, player_method):
        self.csv_path = csv_path
        self.player_method = player_method

        self.players = []
        self.df = pd.read_csv(csv_path)


    def _set_players(self):
        if self.player_method == 'teammates':
            for _, row in self.df.iterrows():
                prefered_teammates = list(map(int, eval(row['prefered_teammates'])))
                player = SimplePlayer(row['player_id'], row['player_name'], prefered_teammates)
                self.players.append(player)
        elif self.player_method == 'teammates+willingness+score':
            for _, row in self.df.iterrows():
                prefered_teammates = list(map(int, eval(row['prefered_teammates'])))
                player = PlayerPreferencesWillingnessScore(row['player_id'], row['player_name'], prefered_teammates, row['offensive_willingness'], row['defensive_willingness'], row['offensive_score'], row['defensive_score'])
                self.players.append(player)
        elif self.player_method == 'complete_player':
            self._set_complete_players()
        else:
            raise ValueError(f"{self.player_method} not supported. Select between [teammates, ]")

    def _set_complete_players(self):
        """ if player_method == 'complete_player' """
        for _, row in self.df.iterrows():
            cutting_score = np.mean(eval(row['cutting_score'])) if 'cutting_score' in self.df.columns else None
            handling_score = np.mean(eval(row['handling_score'])) if 'handling_score' in self.df.columns else None
            defensive_score = np.mean(eval(row['defensive_score'])) if 'defensive_score' in self.df.columns else None
            offensive_score = np.mean(eval(row['offensive_score'])) if 'offensive_score' in self.df.columns else None
            preferred_teammates = eval(row['preferred_teammates']) if 'preferred_teammates' in self.df.columns else None
            offensive_willingness = row['offensive_willingness'] if 'offensive_willingness' in self.df.columns else None
            defensive_willingness = row['defensive_willingness'] if 'defensive_willingness' in self.df.columns else None

            player = CompletePlayer(player_id=row['player_id'],
                                    player_name=row['player_name'],
                                    teammate_preferences=preferred_teammates,
                                    offensive_willingness=offensive_willingness,
                                    defensive_willingness=defensive_willingness,
                                    offensive_score=offensive_score, 
                                    defensive_score=defensive_score,
                                    handling_score=handling_score,
                                    cutting_score=cutting_score)

            self.players.append(player)


    def _set_lineup(self):
        pass


class TeamAdjancy(Team):

    """Determining lineup using adjancy graph"""

    def __init__(self, csv_path, player_method, graph_method, lineup_method, cluster_method):
        super(TeamAdjancy, self).__init__(csv_path, player_method)
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


class TeamILP(Team):

    """
    Determining Lineups with Integer Linear Programming

    ILP Method Supported:
        1) Two Lines: using preferred teammates, handling score and cutting scores. equivalent strength [two-lines]
        2) Offensive/Defensive Lines: using preferred teammates, offense score and defense score [offensive-defensive-lines]
        3) Optimized Offensive/Defensive Lines: using preferred teammates, off/def scores, handling/cutting score [optimized-offensive-defensive-lines]

    """

    def __init__(self, players_csv_path: str, player_method: str, willingness_weight: float, score_weight: float, ilp_method: str):
        """ 
        Attributes
        ----------
        """
        super(TeamILP, self).__init__(players_csv_path, player_method)
        self.willingness_weight = willingness_weight
        self.score_weight = score_weight
        self.ilp_method = ilp_method

        self._set_players()
        self._set_digraph()
        self.indegree_centrality = nx.in_degree_centrality(self.G)

        self._set_lineup()


    def _set_digraph(self):
        self.G = nx.DiGraph()

        # add nodes to graph
        for player in self.players:
            self.G.add_node(player.player_name)

        # add edges -- TODO: weighted edges?
        for player in self.players:
            self.G.add_edges_from([ (player.player_name, teammate) for teammate in player.teammate_preferences])

    def _set_lineup(self):
        # -- set problem variables and constraints
        if self.ilp_method == 'two-lines':
            pass
        elif self.ilp_method == 'offensive-defensive-lines':
            prob = self.__get_ilp_problem_offense_defense_lines()
        elif self.ilp_method == 'optimized-offensive-defensive-lines':
            pass
        else:
            raise ValueError(f"{self.ilp_method} is not a valid method. please select between []")


    def __get_ilp_problem_offense_defense_lines(self):
        """
        set ILP problem if ilp_method == 'offense-defense-lines'.

        Problem Definition:
            - Ojective:
              + Maximize total sum given by offensive line and defensive line
            - Variables:
              + Binary variables for each player: either first line or second line
            - Constraints:
              + Number of players per lines set equally
              + Each player can only be in one line
        """
        # --- set variables
        offensive_line = LpVariable.dicts("offensive_line", [ player.player_name for player in self.players ], cat='Binary')
        defensive_line = LpVariable.dicts("defensive_line", [ player.player_name for player in self.players ], cat='Binary')

        # --- set problem objective
        prob = LpProblem('PlayerAssignment', LpMaximize)
        prob += lpSum(
                [ player.offensive_score * offensive_line[player.player_name] 
                 + player.defensive_score * defensive_line[player.player_name]
                 for player in self.players ]
                )

        # --- set constraints

        # constraint: lines have the same amount of players
        num_players = len(self.players)
        max_line_size = round(num_players / 2, 0)
        prob += lpSum(offensive_line[player.player_name] for player in self.players) <= max_line_size
        prob += lpSum(defensive_line[player.player_name] for player in self.players) <= max_line_size

         # constraint: each player can only be assigned to either offensive or defensive line
        for player in self.players:
            prob += offensive_line[player.player_name] + defensive_line[player.player_name] == 1

        # constraint: consider degree centrality given by teammate preference
        prob += offensive_line[player.player_name] + defensive_line[player.player_name] >= self.indegree_centrality[player.player_name]

        # --- solve problem 
        prob.solve()

        # --- check if solution has been found
        if prob.status == LpStatusOptimal:
            offensive_line_assignments = [ player.player_name for player in self.players if value(offensive_line[player.player_name]) == 1 ]
            defensive_line_assignments = [ player.player_name for player in self.players if value(defensive_line[player.player_name]) == 1 ]

        # -- set the lines
        print(f"Offensive: {offensive_line_assignments}")
        print(f"Defense: {defensive_line_assignments}")


