import networkx as nx
from pulp import *

# Create a list of players with their preferences, offensive scores, defensive scores, coach assessments, and social connections
players = [
    {"name": "Player1", "teammate_preferences": ["Player3", "Player2", "Player4"],
     "offensive_score": 8, "defensive_score": 7, "coach_assessment": 9},
    {"name": "Player2", "teammate_preferences": ["Player1", "Player3", "Player4"],
     "offensive_score": 9, "defensive_score": 6, "coach_assessment": 8},
    {"name": "Player3", "teammate_preferences": ["Player1", "Player2", "Player4"],
     "offensive_score": 7, "defensive_score": 9, "coach_assessment": 7},
    {"name": "Player4", "teammate_preferences": ["Player2", "Player1", "Player3"],
     "offensive_score": 9, "defensive_score": 8, "coach_assessment": 8}
]

# Create a graph to represent social connections between players
G = nx.Graph()
for player in players:
    G.add_node(player["name"])
    G.add_edges_from([(player["name"], teammate) for teammate in player["teammate_preferences"]])

# Calculate the degree centrality of each player in the social network
degree_centrality = nx.degree_centrality(G)

# Create a binary variable for each player indicating their assignment to the offensive line
offensive_line = LpVariable.dicts("offensive_line", [player["name"] for player in players], cat='Binary')

# Create a binary variable for each player indicating their assignment to the defensive line
defensive_line = LpVariable.dicts("defensive_line", [player["name"] for player in players], cat='Binary')

# Create the ILP problem and set the objective as maximizing the total offensive and defensive scores
prob = LpProblem("PlayerAssignment", LpMaximize)
prob += lpSum(
    [player["offensive_score"] * offensive_line[player["name"]] +
     player["defensive_score"] * defensive_line[player["name"]]
     for player in players]
)

# Add constraints to ensure each player is assigned to either the offensive line or the defensive line
for player in players:
    prob += offensive_line[player["name"]] + defensive_line[player["name"]] == 1

# Add a constraint to consider degree centrality in the team composition
for player in players:
    prob += offensive_line[player["name"]] + defensive_line[player["name"]] >= degree_centrality[player["name"]]

# Solve the ILP problem
prob.solve()

# Check if an optimal solution is found
if prob.status == LpStatusOptimal:
    # Get the assignments of players to the offensive and defensive lines
    offensive_line_assignments = [player["name"] for player in players if value(offensive_line[player["name"]]) == 1]
    defensive_line_assignments = [player["name"] for player in players if value(defensive_line[player["name"]]) == 1]

    # Print the assignments
    print(f"Offense: {offensive_line_assignments}")
    print(f"Defense: {defensive_line_assignments}")


#  print(prob)

offensive_line_assignments = [player["name"] for player in players if value(offensive_line[player["name"]]) == 1]
defensive_line_assignments = [player["name"] for player in players if value(defensive_line[player["name"]]) == 1]

# Print the assignments
print("Offense: {offensive_line_assignments}")
print("Defense: {defensive_line_assignments}")


