# Lineup Optimization with Weighted Bipartite Matching Algorithms and Maximum Weighted Graph Cut

## Motivation

The best sport teams usually have a coaching staffs and trainers to help 
players optimize their training and recovery. The coaching staffs spend 
most of their time thinking about training, drills and plays and sometimes 
don't spend as much time to think about lineups, or if they do, their decisions
are most likely based on gut feeling rather than actual numbers.

Our goal will be to find the best lineup possible given: 
1. Players preferences: each player will list out the people they want to play with
2. Players pairing statistics: we will take into account completion rate and completion count between two players
3. Players Individual Statistics: we want the overall line to be the best possible based on individual player statistics

## Defining the problem

Since I play ultimate Frisbee, we will apply this problem to ultimate. We 
will simulate the following situations:

**Situation 1: Only with player teamates preferences**

Unlike in the AUDL, in the club scene, we don't have access to players 
statistics. Therefore, we have to perform lineup optimization only on player's
preferences. 

In this situation, we would like to split the team into two lines: one offensive 
line and one defensive line. Therefore, since we have one set of players and 
we want to split them into two distinct sets, we will apply algorithms for 
*maximum weighted graph cut*, mainly: 
- [X] Spectral Clustering:
    - [X] Graph Laplacian with KMeans: `graph_method = 'adjancy'; lineup_method = 'spectral_clustering'; cluster_method = 'kmeans'`
    - [X] Graph Laplacian with Fiedler (2nd eigenvector):  `graph_method = 'adjancy'; lineup_method = 'spectral_clustering'; cluster_method = 'fiedler'`
- [ ] Normalized Cut
- [ ] Stoer-Wagner Algorithm


**Situation 2: Only with player teamate preferences and positions**

In this situation, players will provide 2 types of information: 
1. Their teammates preferences: each player will provide the top n players with whom they would like to play with
2. Their preference playing offense vs defense: each player will rate their willingness to play offense and defense on a score of 10
3. Individual capacities: coaches will rate players offense and defense capabilities on a score of 10

To simplify the task, we will assume that a player can either be a offensive or 
defensive player, not both. We will use an Integer Linear Programming approach (ILP)
that will be solved with either CPLEX or Gurobi solver


**Situation 3: with player preferences and statistics**

[TODO]

## Ressources

- [Spectral Clustering](https://towardsdatascience.com/spectral-clustering-aba2640c0d5b)
- [Integer vs Linear Programming in Python](https://towardsdatascience.com/integer-programming-vs-linear-programming-in-python-f1be5bb4e60e)
- [Integer Programming in Python](https://towardsdatascience.com/integer-programming-in-python-1cbdfa240df2)
- [Linear Programming with Python](https://apmonitor.com/pdc/index.php/Main/LinearProgramming)



