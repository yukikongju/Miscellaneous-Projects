# Notes


## The classes

- [Abstract Class] LineupOptimizer
    * Attributes:
	* optimizer_name, description
    * Functions:
	* get_lineups()
    * Situation 1: based on teammates preferences [Class] LineupTeamatesOptimizer
    * Situation 2: 
- [Class] Player
    * player_id: str
    * prefered_teamates: [player_id: str]
- [Class] Lineup
    * lineup_compositions: [player]
    * type: str (offense or defense)
    * score: double
- [Class] Team
    * player_method
	+ teamates
    * graph_method
    * lineup_method

## Adjancy Graph Alternatives

We have assumed that players will list the top n player with whom they would 
like to play with. What if they scored their willingness to play with each 
of their teammates instead?

## Lineup Algorithms


**Buddy System: Weighted Bipartite Graph with Hungarian Algorithm**

Given that we have two distinct lines (offensive and defensive), we want to 
pair players that won't be playing at the same time.


**With teamates preferences, off/def preference and coaches off/def assessment**

- Integer Linear Programming

Integer Linear Programming (ILP) model:

Decision Variables:
Let's assume we have N players. We introduce binary decision variables for each player, representing whether they are assigned to the offensive line (O) or the defensive line (D):

    x_i = 1 if player i is assigned to the offensive line
    x_i = 0 if player i is assigned to the defensive line

---

Objective Function:
The objective is to maximize the total offensive and defensive scores based on preferences and assessments. We assign weights to reflect the relative importance of each factor. Let p_i represent the offensive preference score for player i, q_i represent the defensive preference score for player i, s_i represent the offensive skill assessment by the coach for player i, and t_i represent the defensive skill assessment by the coach for player i. We define the objective function as:

Maximize: sum(p_i * x_i + q_i * (1 - x_i) + s_i * x_i + t_i * (1 - x_i)) for all players i

Constraints:
We need to ensure that each player is assigned to either the offensive line or the defensive line, but not both. We also need to include constraints related to the desired size of the lines and any other relevant requirements. Let's assume we want K players on the offensive line and M players on the defensive line. The constraints are:

    sum(x_i) = K for the offensive line
    sum(1 - x_i) = M for the defensive line
    x_i is binary (0 or 1) for all players i

Complete ILP Model:
Putting it all together, the complete ILP model can be written as follows:

```
Maximize
   sum(p_i * x_i + q_i * (1 - x_i) + s_i * x_i + t_i * (1 - x_i)) for all players i

Subject to
   sum(x_i) = K            (constraint for the size of the offensive line)
   sum(1 - x_i) = M        (constraint for the size of the defensive line)
   x_i in {0, 1}           (binary constraint) for all players i
```

---
