# AUDL RL

Learning team best strategy through reinforcement learning

## Mise en Contexte

The AUDL is a semi-professional ultimate frisbee league where 2 teams play against
each other and try to accumulate the most point. To score a point, the team must
catch the frisbee in the endzone.

I believe the game of ultimate frisbee in the highest level is markovian: 
given the disc position and the player position on the field, we can predict 
the probability that a team score the point. It also means that for every disc
position, there exist a best trajectory ie a set of throws each team can do 
to maximize their probability to score the point. The goal of this project is
to find such strategy using reinforcement learning.

## How the simulation is run

All the data will come from the [AUDL API] that fetches data from the official 
AUDL website. This API provide useful information such as the throwing 

For each throw open/break side, we consider (1) probability of completion 
(2) mean and variance given for distance/angle for each position on the field

**Types of throws considered**

- dump
- dish
- swing
- under
- huck

Notes:
- we cannot generate the throw based on mean and variance alone because it 
  doesn't reflect the decision making of the thrower. We have to make the 
  throw more difficult as the user has a less space to throw into.


completion_dict = {
""
}

## Algorithms used

- [Monte Carlo Methods](https://en.wikipedia.org/wiki/Reinforcement_learning)

## Ressources


