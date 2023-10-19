# Information Spread

How does the information spread throughout a social network? We will try to 
answer this question by running a simulation.


**How the simulation works**

- We initialize a closed system with n people in it
- Each person has a given number of people in their inner circle. The number 
  of friends will follow a normal distribution
- Each time step, if a person has information, they can either share that 
  information or keep it to themselves

**The simulations**


- Simulation 1:
    - Closed System with N number of people in it
    - Each person has M friends, where M is a random variable following a normal 
      distribution
    - The information to be shared is ABSOLUTE. There is only one piece of 
      information to be shared and cannot be shared partially
    - Each person is being assigned a parameter p which represent the probability 
      of sharing an information
    - Relationship are not symmetrical: if A considers B as a friends, the inverse is not necessarily true


- Simulation 2:
    - Closed System with N number of people in it
    - Each person has M friends, where M is a random variable following a normal 
      distribution
    - The information to be shared is ABSOLUTE. There is only one piece of 
      information to be shared and cannot be shared partially
    - Each person has a 'trustworthiness' level and 'p' their willingness of 
      sharing information.
      Every time a person share the information, their trustworthiness level go 
      down. If a person is being shared information and doesn't share it,
      their score go up. To determine whether someone will share information, 
      we define a function parametrized by [TODO]
      

- Simulation 3:
    - Closed System with N number of people in it
    - Each person has M friends, where M is a random variable following a normal 
      distribution
    - The information to be shared is ABSOLUTE. There is only one piece of 
      information to be shared and cannot be shared partially
    - There is a 'trust' score that exist between each people in the graph. 
      Every time someone decide to share a piece of information to someone else,
      the trust score between those two individual increase
      

More simulation:
- friend group may expand as simulation is running
- we can share partial information and misinformation
- sharing information between groups


**The people**

- Person 1: probability of sharing information follows a Bernoulli
- Person 2: probability of sharing information is exponential 
    * if more people knows, more likely to share the information
- Person 3: probability of sharing + trustworthiness level

## Vim and C++

**Using bear and ccls to make vim recognize include files**

```
> sudo apt install bear
> bear make // this will create a compile_commands.json
> vi .ccls
> ctags -R .
```


## Ressources

- [C++ Reference](https://cplusplus.com/reference/)
- [Vim and C++](https://blog.octoco.ltd/vim-coc-and-c-dbe99405f7bd)

