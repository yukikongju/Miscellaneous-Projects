# Notes

TODOs:
- [X] verify if throw sequence are correct; do we ever reset in a winning state?
- [o] implement agent: 
    - [X] q-learning
    - [ ] sarsa
- [ ] Debug Agent => why does q-function has value prior endzone?

Guideline:
- [X] 0. Define actions and state spaces
- [X] 1. Define environment: SimpleFrisbeeEnv
- [ ] 2. Define DQN
- [ ] 3. Generate Heatmap from states space
- [ ] 4. Fetch data for each AUDL team + generate heatmap for all frisbee teams


**State and Actions Space and Reward Function**

- State: position of the disc (x, y) [ discrete: 5x5 yards ]
- done: if goal or turn-over
- Actions: throw choice (under, huck, dish, swing, dump, upline, give-go) + side (open/break)
    * 2 transition proba: (1) open; (2) break
- reward: R = 1000 x (1 if goal else 0) + dist to endzone


**Compiling latex**

Compiled with `VimTex`

```
sudo apt-get install texlive-publishers
```

## Papers

- [ ] [DRL in racket sports for player evaluation with tactical contexts](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9775086)
- [ ] [DRL Framework for optimizing player decisions in soccer](https://www.sloansportsconference.com/research-papers/beyond-action-valuation-a-deep-reinforcement-learning-framework-for-optimizing-player-decisions-in-soccer)
- [ ] [Using RL to Evaluate Player Pair Performance in Ice Hockey](https://www.diva-portal.org/smash/get/diva2:1557998/FULLTEXT01.pdf)

