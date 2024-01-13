import gymnasium as gym
import numpy as np

class TicTacToeEnv(gym.Env):


    def __init__(self, show=False):
        """
        Initializing Tic Tac Toe Environment

        Number of combinations: 3**n**2 (n=3) 
            > each cell can either be (1) empty, (2) has 'X' (3) has 'O'
        """
        self.show = show
        self.board_width = 3
        self.n_actions = self.board_width * self.board_width
        self.n_states = 19683 # number of combinations: 3**n**2 (n=3) 
        self.action_space = gym.spaces.Discrete(self.n_actions)
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.p1 = 1
        self.p2 = 2

        self.reset()

    def reset(self):
        """
        reset the board
        """
        self.available_moves = set(range(self.n_actions))
        self.state = np.zeros((self.board_width, self.board_width), dtype=int)
        self.is_p1_turn = True
        self.info = {"players": {1: {"actions": []}, 2: {"actions": []}}}
        return self.state.flatten(), self.info

    def step(self, action: int):
        """

        Parameters
        ----------
        action: int
            > position user choose to play [0-8]

        Reward function
        ---------------
            > -1 whenever player play a move
            > +/- 100 whenever player wins

        """
        # 1. check if player's move is valid
        self._play(action)

        # 2. assign reward
        reward = -1

        # 3. check if game is done
        done = self._is_winner()
        if done: 
            reward += 100

        if self.show:
            self.render()

        # switch turn
        self.is_p1_turn = not self.is_p1_turn

        return self.state, reward, done, self.info

    def _play(self, action):
        """
        Update board given player action
        """
        if not self.action_space.contains(action):
            raise ValueError(f"{action} is not in action_space")

        (row, col) = action // self.board_width, action % self.board_width
        if self.state[row, col] != 0:
            raise ValueError(f"This position has already been played!")

        self.available_moves.remove(action)
        self.state[row, col] = self.p1 if self.is_p1_turn else self.p2
        if self.is_p1_turn:
            self.info["players"][self.p1]["actions"].append(action)
        else:
            self.info["players"][self.p2]["actions"].append(action)



    def _is_winner(self) -> bool:
        """
        Check if game is done
        """
        done = False
        player = self.p1 if self.is_p1_turn else self.p2

        # vertical
        for i in range(self.board_width):
            done |= all(c == player for c in self.state[i, :])

        # horizontal
        for j in range(self.board_width):
            done |= all(c == player for c in self.state[:, j])
        
        # diagonal
        diago1 = [self.state[i, i] for i in range(self.board_width)]
        diago2 = [self.state[self.board_width-i-1, i] for i in range(self.board_width)]
        done |= all(c == player for c in diago1)
        done |= all(c == player for c in diago2)


        # check if all cell are filled ie Tie
        filled = all(cell != 0 for cell in self.state.flatten())

        # add info 
        if done:
            self.info["winner"] = player
        elif filled:
            self.info["winner"] = 'tie'


        return done | filled


    def render(self):
        """
        Print the board

        """
        for i in range(self.board_width):
            row = []
            for j in range(self.board_width):
                if self.state[i, j] == 0:
                    row.append(self.board_width*i+j)
                elif self.state[i, j] == self.p1:
                    row.append('X')
                else:
                    row.append('O')
            print('|' + '|'.join(map(str, row)) + '|')

