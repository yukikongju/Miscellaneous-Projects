import torch
import random
import numpy as np

from collections import deque
from game import Direction, Coordinate
from model import Linear_QNet, QTrainer
from game import SnakeGame

ACTION_MAPPING = {'UP': ['UP', 'LEFT', 'RIGHT'],
        'LEFT': ['LEFT', 'UP', 'DOWN'],
        'RIGHT': ['RIGHT', 'UP', 'DOWN'],
        'DOWN': ['DOWN', 'LEFT', 'RIGHT']}
ACTION_SPACE = ['UP', 'LEFT', 'RIGHT', 'DOWN']
ACTION_MAP = {'UP': 0, 'LEFT': 1, 'RIGHT':2, 'DOWN':3}

MAX_MEMORY = 10_000
BATCH_SIZE = 1000
LR = 0.001


class CustomAgent:

    def __init__(self, n_games=50, epsilon=0.6, gamma=0.9):
        self.n_games = n_games
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game): 
        """ 
        Function that return state of the snake

        """
        head = game.snake[0]

        point_l = Coordinate(head.x - 20, head.y)
        point_r = Coordinate(head.x + 20, head.y)
        point_u = Coordinate(head.x, head.y - 20)
        point_d = Coordinate(head.x, head.y + 20)

        dir_l = (game.direction == Direction.LEFT)
        dir_r = (game.direction == Direction.RIGHT)
        dir_u = (game.direction == Direction.UP)
        dir_d = (game.direction == Direction.DOWN)


        state = [
                # danger straight
                (dir_r and game._is_dead(point_r)) or 
                (dir_l and game._is_dead(point_l)) or 
                (dir_u and game._is_dead(point_u)) or 
                (dir_d and game._is_dead(point_d)),

                # danger right 
                (dir_r and game._is_dead(point_d)) or 
                (dir_l and game._is_dead(point_u)) or 
                (dir_u and game._is_dead(point_r)) or 
                (dir_d and game._is_dead(point_l)),

                # danger left
                (dir_r and game._is_dead(point_u)) or 
                (dir_l and game._is_dead(point_d)) or 
                (dir_u and game._is_dead(point_l)) or 
                (dir_d and game._is_dead(point_r)),

                # snake direction
                dir_l, 
                dir_r, 
                dir_u, 
                dir_d, 

                # food location
                game.food.x < game.head.x, # food left
                game.food.x > game.head.x, # food right
                game.food.y < game.head.y, # food up
                game.food.y > game.head.y  # food down
            ]

        return state


    def remember(self, state, action, reward, next_state, done):
        """ 
        pop left if MAX_MEMORY reached
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self): # FIXME
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, last_action):
        """ 
        we choose action based on decaying epsilon-greedy strategy
        """
        if random.random() > self.epsilon:
            if last_action is None:
                action = random.randrange(0, len(ACTION_SPACE))
            else:
                choices = ACTION_MAPPING.get(last_action)
                action_name = random.choices(choices, k=1)[0]
                action = ACTION_MAP[action_name]
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action = move

        return action


class Agent:

    def __init__(self, env):
        self.env = env

    def get_action(self):
        pass
        

class NaiveAgent(Agent):

    """ 
    Agent that choose action at random, but doesn't go into itself
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.last_action = None

    def get_action(self):
        """ 
        Choose action at random
        """
        if self.last_action:
            choices = ACTION_MAPPING.get(self.last_action)
        else: 
            choices = ACTION_SPACE.copy()

        action = random.sample(choices, k=1)[0]
        self.last_action = action

        return action

class LongRideAgent(Agent):

    """
    Agent that travel all the grid
    """

    def __init__(self, env=None):
        super().__init__(env)
        self.last_action = None

    def get_action(self):
        pass
        
        
def train():
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = CustomAgent()
    game = SnakeGame()

    last_action = None
    while True:
        state_old = agent.get_state(game)
        final_action_idx = agent.get_action(state_old, last_action)
        #  print(final_action)

        final_action = ACTION_SPACE[final_action_idx]
        reward, done, score = game.play_step(final_action)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_action_idx, reward, state_new, done)

        agent.remember(state_old, final_action_idx, reward, state_new, done)

        last_action = final_action

        if done:
            game.reset_game()
            last_action = None
            agent.n_games += 1
            agent.train_long_memory()

            if score > record: 
                record = score
                agent.model.save()

            print('Games', agent.n_games, 'Score', score, 'Record', record)

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)

            #  plot(scores, mean_scores)


if __name__ == "__main__":
    train()

