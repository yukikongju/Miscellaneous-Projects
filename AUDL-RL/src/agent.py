import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC


class BaseAgent(ABC):

    def __init__(self, alpha: float, gamma: float, epsilon: float, 
                 field_length: int, field_width: int, n_actions: int):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.field_length = field_length
        self.field_width = field_width
        self.n_actions = n_actions

        # define value functions tables
        self.q_table = np.zeros((self.field_width, self.field_length, self.n_actions))
        self.v_table = np.zeros((self.field_width, self.field_length))


    def get_q_value(self, x_pos: int, y_pos: int, action: int):
        return self.q_table[x_pos][y_pos][action]

    def get_v_value(self, x_pos: int, y_pos: int):
        return self.v_table[x_pos][y_pos]

    def _set_q_table(self, x_pos: int, y_pos: int, action: int, val: int):
        self.q_table[x_pos][y_pos][action] = val

    def _set_v_table(self, x_pos: int, y_pos: int, val: int):
        self.v_table[x_pos][y_pos] = val
        

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon: # exploration
            return self._explore()
        else: # exploitation
            return self._exploit(state)
        
    def _exploit(self, state):
        x_pos, y_pos = state
        next_action = np.argmax([ self.get_q_value(x_pos, y_pos, a) for a in range(self.n_actions) ])
        return next_action

    def _explore(self):
        return np.random.choice(range(self.n_actions))

    def update_q_table(self):
        pass
        
    def update_v_table(self):
        pass
        
    def render_v_function(self): 
        plt.imshow(self.v_table, cmap='viridis', vmax=1)
        plt.title("QLearning State Value Function")
        plt.colorbar()
        plt.show()

    def render_q_function(self, action): 
        plt.imshow(self.q_table[:, :, action], cmap='viridis', vmax=1)
        plt.title("QLearning State-Action Value Function")
        plt.colorbar()
        plt.show()


class QLearningAgent(BaseAgent):


    def __init__(self, alpha: float, gamma: float, epsilon: float, 
                 field_length: int, field_width: int, n_actions: int):
        BaseAgent.__init__(self, alpha, gamma, epsilon, field_length, 
                           field_width, n_actions)

    def update_q_table(self, state, next_state, action, reward):
        x_pos_old, y_pos_old = state
        old_q_value = self.get_q_value(x_pos_old, y_pos_old, action)
        x_pos, y_pos = next_state
        max_next_q_value = max([self.get_q_value(x_pos, y_pos, a) for a in range(self.n_actions) ])
        updated_val = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_next_q_value)
        self._set_q_table(x_pos_old, y_pos_old, action, updated_val)

        
    def update_v_table(self, state, next_state, action, reward):
        x_pos_old, y_pos_old = state
        x_pos, y_pos = next_state
        next_v_value = self.get_v_value(x_pos, y_pos)
        old_v_value = self.get_v_value(x_pos_old, y_pos_old)
        val = (1 - self.alpha) * old_v_value + self.alpha * (reward + self.gamma * next_v_value)
        self._set_v_table(x_pos_old, y_pos_old, val)

        
class SarsaAgent(BaseAgent):


    def __init__(self, alpha: float, gamma: float, epsilon: float, 
                 field_length: int, field_width: int, n_actions: int):
        BaseAgent.__init__(self, alpha, gamma, epsilon, field_length, 
                           field_width, n_actions)

    def update_q_table(self):
        pass
        
    def update_v_table(self):
        pass


