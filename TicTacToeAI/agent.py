import numpy as np
from abc import ABC
from collections import defaultdict

"""
Difference between Q-learning and SARSA:
    - the way they update state action table is different:
      * Q-learning:
      * SARSA: 
      
"""

class MinimaxAgent():

    def __init__(self, ):
        pass
        


class RandomAgent():

    def __init__(self, ):
        pass

    def choose_action(self, available_moves):
        return np.random.choice(available_moves)


class BasedAgent(ABC):

    def __init__(self, alpha: float, gamma: float, epsilon: float, 
                 num_states: int, num_actions: int):
        """

        Parameters
        ----------
        alpha: float
            > learning rate
        gamma: float
            > discount factor
        epsilon: float
            > exploration rate
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_states = num_states
        self.num_actions = num_actions

        self.q_table = {} # store state-action values
        self.v_table = {} # store state value function
        self.state_counts = defaultdict(int) # store number of times we have explored each state

    def _get_q_value(self, state: str, action: int):
        return self.q_table.get((state, action), 0.0)

    def _get_v_value(self, state):
        return self.v_table.get(state, 0.0)
        

    def choose_action(self, state, available_actions):
        """

        """
        # count number of time we end up in state
        self.state_counts[state] += 1

        # select an action based on exploration or exploitation
        if np.random.uniform() < self.epsilon: # exploration
            return np.random.choice(available_actions)
        else: # exploitation
            q_values = np.array([(action, self._get_q_value(state, action)) for action in available_actions], dtype=[('action', int), ('value', float)])

            #  return available_actions[np.argmax(q_values)]
            max_index = np.argmax(q_values['value'])
            max_action = q_values['action'][max_index]

            return max_action

    def update_q_value(self, old_state: int, state: int, action: int, reward,
                       available_actions: [int]):
        pass

    def update_v_value(self, old_state: int, state: int, action: int, reward):
        pass


class QLearningAgent(BasedAgent):

    def __init__(self, alpha: float, gamma: float, epsilon: float,
                 num_states: int, num_actions: int):
        BasedAgent.__init__(self, alpha, gamma, epsilon, num_states, num_actions)

    def update_q_value(self, old_state: int, state: int, action: int, reward,
                       available_actions: [int]):
        old_q_value = self._get_q_value(state=old_state, action=action)
        max_next_q_value = max([self._get_q_value(state, next_action) for next_action in available_actions])
        self.q_table[(state, action)] = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_next_q_value)

    def update_v_value(self, old_state: int, state: int, action: int, reward):
        old_v_value = self._get_v_value(old_state)
        new_v_value = reward + self.gamma * self._get_v_value(state)
        self.v_table[old_state] = (1 - self.alpha) * old_v_value + self.alpha * new_v_value


class SARSAAgent(BasedAgent):

    def __init__(self, alpha, gamma, epsilon, num_states, num_actions):
        BasedAgent.__init__(self, alpha, gamma, epsilon, num_states, num_actions)

    def update_q_value(self, old_state: int, state: int, old_action: int,
                       action: int, reward, available_actions: [int]):
        current_q_value = self._get_q_value(old_state, old_action)
        target_q_value = reward + self.gamma * self._get_q_value(state, action)
        self.q_table[(old_state, old_action)] = (1 - self.alpha) * current_q_value + self.alpha * target_q_value
        
    def update_v_value(self, old_state: int, state: int, old_action: int, 
                       action: int, reward):
        old_v_value = self._get_v_value(old_state)
        target_v_value = reward + self.gamma * self._get_v_value(state)
        self.v_table[old_state] = (1 - self.alpha) * old_v_value + self.alpha * target_v_value
        

                
