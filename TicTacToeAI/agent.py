import numpy as np
from abc import ABC

"""
Difference between Q-learning and SARSA:
    - the way they update state action table is different:
      * Q-learning:
      * SARSA: 
      
"""

class BasedAgent(ABC):

    def __init__(self, alpha: float, gamma: float, epsilon: float):
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
        self.q_table = {} # store state-action values

    def _get_q_value(self, state: str, action: int):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        """

        """
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


class QLearningAgent(BasedAgent):

    def __init__(self, alpha: float, gamma: float, epsilon: float):
        BasedAgent.__init__(self, alpha, gamma, epsilon)

    def update_q_value(self, old_state: int, state: int, action: int, reward,
                       available_actions: [int]):
        old_q_value = self._get_q_value(state=old_state, action=action)
        max_next_q_value = max([self._get_q_value(state, next_action) for next_action in available_actions])
        self.q_table[(state, action)] = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_next_q_value)


class SARSAAgent(BasedAgent):

    def __init__(self, alpha, gamma, epsilon):
        BasedAgent.__init__(self, alpha, gamma, epsilon)

    def update_q_value(self, old_state: int, state: int, old_action: int,
                       action: int, reward, available_actions: [int]):
        current_q_value = self._get_q_value(old_state, old_action)
        target_q_value = reward + self.gamma * self._get_q_value(state, action)
        self.q_table[(old_state, old_action)] = (1 - self.alpha) * current_q_value + self.alpha * target_q_value
        
        

                
