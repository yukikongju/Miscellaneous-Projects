import math
import numpy as np
import pandas as pd
import gymnasium as gym

from UltimateField import UltimateGameResults


class UltimateFrisbeeEnv(gym.Env):


    def __init__(self, game_results: UltimateGameResults):
        """

        Parameters
        ----------
        action_space:
            > (throw_type, distance, angle)

        """
        self.game_results = game_results
        self.field_width = game_results.field_width
        self.field_length = game_results.field_length
        self.endzone_length = game_results.endzone_length
        #  self.offset_width = game_results.offset_width

        # define observation and action space
        self.n_actions = len(self.game_results.throws_distributions.throws_distribution)
        self.n_states = self.field_width * self.field_length
        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # init (throw_type, throw_side) <-> action mapping
        self.dct_action_to_throws, self.dct_throws_to_action = self._init_throw_action_mapping()

        # reset environment state
        self.state, self.info = self.reset()

    def reset(self):
        """
        Start catching the pull at a random position. 
        Assumptions: 
            - pull is always caught
            - pull is thrown in the last third of the field

        Parameters
        ----------
        state: (int, int)
            > disc position
        info: dict

        """
        # generate where the pull is caught 
        x = np.random.choice(range(self.field_width))
        y = np.random.choice(range(self.field_length // 3))
        state = (x, y)
        info = {}
        return state, info

    def _init_throw_action_mapping(self):
        """

        Returns
        -------
        dct_action_to_throws: dict => (action: int, (throw_type: str, throw_side: str))
            > dictionary that maps action to (throw_type, throw_side)
        dct_throws_to_action: dict => ((throw_type: str, throw_side: str), action: int)
            > dictionary that maps (throw_type, throw_side) to action
        """
        dct_action_to_throws = {action: throws for action, throws in enumerate(game.throws_distributions.throws_distribution.keys())}
        dct_throws_to_action = {throws: action for action, throws in dct_action_to_throws.items()}
        return dct_action_to_throws, dct_throws_to_action

    def get_type_side_map(self, throw_type: str, throw_side: str) -> int: 
        """
        Given throw_type and throw_side, return action number
        See get_action_type_side() for its inverse
        
        Example
        -------
        > get_type_side_map(throw_type='dump', throw_side='left')
        > 1
        """
        return self.dct_throws_to_action[(throw_type, throw_side)]


    def get_action_type_side(self, action: int) -> (str, str):
        """
        Given action, return throw_type and throw_side
        See get_type_side_map() for its inverse
        """
        return self.dct_action_to_throws[action]

    def get_current_throw_distribution(self, action: int, param: str):
        """
        Given action, return throw distribution for current state-action

        """

        x_pos, y_pos = self.state
        throw_type, throw_side = self.get_action_type_side(action)
        #  print(action, x_pos, y_pos, throw_type, throw_side)
        param_value = getattr(self.game_results.throws_distributions.throws_distribution[(throw_type, throw_side)][x_pos][y_pos], param).item()
        return param_value


    def step(self, action: int): # TO TEST
        """

        """
        # make the throw + determine if success or not + update state
        is_throw_successful = self._is_throw_successful(action)

        # give reward + determine if we are done (turnover or endzone) + info
        next_state, reward, done, info = self.__step(action, is_throw_successful)
        #  print(next_state)
        self.state = next_state

        return next_state, reward, done, info 

    def _is_throw_successful(self, action) -> bool:
        """ 
        Helper function for step() : determine whether a throw is successful or
        not

        Parameters
        ----------
        action: int

        Returns
        -------
        is_successful: bool
            > True if throw was successful (ie caught); False if turnover or stall
        """
        success_mean = self.get_current_throw_distribution(action, 'surface_proba_mean')
        success_var = self.get_current_throw_distribution(action, 'surface_proba_var')

        # what if proba > 1 or proba < 0
        expected_proba = np.random.normal(success_mean, np.sqrt(success_var))
        success_proba_upper = min(1.0, expected_proba)
        success_proba_lower = max(0.0, expected_proba)
        success_proba = min(success_proba_upper, success_proba_lower)
        #  print(success_proba)

        return np.random.uniform() <= success_proba


    def __step(self, action: int, is_throw_successful: bool):
        """
        Helper function for step() : return next_state, reward, done, info given a 
        successful throw

        Reward function:
        > -1 per throw
        > +200 per endzone completion
        > -200 per turnover

        Returns
        -------
        next_state: (int, int)
            > disc position after throw is caught
        reward: int
            > reward of successful pass or a throw
        done: bool
            > determine if point is over ie if a turnover occured or point scored
        info: dict
            > additional information
        """
        # 
        if not is_throw_successful:
            return self.state, -200, True, {'outcome': 'turnover'}

        # - determine next state from x_delta and y_delta
        x_delta_mean = self.get_current_throw_distribution(action, 'x_delta_mean')
        x_delta_var = self.get_current_throw_distribution(action, 'x_delta_var')
        y_delta_mean = self.get_current_throw_distribution(action, 'y_delta_mean')
        y_delta_var = self.get_current_throw_distribution(action, 'y_delta_var')

        # compute throw distance with normal distribution
        x_delta = math.floor(np.random.normal(x_delta_mean, np.sqrt(x_delta_var)))
        y_delta = math.floor(np.random.normal(y_delta_mean, np.sqrt(y_delta_var)))

        # make sure that throw is inbounds 
        x_pos, y_pos = self.state
        x_new_pos = max(0, min(x_pos + x_delta, self.field_width - 1))
        y_new_pos = max(0, min(y_pos + y_delta, self.field_length - 1))
        next_state = (x_new_pos, y_new_pos)
        #  print(next_state)

        # - verify if team has scored
        if (self.field_length - self.endzone_length < y_new_pos < self.field_length):
            #  print(self.field_length - self.endzone_length, y_new_pos, self.field_length)
            done = True
            reward = 100
            info = {'outcome': 'goal'}
        else: 
            done = False
            reward = -1
            info = {'outcome': 'in progress'}

        return next_state, reward, done, info
        

    def render(self, ): # TODO
        """

        """
        pass
        
