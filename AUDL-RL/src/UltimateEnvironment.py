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
        dct_action_to_throws, dct_throws_to_action = self._init_throw_action_mapping()

        # reset environment state
        self.reset()

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


    def step(self, action: int): # TODO
        """

        """
        # make the throw + determine if success or not

        # give reward + determine if we are done (turnover or endzone) + info
        pass
        

    def _is_done(self, ):
        pass


    def render(self, ):
        """

        """
        pass
        

        
