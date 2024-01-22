import numpy as np
import gymnasium as gym

class UltimateFrisbeeEnv(gym.Env):


    def __init__(self, completion_dict):
        """

        Parameters
        ----------
        action_space:
            > (throw_type, distance, angle)

        """
        self.FIELD_LENGTH = 120
        self.FIELD_WIDTH = 80
        self.ENDZONE_LENGTH = 20
        self.completion_dict = completion_dict

        self.observation_space = gym.spaces.Discrete(self.FIELD_WIDTH * self.FIELD_LENGTH)
        #  self.action_space = gym.spaces.Box()

        self.reset()

    def reset(self, ):
        """

        Parameters
        ----------
        state: (int, int)
            > disc position

        """
        self.info = {}


    def step(self, action):
        """

        """
        # make the throw

        # give reward + determine if we are done
        pass
        

    def _is_done(self, ):
        pass


    def render(self, ):
        """

        """
        pass
        

        
