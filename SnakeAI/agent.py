import random

ACTION_MAPPING = {'UP': ['UP', 'LEFT', 'RIGHT'],
        'LEFT': ['LEFT', 'UP', 'DOWN'],
        'RIGHT': ['RIGHT', 'UP', 'DOWN'],
        'DOWN': ['DOWN', 'LEFT', 'RIGHT']}
ACTION_SPACE = ['UP', 'LEFT', 'RIGHT', 'DOWN']

class NaiveAgent:

    """ 
    Agent that choose action at random
    """

    def __init__(self, env=None):
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


        
