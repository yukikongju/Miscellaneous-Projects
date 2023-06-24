from abc import ABC

class Player(ABC):

    def __init__(self, player_id, player_name):
        self.player_id = player_id
        self.player_name = player_name


class SimplePlayer(Player):

    """ Player only provide prefered teammates """

    def __init__(self, player_id: int, player_name: str, prefered_teamates: [int]):
        super(SimplePlayer, self).__init__(player_id, player_name)
        self.prefered_teamates = prefered_teamates

    def __str__(self):
        return f"Player: {self.player_id} => Prefered Teammates: {self.prefered_teamates}"


class PlayerPreferencesWillingnessScore(Player):

    """
    Player provide their teamate preferences, their willingness to play 
    offense and defense and have an offensive and defensive score
    """


    def __init__(self, player_id, prefered_teammates, offensive_willingness, 
                 defensive_willingness, offensive_score, defensive_score):
        super(PlayerPreferencesWillingnessScore, self).__init__(player_id, player_name)
        self.prefered_teammates = prefered_teammates
        self.offensive_willingness = offensive_willingness
        self.defensive_willingness = defensive_willingness
        self.offensive_score = offensive_score
        self.defensive_score = defensive_score

    def __str__(self):
        return f"Player: {self.player_id} => Prefered Teammates: {self.prefered_teammates}; Off/Def Willigness: {self.offensive_willingness}/{self.defensive_willingness}; Off/Def Score: {self.offensive_score}/{self.defensive_score}"


class CompletePlayer(Player):

    """ 
    Player class where we initialize player: 
    - teamate preferences: [str]
    - offense/defense willingness: [int]
    - offense/defense score: [int]
    - handling/cutter score: [int]
    """

    def __init__(self, player_id: int, player_name: str, 
                 teammate_preferences: [str], offensive_willingness: int, 
                 defensive_willingness: int, offensive_score: float, 
                 defensive_score: float, handling_score: float, cutting_score: float):
        super(CompletePlayer, self).__init__(player_id, player_name)
        self.player_id = player_id
        self.player_name = player_name
        self.teammate_preferences = teammate_preferences
        self.offensive_willingness = offensive_willingness
        self.defensive_willingness = defensive_willingness
        self.offensive_score = offensive_score
        self.defensive_score = defensive_score
        self.handling_score = handling_score
        self.cutting_score = cutting_score

    def __str__(self):
        return f"Player: {self.player_id} => Prefered Teammates: {self.prefered_teammates}; Off/Def Willigness: {self.offensive_willingness}/{self.defensive_willingness}; Off/Def Score: {self.offensive_score}/{self.defensive_score}; Handling/Cutting Score: {self.handling_score}/{self.cutting_score}"
