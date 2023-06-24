from abc import ABC

class Player(ABC):

    def __init__(self, player_id):
        self.player_id = player_id


class SimplePlayer(Player):

    "Player only provide prefered teammates"

    def __init__(self, player_id, prefered_teamates):
        super(SimplePlayer, self).__init__(player_id)
        self.prefered_teamates = prefered_teamates

    def __str__(self):
        return f"Player: {self.player_id} => Prefered Teammates: {self.prefered_teamates}"


class PlayerPreferencesWillingnessScore(Player):

    def __init__(self, player_id, prefered_teammates, offensive_willingness, 
                 defensive_willingness, offensive_score, defensive_score):
        super(PlayerPreferencesWillingnessScore, self).__init__(player_id)
        self.prefered_teammates = prefered_teammates
        self.offensive_willingness = offensive_willingness
        self.defensive_willingness = defensive_willingness
        self.offensive_score = offensive_score
        self.defensive_score = defensive_score

    def __str__(self):
        return f"Player: {self.player_id} => Prefered Teammates: {self.prefered_teammates}; Off/Def Willigness: {self.offensive_willingness}/{self.defensive_willingness}; Off/Def Score: {self.offensive_score}/{self.defensive_score}"


