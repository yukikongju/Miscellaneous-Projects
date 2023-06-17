class SimplePlayer:


    def __init__(self, player_id, prefered_teamates):
        self.player_id = player_id
        self.prefered_teamates = prefered_teamates

    def __str__(self):
        return f"Player: {self.player_id}; Prefered Teamates: {self.prefered_teamates}"
