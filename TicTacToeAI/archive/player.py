#!/usr/bin/env/python

import math
import random
import time


class Player():

    def __init__(self, letter: str):
        self.letter = letter

    def get_move(self, game: list) -> int:
        pass


class HumanPlayer(Player):

    """ Movement is controlled by human """

    def __init__(self, letter: str):
        super().__init__(letter)

    def get_move(self, game: list) -> int:
        """ TODO: player has to select valid move """
        is_valid_move = False
        while not is_valid_move:
            move = input(f"{self.letter}'s turn. Select your move (0-8)'")
            try:
                value = int(move)
                if value not in game.available_moves():  # if cell not empty
                    raise ValueError
                is_valid_move = True
            except ValueError:
                print("Invalid move. Please try again.")
        return value


class DumbComputerPlayer(Player):

    def __init__(self, letter: str):
        super().__init__(letter)

    def get_move(self, game: list) -> int:
        """ Computer choose a random move """
        return random.choice(game.available_moves())


class SmartComputerPlayer(object):

    def __init__(self, letter: str):
        super().__init__(letter)

    def get_move(self, game: list) -> int:
        """ TODO: implement AI to choose move """
        pass
