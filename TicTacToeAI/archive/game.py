#!/usr/bin/env/python

import math
import random
import time

from player import HumanPlayer, DumbComputerPlayer, SmartComputerPlayer


class TicTacToe():

    def __init__(self):
        self.board = self.create_empty_board()
        self.winner = None

    @staticmethod
    def create_empty_board() -> list:
        return [' ' for _ in range(9)]

    def show_board(self) -> None:
        for row in [self.board[i*3:(i+1) * 3] for i in range(3)]:
            print('|' + '|'.join(row) + '|')

    @staticmethod
    def show_board_nums() -> None:
        numbers = [str(i) for i in range(9)]
        for row in [numbers[i*3:(i+1) * 3] for i in range(3)]:
            print('|' + '|'.join(row) + '|')

    def available_moves(self) -> list:
        return [i for i, spot in enumerate(self.board) if spot == " "]

    def has_empty_squares(self) -> bool:
        return ' ' in self.board

    def num_empty_square(self) -> int:
        return self.board.count(' ')

    def make_move(self, move: int, letter: str) -> None:
        """ Put player make move on the board """
        if self.board[move] == ' ':
            self.board[move] = letter

    def is_winner(self, move: int, letter: str) -> bool:
        """ TODO: check if player's move was a winner """
        # check horizontal
        row_index = math.floor(move/3)
        row = self.board[row_index*3:(row_index+1)*3]
        if all([s == letter for s in row]):
            return True

        # check vertical
        col_index = move % 3
        column = [self.board[col_index+i*3] for i in range(3)]
        if all([s == letter for s in column]):
            return True

        if move % 2 == 0:
            # check diagonal \
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([s == letter for s in diagonal1]):
                return True

            # check diagonal /
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([s == letter for s in diagonal2]):
                return True

        return False


def play(game, x_player, o_player, print_game=True) -> None:
    print("Controls:\n")
    if print_game:
        game.show_board_nums()

    letter = 'X'  # Upgrade: choose starting player at random
    while game.has_empty_squares() and game.winner is None:
        # player choose a valid move
        if letter == 'X':
            move = x_player.get_move(game)
        else:
            move = o_player.get_move(game)

        # play the move on board
        game.make_move(move, letter)
        if print_game:
            print(f"\n{letter} move to square {move}\n")
            game.show_board()
            print('')

        # check if game is finished: game winner or no more empty cells
        is_winner = game.is_winner(move, letter)
        if is_winner:
            print(f"{letter} wins!")  # to fix: print wrong winner
            game.winner = letter
            #  if input("Press r to play again") is not 'r':
            #  play = False
        if not game.has_empty_squares() and not is_winner:
            print(f"It's a tie!")

        # switch turns
        letter = 'O' if letter == 'X' else 'X'

    time.sleep(1)


def main():
    game = TicTacToe()
    x_player = DumbComputerPlayer('X')
    o_player = HumanPlayer('O')
    play(game, x_player, o_player, print_game=True)


if __name__ == "__main__":
    main()
