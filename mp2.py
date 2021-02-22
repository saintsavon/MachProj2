#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NAME: Jaylen Scott
# DATE: February 21st, 2021
# COURSE: Artificial Intelligence
# SEMESTER: Spring 2021
# MP2: Alpha/Beta Search Program

import numpy as np
import random
import math


class GenGameBoard:
    """
    Class responsible for representing the game board and game playing methods
    """
    num_pruned = 0  # counts number of pruned branches due to alpha/beta
    MAX_DEPTH = 20  # max depth before applying evaluation function
    depth = 0  # current depth within minimax search

    UP = 'w'
    DOWN = 's'
    LEFT = 'a'
    RIGHT = 'd'
    UP_BUILD = 'wb'
    DOWN_BUILD = 'sb'
    LEFT_BUILD = 'ab'
    RIGHT_BUILD = 'db'

    def __init__(self, board_size=4):
        """
        Constructor method - initializes each position variable and the board
        """
        self.board_size = board_size  # Holds the size of the board
        self.marks = np.empty((board_size, board_size), dtype='str')  # Holds the mark for each position
        self.marks[:, :] = ' '
        self.has_gold = False
        self.monster_pos = (0, 0)
        self.player_pos = (3, 0)
        self.gold_pos = (1, 2)
        self.exit_pos = (3, 0)
        self.max_moves = self.board_size * 2 + 1
        self.num_moves = 0
        self.depth_reached = 0

    def print_board(self):
        """
        Prints the game board using current marks
        """
        # Print column numbers
        print(' ', end='')
        for j in range(self.board_size):
            print(" " + str(j + 1), end='')

            # Print rows with marks
        print("")
        for i in range(self.board_size):
            # Print line separating the row
            print(" ", end='')
            for j in range(self.board_size):
                print("--", end='')

            print("-")

            # Print row number
            print(i + 1, end='')

            # Print marks on self row
            for j in range(self.board_size):
                if (i, j) == self.monster_pos:
                    print("|W", end='')
                elif (i, j) == self.gold_pos and not self.has_gold:
                    print("|G", end='')
                elif (i, j) == self.player_pos:
                    print("|P", end='')
                else:
                    print("|" + self.marks[i][j], end='')

            print("|")

        # Print line separating the last row
        print(" ", end='')
        for j in range(self.board_size):
            print("--", end='')

        print("-")
        print("Number pruned due to a/b: {}")

    def make_move(self, action, player_move):
        """
        Makes the move for either player or monster
        """
        assert action in self.get_actions(player_move)

        # Make the move
        if player_move:
            if action == self.UP:
                self.player_pos = (self.player_pos[0] - 1, self.player_pos[1])
            elif action == self.DOWN:
                self.player_pos = (self.player_pos[0] + 1, self.player_pos[1])
            elif action == self.LEFT:
                self.player_pos = (self.player_pos[0], self.player_pos[1] - 1)
            elif action == self.RIGHT:
                self.player_pos = (self.player_pos[0], self.player_pos[1] + 1)
            elif action == self.UP_BUILD:
                self.marks[self.player_pos[0] - 1, self.player_pos[1]] = '#'
            elif action == self.DOWN_BUILD:
                self.marks[self.player_pos[0] + 1, self.player_pos[1]] = '#'
            elif action == self.LEFT_BUILD:
                self.marks[self.player_pos[0], self.player_pos[1] - 1] = '#'
            elif action == self.RIGHT_BUILD:
                self.marks[self.player_pos[0], self.player_pos[1] + 1] = '#'
            self.num_moves = self.num_moves + 1
        else:
            if action == self.UP:
                self.monster_pos = (self.monster_pos[0] - 1, self.monster_pos[1])
            elif action == self.DOWN:
                self.monster_pos = (self.monster_pos[0] + 1, self.monster_pos[1])
            elif action == self.LEFT:
                self.monster_pos = (self.monster_pos[0], self.monster_pos[1] - 1)
            elif action == self.RIGHT:
                self.monster_pos = (self.monster_pos[0], self.monster_pos[1] + 1)

    def game_won(self, player_move):
        """
        Determines whether a game winning condition exists for the player or monster
        """
        if player_move:
            if self.has_gold and self.player_pos == self.exit_pos:
                return True
            else:
                return False
        else:
            if self.num_moves == self.max_moves or self.monster_pos == self.player_pos:
                return True
            else:
                return False

    def get_actions(self, player_move):
        '''Generates a list of possible moves'''
        moves = []

        if player_move:
            if self.player_pos[0] > 0 and self.marks[self.player_pos[0] - 1, self.player_pos[1]] == ' ':
                moves.append(self.UP)
                moves.append(self.UP_BUILD)
            if self.player_pos[0] < self.marks.shape[0] - 1 and self.marks[
                self.player_pos[0] + 1, self.player_pos[1]] == ' ':
                moves.append(self.DOWN)
                moves.append(self.DOWN_BUILD)
            if self.player_pos[1] > 0 and self.marks[self.player_pos[0], self.player_pos[1] - 1] == ' ':
                moves.append(self.LEFT)
                moves.append(self.LEFT_BUILD)
            if self.player_pos[1] < self.marks.shape[1] - 1 and self.marks[
                self.player_pos[0], self.player_pos[1] + 1] == ' ':
                moves.append(self.RIGHT)
                moves.append(self.RIGHT_BUILD)
        else:
            if self.monster_pos[0] > 0 and self.marks[self.monster_pos[0] - 1, self.monster_pos[1]] == ' ':
                moves.append(self.UP)
            if self.monster_pos[0] < self.marks.shape[0] - 1 and self.marks[
                self.monster_pos[0] + 1, self.monster_pos[1]] == ' ':
                moves.append(self.DOWN)
            if self.monster_pos[1] > 0 and self.marks[self.monster_pos[0], self.monster_pos[1] - 1] == ' ':
                moves.append(self.LEFT)
            if self.monster_pos[1] < self.marks.shape[1] - 1 and self.marks[
                self.monster_pos[0], self.monster_pos[1] + 1] == ' ':
                moves.append(self.RIGHT)
            moves.append('')  # stay move

        return moves

    def no_more_moves(self, player_move):
        """
        Determines whether there are any moves left for player or monster
        """
        return len(self.get_actions(player_move)) == 0

    # TODO - self method should run minimax to determine the value of each move
    # Then make best move for the computer by placing the mark in the best spot
    def make_comp_move(self):
        # This code chooses a random computer move
        possible_moves = self.get_actions(False)
        rand_move_index = random.randrange(len(possible_moves))
        self.make_move(possible_moves[rand_move_index], False)

        # Make AI move
        best_action = self.alpha_beta_search()
        self.make_move(best_action, False)

        best_action = self.minimax_search()
        self.make_move(best_action[0] + 1, best_action[1] + 1, 'M')

    def alpha_beta_search(self):
        v, best_action = self.max_value()
        return best_action

    def minimax_search(self):
        v, best_action = self.max_value()
        return best_action

    def max_value(self):
        if self.is_terminal():
            return self.get_utility(), None
        v = -math.inf
        beta = math.inf
        alpha = -math.inf
        for action in self.get_actions():
            self.marks[action[0]][action[1]]: 'M'
            min_val, _ = self.min_value()
            self.marks[action[0]][action[1]]: ' '
            if min_val > v:
                v = min_val
                best_action = action
            if v == 1:
                return v, best_action
            if v >= beta:
                max(alpha, v)
                return v, best_action

            return v, best_action

    def min_value(self):
        if self.is_terminal():
            return self.get_utility(), None
        v = math.inf
        beta = math.inf
        alpha = -math.inf
        for action in self.get_actions():
            self.marks[action[0]][action[1]]: 'P'
            max_val, _ = self.max_value()
            self.marks[action[0]][action[1]]: ' '
            if max_val < v:
                v = max_val
                best_action = action
            if v == -1:
                return v, best_action
            if v <= alpha:
                min(beta, v)
                return v, best_action
            return v, best_action

    def get_utility(self):
        if self.has_gold and self.player_pos == self.exit_pos:
            return -1
        elif self.num_moves == self.max_moves or self.monster_pos == self.player_pos:
            return 1
        else:
            return 0

    def is_terminal(self, player_move):
        """
        Determines if the current board state is a terminal state
        """
        if self.no_more_moves(player_move) or self.game_won(player_move):
            return True
        else:
            return False

###########################
### Program starts here ###
###########################        

# Print out the header info
print("CLASS: Artificial Intelligence, Lewis University")
print("NAME: Jaylen Scott")

# Define constants
LOST = 0
WON = 1

# Create the game board of the given size and print it
board = GenGameBoard(4)
board.print_board()

# Start the game loop
while True:
    # *** Player's move ***        

    # Try to make the move and check if it was possible  
    print("Player's Move #", (board.num_moves + 1))
    possible_moves = board.get_actions(True)
    move = input("Choose your move " + str(possible_moves) + ": ")
    while move not in possible_moves:
        print("Not a valid move")
        move = input("Choose your move " + str(possible_moves) + ": ")
    board.make_move(move, True)

    # Check for gold co-location    
    if not board.has_gold and board.player_pos == board.gold_pos:
        board.has_gold = True

    # Display the board
    board.print_board()

    # Check for ending condition
    # If game is over, check if player won and end the game
    if board.game_won(True):
        # Player won
        result = WON
        break
    elif board.no_more_moves(True):
        # No moves left -> lost
        result = LOST
        break

    # *** Computer's move ***
    board.make_comp_move()

    # Print out the board again
    board.print_board()


    # Check for ending condition
    # If game is over, check if computer won and end the game
    if board.game_won(False):
        # Computer won
        result = LOST
        break

# Check the game result and print out the appropriate message
print("GAME OVER")
if result == WON:
    print("You Won!")
else:
    print("You Lost!")
