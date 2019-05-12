import random
import numpy as np
import logic
import constants as c


class GameGrid():
    def __init__(self):

        self.init_matrix()
        self.is_over = False
        self.max_value = 2
        self.sum_value = 4
        self.is_win = False
        self.step = 0
        self.quadratic_sum=8


        self.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down,
                         c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right,
                         c.KEY_UP_ALT: logic.up, c.KEY_DOWN_ALT: logic.down,
                         c.KEY_LEFT_ALT: logic.left,
                         c.KEY_RIGHT_ALT: logic.right}


    def action(self, event):
        key = event
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            self.matrix, done = self.commands[key](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                done = False
                if logic.game_state(self.matrix) == 'win':
                    self.is_win = True 
                    self.is_over = True
                if logic.game_state(self.matrix) == 'lose':
                    self.is_win = False 
                    self.is_over = True
        self.step =self.step+1
        self.max_value = max(max(row) for row in self.matrix)
        self.sum_value = sum(sum(np.array(self.matrix)))
        # matrix's Frobenius value
        self.sum_value = np.square(np.linalg.norm(self.matrix)) 
        return self 

    def init_matrix(self):
        self.matrix = logic.new_game(4)
        self.history_matrixs = list()
        self.matrix = logic.add_two(self.matrix)
        self.matrix = logic.add_two(self.matrix)

