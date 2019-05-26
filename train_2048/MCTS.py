import time
import numpy as np
import constants 
from Resnet_funs import prediction 
import copy
from constants import (UPDATE_TIMES, EPSILON, KEY, 
        ALPHA, CPUCT,MAXVALUE_WEIGHT, MAX_SCORE, CPUCT_denominator)
import constants as c
import logic


commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down,
                c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right,
                c.KEY_UP_ALT: logic.up, c.KEY_DOWN_ALT: logic.down,
                c.KEY_LEFT_ALT: logic.left,
                c.KEY_RIGHT_ALT: logic.right}

class Node():


    def __init__(self, matrix, father = None, prior = 1, action = None):
        self.matrix = matrix 
        self.is_over = False
        if logic.game_state(self.matrix) == 'win':
            self.is_win = True 
            self.is_over = True
        if logic.game_state(self.matrix) == 'lose':
            self.is_win = False 
            self.is_over = True
        self.max_value = max(max(row) for row in self.matrix)
        self.sum_value = sum(sum(np.array(self.matrix)))

        self.childs= []
        self.father = father 
        self.action= action 
        self.is_dead = False
        self.N = 0
        self.S = 0
        self.Q = 0
        self.U = 0
        self.P = prior
    
    def is_leaf(self):
        if len(self.childs) > 0:
            return False
        else:
            return True

    def add_child(self, node):
        self.childs.append(node)

    def act(self, event):
        key = event
        if key in commands:
            new_matrix, done = commands[key](self.matrix)
            if done:
                new_matrix = logic.add_two(new_matrix)
                # record last move
                done = False
        return new_matrix 

class MCTS():

    def __init__(self, root, cpuct = CPUCT):
        self.root = root
        self.tree = [] 
        self.cpuct = cpuct
        self.add_to_tree(root)
    
    def __len__(self):
        return len(self.tree)

    def update_tree(self, model):
        currentNode = self.root
        while not currentNode.is_leaf():
            if currentNode == self.root:
                epsilon = EPSILON
                nu = np.random.dirichlet([ALPHA] * len(currentNode.childs))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.childs)
            Nb = currentNode.N
            simulation_child = None

            is_first = True
            for idx, child in enumerate(currentNode.childs):
                # Important!
                child.U =  ((1-epsilon) * child.P + epsilon * nu[idx] )  * \
                    (Nb) / (child.N)
                U = child.U
                Q = self.cpuct * child.Q 
                if not(child.is_over) and not(child.is_dead):
                    if is_first:
                        maxQU = Q+U
                        simulation_child= child
                        is_first = False
                    elif Q + U > maxQU :
                        maxQU = Q + U
                        simulation_child = child 
            currentNode = simulation_child

            if currentNode == None:
                return False
        self.expand_leaf(currentNode,model)
        self.back_fill(currentNode)

        return True

    def expand_leaf(self, currentNode, model):
        p,s = prediction(currentNode.matrix, model)

        for i in range(4):
            temp = currentNode.act(KEY[i])
            child = Node(temp, currentNode, p[0][i], action = KEY[i])
            child.N = 1
            S_temp = (1-MAXVALUE_WEIGHT)*child.sum_value+\
                    MAXVALUE_WEIGHT*child.max_value
            child.S = S_temp / MAX_SCORE 
            child.Q = child.S # Revise: avoid the Q=0
            currentNode.add_child(child)
            self.add_to_tree(child)
 
            # Important: over==lose
            if (child.matrix == currentNode.matrix)or(child.is_over):
                child.is_dead = True
                child.S = -child.S
               

    def back_fill(self, currentNode):
        while currentNode != None :
            
            sum_scorce = 0
            sum_N = 0
            for child in currentNode.childs:
                sum_N += child.N
                sum_scorce += child.S
            currentNode.N = sum_N
            currentNode.S = sum_scorce 
            currentNode.Q = currentNode.S / currentNode.N
            currentNode = currentNode.father

    def add_to_tree(self, node):
        self.tree.append(node)

def mcts_process(matrix, model, tau=1):
    NN_data = {}
    event = None

    root_mct = Node(matrix)
    mct = MCTS(root_mct, CPUCT)

    for i in range(UPDATE_TIMES):
        mct.cpuct = mct.root.N/CPUCT_denominator
        is_update= mct.update_tree(model)
        if not(is_update):
            break

    feature = mct.root.matrix
    label = {}
    label["P"] = []
    label["S"] = [mct.root.Q]



    # If the gamegrid come to last step, all childs' N is 1.
    # the status is "is_over" or "is_dead".
    is_last = True
    for child in mct.root.childs:
        if child.N!=1:
            is_last = False

    if is_last:
        label["P"] = []
        for child in mct.root.childs:
            if child.is_over:
                p = np.power(child.S, 1/tau) 
            elif child.is_dead:
                p = 0
            else:
                raise Exception("It's not last step, logic is wrong!")
            label["P"].append(p)
        label["P"]=[label["P"]/sum(label["P"])]
    else:
        for child in mct.root.childs:
            p = np.power(child.N, 1/tau) 
            label["P"].append(p)
        label["P"]=[label["P"]/sum(label["P"])]

    NN_data["feature"] = [feature] 
    NN_data["label"] = label

    event = np.argmax(label["P"])
    event = KEY[event]
    if event==None:
        raise Exception("The mcts isn't updated!")

    return NN_data, event
