import time
import numpy as np
import constants 
import copy
from constants import (UPDATE_TIMES, EPSILON, KEY, 
        ALPHA, CPUCT,MAXVALUE_WEIGHT, MAX_SCORE, CPUCT_denominator)

class Node():

    def __init__(self, state, father = None, action = None):
        self.state = state
        self.childs= []
        self.father = father 
        self.action= action 
        self.is_dead = False
        self.N = 0
        self.S = 0
        self.Q = 0
        self.U = 0
        self.P = 1/4
    
    def is_leaf(self):
        if len(self.childs) > 0:
            return False
        else:
            return True

    def add_child(self, node):
        self.childs.append(node)


class MCTS():

    def __init__(self, root, cpuct = CPUCT):
        self.root = root
        self.tree = [] 
        self.cpuct = cpuct
        self.add_to_tree(root)
    
    def __len__(self):
        return len(self.tree)

    def update_tree(self):
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
                child.U = ((1-epsilon)*1/4  + epsilon * nu[idx] )  * \
                    (Nb) / (child.N)
                U = child.U
                Q = self.cpuct * child.Q 
                if not(child.state.is_over) and not(child.is_dead):
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
        self.expand_leaf(currentNode)
        self.back_fill(currentNode)
        return True

    def expand_leaf(self, currentNode):

        for i in range(4):
            temp = copy.copy(currentNode.state) 
            temp.action(KEY[i])
            child = Node(temp, currentNode, action = KEY[i])
            child.N = 1
            S_temp = (1-MAXVALUE_WEIGHT)*child.state.sum_value+\
                    MAXVALUE_WEIGHT*child.state.max_value
            child.S = S_temp / MAX_SCORE 
            child.Q = child.S # Revise: avoid the Q=0
            currentNode.add_child(child)
            self.add_to_tree(child)
            if (child.state.matrix == currentNode.state.matrix)or(child.state.is_over):
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

def mcts_process(gamegrid, tau = 1):
    event = None

    state = gamegrid
    root_mct = Node(state)
    mct = MCTS(root_mct, CPUCT)

    q = []
    u = []
    p = []
    deltaQ = []
    deltaU = []
    N_prior = []

    for i in range(UPDATE_TIMES):
        mct.cpuct = mct.root.N/CPUCT_denominator
        is_update = mct.update_tree()
    

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
            if child.state.is_over:
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

    event = np.argmax(label["P"])
    event = KEY[event]
    if event==None:
        raise Exception("The mcts isn't updated!")

    return event
