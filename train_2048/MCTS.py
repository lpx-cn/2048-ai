import time
import numpy as np
import constants 
from Resnet_funs import prediction 
import copy
from constants import UPDATE_TIMES, EPSILON, KEY, ALPHA, CPUCT

class Node():

    def __init__(self, state, father = None, prior = 1, action = None):
        self.state = state
        self.childs= []
        self.father = father 
        self.action= action 
        self.is_dead = False
        self.N = 0
        self.S = 0
        self.Q = 0
        self.P = prior
    
    def is_leaf(self):
        if len(self.childs) > 0:
            return False
        else:
            return True

    def add_child(self, node):
        self.childs.append(node)


class MCTS():

    def __init__(self, root, cpuct):
        self.root = root
        self.tree = [] 
        self.cpuct = cpuct
        self.add_to_tree(root)
    
    def __len__(self):
        return len(self.tree)

    def update_tree(self, model):
        currentNode = self.root
        value = 0
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
                U = self.cpuct * \
                    ((1-epsilon) * child.P + epsilon * nu[idx] )  * \
                    np.sqrt(Nb) / (4 + child.N)
                Q = child.Q 
                # print("Q: %.2f, U: %.2f" %(Q,U))
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
        self.expand_leaf(currentNode,model)
        self.back_fill(currentNode)
        return True

    def expand_leaf(self, currentNode, model):
        p,s = prediction(currentNode.state.matrix, model)

        for i in range(4):
            temp = copy.deepcopy(currentNode) 
            temp.state.action(KEY[i])
            child = Node(temp.state, currentNode, p[0][i], action = KEY[i])

            if child.state.matrix == currentNode.state.matrix:
                child.is_dead = True

            child.S = child.state.max_value
            currentNode.add_child(child)
            self.add_to_tree(child)
        

    def back_fill(self, currentNode):
        while currentNode != None :
            currentNode.N += 4
            
            sum_scorce = 0
            for child in currentNode.childs:
                sum_scorce += child.S
            currentNode.S = sum_scorce 
            currentNode.Q = currentNode.S / currentNode.N
            currentNode = currentNode.father

    def add_to_tree(self, node):
        self.tree.append(node)




def mcts_process(gamegrid, model, tau=1):
    NN_data = {}
    event = None

    state = gamegrid
    root_mct = Node(state)
    mct = MCTS(root_mct, CPUCT)


    for i in range(UPDATE_TIMES):
        # print("*"*25,"step: %d"% i, "*"*25)
        is_update = mct.update_tree(model)

        if not(is_update):
            break

    feature = mct.root.state.matrix
    label = {}
    label["P"] = []
    label["S"] = [mct.root.Q]

    for child in mct.root.childs:
        p = np.power(child.N, 1/tau) 
        label["P"].append(p)

    # If the gamegrid come to last step, all childs' N is 0.
    # the status is "is_over" or "is_dead".
    if sum(label["P"])==0:
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
        label["P"]=[label["P"]/sum(label["P"])]

    NN_data["feature"] = [feature] 
    NN_data["label"] = label

    event = np.argmax(label["P"])
    event = KEY[event]
    if event==None:
        raise Exception("The mcts isn't updated!")

    return NN_data, event

