import time
import numpy as np
import constants 
from Resnet_funs import prediction 
import copy
from constants import (UPDATE_TIMES, EPSILON, KEY, 
        ALPHA, CPUCT,MAXVALUE_WEIGHT, MAX_SCORE)

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
        self.U = 0
    
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
                # Important!
                child.U =  ((1-epsilon) * child.P + epsilon * nu[idx] )  * \
                    (Nb) / (child.N)
                U = child.U
                Q = self.cpuct * child.Q 
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
            # temp = copy.deepcopy(currentNode.state) 
            temp = copy.copy(currentNode.state) 
            temp.action(KEY[i])
            child = Node(temp, currentNode, p[0][i], action = KEY[i])

            if child.state.matrix == currentNode.state.matrix:
                child.is_dead = True

            child.N = 1
            S_temp = (1-MAXVALUE_WEIGHT)*child.state.sum_value+\
                    MAXVALUE_WEIGHT*child.state.max_value
            child.S = S_temp / MAX_SCORE 
            child.Q = child.S # Revise: avoid the Q=0
            currentNode.add_child(child)
            self.add_to_tree(child)
 
               

    def back_fill(self, currentNode):
        while currentNode != None :
            
            sum_scorce = 0
            sum_N = 0
            for child in currentNode.childs:
                sum_N += child.N
                sum_scorce += child.S
            currentNode.N = sum_N
            currentNode.S = sum_scorce 
            # print("s: %.2f, n:%.2f "%(currentNode.S, currentNode.N))
            currentNode.Q = currentNode.S / currentNode.N
            currentNode = currentNode.father

    def add_to_tree(self, node):
        self.tree.append(node)

    def print_treeQU(self):
        father = self.root
        if not(father.is_leaf()):
            QU = []
            for child in father.childs:
                QU.append([(child.Q),(child.U), child.N])
                print([(child.Q),(child.U), child.N])
            print("-"*20)
            # print(QU)

            for child in father.childs:
                child_tree = MCTS(child)
                child_tree.print_treeQU()
        # else:
            # print(" "*10,"-"*10,"It's a leaf","-"*10," "*10,)

        


def mcts_process(gamegrid, model, tau=1):
    NN_data = {}
    event = None

    state = gamegrid
    root_mct = Node(state)
    mct = MCTS(root_mct, CPUCT)

    # print("*"*50)
    q = []
    u = []
    deltaQ = []
    deltaU = []
    N_prior = []
    for i in range(UPDATE_TIMES):
        print("-"*5,"step: %d"% i, "-"*5)
        is_update = mct.update_tree(model)
        q.append([CPUCT*mct.root.childs[0].Q,CPUCT*mct.root.childs[1].Q,CPUCT*mct.root.childs[2].Q,CPUCT*mct.root.childs[3].Q])
        u.append([mct.root.childs[0].U,mct.root.childs[1].U,mct.root.childs[2].U,mct.root.childs[3].U])
        N_prior.append([mct.root.N/mct.root.childs[0].N,mct.root.N/mct.root.childs[1].N,mct.root.N/mct.root.childs[2].N,mct.root.N/mct.root.childs[3].N])
        if i != 0 :
            # deltaQ.append(q[i]-q[i-1])
            deltaQ.append(np.array(q[i])-np.array(q[i-1]))
            deltaU.append(np.array(u[i])-np.array(u[i-1]))
            print("Q: ",q[i-1])
            print("deltaQ:",deltaQ[i-1])
            print("deltaU:",deltaU[i-1])
            print("U: ",u[i-1])
            print("N_prior: ", N_prior[i-1])

        if not(is_update):
            break
    # mct.print_treeQU()
    # print("delta Q all: ",q[-1]-q[0], "expand times :", len(q))
    # print("delta U all: ",u[-1]-u[0], "expand times :", len(u))
    # print("delta Q: mean-%f, max-%f(%d), min-%f(%d)."
            # %(np.mean(deltaQ),np.max(deltaQ),np.argmax(deltaQ),np.min(deltaQ),np.argmin(deltaQ)))
    # print("delta U: mean-%f, max-%f(%d), min-%f(%d)."
            # %(np.mean(deltaU),np.max(deltaU),np.argmax(deltaU),np.min(deltaU),np.argmin(deltaU)))

    feature = mct.root.state.matrix
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

    NN_data["feature"] = [feature] 
    NN_data["label"] = label
    # print(label["P"])

    event = np.argmax(label["P"])
    event = KEY[event]
    if event==None:
        raise Exception("The mcts isn't updated!")

    return NN_data, event
