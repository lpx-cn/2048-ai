import time
import numpy as np
import constants 
from Resnet_funs import prediction 
import copy
from constants import (UPDATE_TIMES, EPSILON, KEY, 
        ALPHA, CPUCT,MAXVALUE_WEIGHT, MAX_SCORE, CPUCT_denominator)

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
        q=0
        u=0
        N_prior = 0
        p = 0
        normal = []
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
                if currentNode == self.root:
                    normal.append(not(child.state.is_over) and not(child.is_dead))
                if not(child.state.is_over) and not(child.is_dead):
                    if is_first:
                        maxQU = Q+U
                        simulation_child= child
                        is_first = False
                    elif Q + U > maxQU :
                        maxQU = Q + U
                        simulation_child = child 
            currentNode = simulation_child
            if not self.root.is_leaf():
                p=np.array([self.root.childs[0].P,self.root.childs[1].P,self.root.childs[2].P,self.root.childs[3].P])
                q=np.array([self.cpuct*self.root.childs[0].Q,self.cpuct*self.root.childs[1].Q,self.cpuct*self.root.childs[2].Q,self.cpuct*self.root.childs[3].Q])
                u=np.array([self.root.childs[0].U,self.root.childs[1].U,self.root.childs[2].U,self.root.childs[3].U])
                N_prior=np.array([self.root.N/self.root.childs[0].N,self.root.N/self.root.childs[1].N,self.root.N/self.root.childs[2].N,self.root.N/self.root.childs[3].N])

            if currentNode == None:
                return False, q, u, N_prior, p,normal
        self.expand_leaf(currentNode,model)
        self.back_fill(currentNode)

        return True, q, u, N_prior,p,normal

    def expand_leaf(self, currentNode, model):
        p,s = prediction(currentNode.state.matrix, model)

        for i in range(4):
            # temp = copy.deepcopy(currentNode.state) 
            temp = copy.copy(currentNode.state) 
            temp.action(KEY[i])
            child = Node(temp, currentNode, p[0][i], action = KEY[i])
            child.N = 1
            S_temp = (1-MAXVALUE_WEIGHT)*child.state.sum_value+\
                    MAXVALUE_WEIGHT*child.state.max_value
            child.S = S_temp / MAX_SCORE 
            child.Q = child.S # Revise: avoid the Q=0
            currentNode.add_child(child)
            self.add_to_tree(child)
 
            # Important: over=lose
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
    p = []
    deltaQ = []
    deltaU = []
    N_prior = []
    for i in range(UPDATE_TIMES):
        print("-"*5,"step: %d"% i, "-"*5)
        mct.cpuct = mct.root.N/CPUCT_denominator
        is_update, q_,u_,n_,p_,normal_= mct.update_tree(model)
        q.append(q_)
        u.append(u_)
        N_prior.append(n_)
        p.append(p_)
        if i != 0 :
            deltaQ.append(np.array(q[i])-np.array(q[i-1]))
            deltaU.append(np.array(u[i])-np.array(u[i-1]))
            print("Q: ",q[i-1])
            print("deltaQ:",deltaQ[i-1])
            print("deltaU:",deltaU[i-1])
            print("U: ",u[i-1])
            print("N_prior: ", N_prior[i-1])
            print("p: ", p[i-1])
            print("normal:", normal_)
            print("cpuct: ",mct.cpuct)

        if not(is_update):
            break
    # mct.print_treeQU()

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
