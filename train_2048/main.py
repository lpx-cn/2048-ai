import puzzle
import MCTS
import Resnet_funs
from Resnet_funs import obtain_model, train_init, train_step
import time
from constants import (EPOCHs, KEY, POLICY_LOSS_WEIGHT, 
        TRAIN_TIMES, PLAY_TIMES, NETWORK_INIT) 


def data_merge(N,n):
    N["feature"] += n["feature"]
    N["label"]["P"] += n["label"]["P"]
    N["label"]["S"] += n["label"]["S"]

def play_one_time():
    gamegrid = puzzle.GameGrid()
    train_times = 0
    NN_data = {}

    model = obtain_model()

    i=0
    while(1):
        i+=1
        print("*"*25,"game_step: ", i, "*"*25)
        NN_data_temp, event= MCTS.mcts_process(gamegrid.matrix, model)
        if i == 1 :
            NN_data = NN_data_temp
        else:
            data_merge(NN_data, NN_data_temp)
        gamegrid.action(event)
        for l in gamegrid.matrix:
            print(l, event)
        if gamegrid.is_over:
            score_tem = gamegrid.max_value
            break
    return NN_data, score_tem 

def training(training_times = TRAIN_TIMES):
    score = []
    if NETWORK_INIT: 
        train_init(POLICY_LOSS_WEIGHT)

    for game_times in range(training_times):

        NN_data, score_tem = play_one_time()

        score.append(score_tem)
        if score_tem >= max(score):
            train_times = int(score_tem**2/max(score))
        else:
            train_times = 0 
        train_times = int(train_times * EPOCHs)

        train_step(NN_data, train_times)
        print("the %dth game score is: %d" %(game_times, score_tem))
    print(score)
    print("The score can be %d" % max(score))

def playing(model):
    gamegrid = puzzle.GameGrid()
    i=0
    while(1):
        NN_data_temp, event= MCTS.mcts_process(gamegrid.matrix, model)
        gamegrid.action(event)
        print("step: ", i)
        i+=1
        for l in gamegrid.matrix:
            print(l, event)
        if gamegrid.is_over:
            score_tem = gamegrid.max_value
            break
    return score_tem


if __name__ == '__main__':
    training()

    score = [2]
    model = obtain_model()
    for game_times in range(PLAY_TIMES):
        print("-"*20,game_times,"th play""-"*20)
        score_tem = playing(model)
        print("the score is: ", score_tem)
        score.append(score_tem)

    print(score)
    print("Play %d times, the score can be %d" % (PLAY_TIMES,max(score)))
