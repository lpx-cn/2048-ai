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

def training(train_tims = TRAIN_TIMES):
    score = []
    if NETWORK_INIT: 
        train_init(POLICY_LOSS_WEIGHT)

    for game_times in range(train_tims):
        gamegrid = puzzle.GameGrid()
        train_times = 0
        NN_data = {}

        model = obtain_model()

        i=0
        while(1):
            NN_data_temp, event= MCTS.mcts_process(gamegrid, model)
            if i == 0 :
                NN_data = NN_data_temp
            else:
                data_merge(NN_data, NN_data_temp)
            gamegrid.action(event)
            print("step: ", i)
            i+=1
            for l in gamegrid.matrix:
                print(l, event)
            if gamegrid.is_over:
                score_tem = gamegrid.max_value
                score.append(score_tem)
                if score_tem >= max(score):
                    train_times = int(score_tem/max(score))
                train_times *= int(EPOCHs) 
                break
        # if is_stable(score):
            # print("The score can be %d" % max(score))
            # break
        train_step(NN_data, train_times)
        print("the score is: ", score_tem)
    print(score)
    print("The score can be %d" % max(score))

def playing(model):
    score = []

    for game_times in range(PLAY_TIMES):
        print("-"*20,game_times,"-"*20)
        gamegrid = puzzle.GameGrid()
        NN_data = {}

        i=0
        while(1):
            NN_data_temp, event= MCTS.mcts_process(gamegrid, model)
            gamegrid.action(event)
            print("step: ", i)
            i+=1
            for l in gamegrid.matrix:
                print(l, event)
            if gamegrid.is_over:
                score_tem = gamegrid.max_value
                score.append(score_tem)
                break
        print("the score is: ", score_tem)
    print(score)
    print("The score can be %d" % max(score))


if __name__ == '__main__':
    training()
