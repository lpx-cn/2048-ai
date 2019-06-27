import puzzle
import numpy as np
import MCTS
import Resnet_funs
from Resnet_funs import obtain_model, train_init, train_step
import time
from constants import (EPOCHs, KEY, POLICY_LOSS_WEIGHT, 
        TRAIN_TIMES, PLAY_TIMES, NETWORK_INIT, DATA_SIZE) 


def data_merge(N,n):
    N["feature"] += n["feature"]
    N["label"]["P"] += n["label"]["P"]
    N["label"]["S"] += n["label"]["S"]
    
def data_list_merge(data_list):
    new_data = data_list[0]
    for i in range(1,len(data_list)):
        data_merge(new_data, data_list[i])
    return new_data


def data_moniter(data_list, score_list, best_score):
    print("-"*50)

    for i in range(len(score_list)):
        for j in range(i,len(score_list)):
            if score_list[i] < score_list[j]:
                score_list[i],score_list[j] = score_list[j],score_list[i]
                data_list[i],data_list[j] = data_list[j],data_list[i]

    size_list = []
    count = 0
    new_data_list = []
    new_score = []

    i=0
    for data in data_list:
        new_score.append(score_list[i])
        i += 1
        new_data_list.append(data)
        size_list.append(len(data["label"]["S"]))
        count += len(data["label"]["S"])
        print(size_list, count)
        if count >= DATA_SIZE:
            # weights = np.array(size_list)/(count)
            # print(weights)
            # print(new_score)
            # mean_score = np.array(weights) * np.array(new_score)
            # print(mean_score)
            mean_score = sum(np.array(new_score)/count)
            print(mean_score)

            if mean_score > best_score:
                return True, new_data_list, mean_score 
            break
    return False, None, None

def play_one_time():
    gamegrid = puzzle.GameGrid()
    train_times = 0
    NN_data = {}

    model = obtain_model()

    i=0
    while(1):
        i+=1
        # print("*"*25,"game_step: ", i, "*"*25)
        NN_data_temp, event= MCTS.mcts_process(gamegrid.matrix, model)
        if i == 1 :
            NN_data = NN_data_temp
        else:
            data_merge(NN_data, NN_data_temp)
        gamegrid.action(event)
        for l in gamegrid.matrix:
            # print(l, event)
            pass
        if gamegrid.is_over:
            score_tem = gamegrid.max_value
            square_score = gamegrid.sum_square
            break
    return NN_data, score_tem, square_score

def training(training_times = TRAIN_TIMES):
    mean_score = [64]
    if NETWORK_INIT: 
        train_init(POLICY_LOSS_WEIGHT)

    for game_times in range(training_times):

        NN_data_list=[]
        score_list=[]
        real_score = []

        while(1):
            NN_data_temp, _, score_tem = play_one_time()
            real_score.append(_)
            NN_data_list.append(NN_data_temp)
            score_list.append(score_tem)
            IsEnough, new_data_list, mean_score_temp= data_moniter(NN_data_list, score_list, mean_score[-1])
            if IsEnough:
                mean_score.append(mean_score_temp)
                new_data = data_list_merge(new_data_list)
                break

        train_step(new_data, EPOCHs)
        print("the %dth game mean_score is: %d" %(game_times, mean_score_temp))
        print("the %dth game real_score is: " %(game_times), real_score)
    print(mean_score)
    print("The mean_score can be %d" % max(mean_score))

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
