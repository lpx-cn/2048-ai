import puzzle
import MCTS
import time
from constants import KEY, PLAY_TIMES 

def playing(cpuct, times):
    gamegrid = puzzle.GameGrid()
    i=0
    while(1):
        event= MCTS.mcts_process(gamegrid.matrix, cpuct = cpuct, update_times = times)
        gamegrid.action(event)
        # print("step: ", i)
        i+=1
        for l in gamegrid.matrix:
            # print(l, event)
            pass
        if gamegrid.is_over:
            score_tem = gamegrid.max_value
            break
    return score_tem


if __name__ == '__main__':

    for cpuct in range(22,42,2):
        for times in [512,1024,2048]:
            score = [2]
            for game_times in range(PLAY_TIMES):
                # print("-"*20,game_times,"-"*20)
                score_tem = playing(cpuct, times)
                # print("the score is: ", score_tem)
                score.append(score_tem)

            print("-"*50)
            print("cpuct: %d,times: %d,2048: %d"%(cpuct,times,score.count(2048)))
            print(score)
            # print("Play %d times, the score can be %d" % (PLAY_TIMES,max(score)))
