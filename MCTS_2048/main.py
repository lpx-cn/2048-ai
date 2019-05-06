import puzzle
import MCTS
import time
from constants import KEY, PLAY_TIMES 

def playing():
    gamegrid = puzzle.GameGrid()
    i=0
    while(1):
        event= MCTS.mcts_process(gamegrid)
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

    score = [2]
    for game_times in range(PLAY_TIMES):
        print("-"*20,game_times,"-"*20)
        score_tem = playing()
        print("the score is: ", score_tem)
        score.append(score_tem)

    print(score)
    print("Play %d times, the score can be %d" % (PLAY_TIMES,max(score)))
