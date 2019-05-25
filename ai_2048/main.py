from show_puzzle import GameGrid
import random
import time
from show_constants import Mode, key_list



def random_player():
    value= []
    for i in range(1000):
        gamegrid = GameGrid()
        while(gamegrid.is_over == False):
            # time.sleep(1)
            k = random.randint(0,3)
            gamegrid.action(key_list[k])
        print("%dth step's max value is %d" %(i, gamegrid.max_value))
        value.append(gamegrid.max_value)

        gamegrid.windows.destroy()
    print("*"*50)
    print("the max value is ", max(value))
    # time.sleep(3600)

def AI_player():
    pass

def MTCS_player():
    import MCTS 

    value= []
    for i in range(2):
        gamegrid = GameGrid()
        while(gamegrid.is_over == False):
            # time.sleep(1)
            event = MCTS.mcts_process(gamegrid)
            gamegrid.action(event)
            gamegrid.update_grid_cells()
            print(gamegrid.matrix)
        print("%dth step's max value is %d" %(i, gamegrid.max_value))
        value.append(gamegrid.max_value)

        gamegrid.windows.destroy()
    print("*"*50)
    print("the max value is ", max(value))

if __name__ == '__main__':
    Mode = 'MTCS'
    if Mode == 'random':
        random_player()
    elif Mode == 'MTCS':
        MTCS_player()
    elif Mode == 'AI':
        AI_player()
    else:
        print('Mode does not exist!')

