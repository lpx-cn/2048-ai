import puzzle
# import MCTS


def main():
    key_list = ["'w'", "'s'", "'a'", "'d'"]    
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
    
    # while(gamegrid.is_over == False):
        # keyboard.tap_key('W')

if __name__ == '__main__':
    main()

