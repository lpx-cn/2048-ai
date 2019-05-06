# Parameters in main.py
PLAY_TIMES =  100 # The game times for testing the NN.
KEY= ["'w'", "'s'", "'a'", "'d'"] # The action 


# Parameters in MCTS.py
CPUCT = 20 # The upper confidence bound's weight. Q+U=score/N+CPUCT*SUM(N)/N
UPDATE_TIMES = 32 # Expand * times before deciding the action
MAXVALUE_WEIGHT = 0.8 # The weight of max_value in score (max_value + sum_value)
EPSILON = 0.2  # The pretection's unbelievable probability
KEY = ["'w'","'s'","'a'","'d'"]
ALPHA = 0.8 # dirichlet distribution parameters

# Parameters in Resnet_funs.py
MODEL_PATH = "./Model_Resnet/" # The path to store the old/new Resnet model. And model.png, tensorboard, log(for loss)


# Parameters in logic.py
KEY_UP_ALT = "\'\\uf700\'"
KEY_DOWN_ALT = "\'\\uf701\'"
KEY_LEFT_ALT = "\'\\uf702\'"
KEY_RIGHT_ALT = "\'\\uf703\'"

KEY_UP = "'w'"
KEY_DOWN = "'s'"
KEY_LEFT = "'a'"
KEY_RIGHT = "'d'"
KEY_BACK = "'b'"
