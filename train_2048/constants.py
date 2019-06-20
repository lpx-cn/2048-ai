# Parameters in main.py
TRAIN_TIMES = 10 # The game times for training the NN.
PLAY_TIMES =  0 # The game times for testing the NN.
POLICY_LOSS_WEIGHT = 1 # The weight of policy output for NN, which of score is 1. old =2048/7.3 
EPOCHs = 1 # Times for every training (the score as its weight)
NETWORK_INIT = True # Decide if a train model init is needed.
KEY= ["'w'", "'s'", "'a'", "'d'"] # The action 


# Parameters in MCTS.py
CPUCT = 8 # The Q's weight. Q+U=CPUCT*score/child.N+SUM(N)/N
UPDATE_TIMES = 4096 # Expand * times before deciding the action
MAX_SCORE = 65536# The target score used for normalization
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
