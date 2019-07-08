# Parameters in main.py
PLAY_TIMES =  10 # The game times for testing the NN.
KEY= ["'w'", "'s'", "'a'", "'d'"] # The action 


# Parameters in MCTS.py
CPUCT = 11 # The Q's weight. Q+U=CPUCT*sum_score/child.N+SUM(N)/N
UPDATE_TIMES = 512 # Expand * times before deciding the action
MAX_SCORE = 65536 # The target score used for normalization
EPSILON = 0.1  # The pretection's unbelievable probability
KEY = ["'w'","'s'","'a'","'d'"]
ALPHA = 0.8 # dirichlet distribution parameters



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
