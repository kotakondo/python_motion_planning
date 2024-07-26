class Params:

    def __init__(self):
        self.algorithms = ["bjps", "dynamic_a_star"] # benchmark algorithms
        self.num_runs = 1 # number of runs (if there's no randomness in the algorithm, num_runs = 1)
        self.bound_list = [i for i in range(5, 75, 5)] # bound list
        self.weight_list = [0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0] # weight list