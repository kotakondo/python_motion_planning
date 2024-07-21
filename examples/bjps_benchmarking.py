"""
@file: global_example.py
@breif: global planner application examples
@author: Winter
@update: 2023.3.2
"""

from python_motion_planning.utils import Grid, Map, SearchFactory
from argparse import ArgumentParser
import itertools

def main():
    
    # path searcher constructor
    search_factory = SearchFactory()

    # build environment
    start = (5, 5)
    goal = (45, 5)
    env = Grid(51, 31)

    # benchmark parameters
    algorithms = ["bjps", "dynamic_a_star"] # benchmark algorithms
    num_runs = 1 # number of runs (if there's no randomness in the algorithm, num_runs = 1)
    bound_list = [5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0] # jump bound (= interval length) list
    weight_list = [0.0001, 0.01, 0.1, 0.05, 0.02, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0] # weight list
    
    # create combinations for algorithms, bounds, and weights
    combinations = list(itertools.product(algorithms, bound_list, weight_list))

    # run benchmarking
    for search_algorithm, bound, weight in combinations:

        print("Algorithm: ", search_algorithm)
        print("Bound: ", bound)
        print("Weight: ", weight)

        # creat planner
        planner = search_factory(search_algorithm, start=start, goal=goal, env=env)

        # benchmarking
        cost, time, success_rate = planner.benchmarking_run(num_runs, bound, weight)

        print("Cost: ", cost)
        print("Time: ", time)
        print("Success Rate: ", success_rate)

        # save it to a file
        with open("benchmarking_results.txt", "a") as f:
            f.write(f"Algorithm: {search_algorithm}, Bound: {bound}, Weight: {weight}, Cost: {round(cost,2)}, Time: {round(time,2)}, Success Rate: {round(success_rate,2)}\n")

if __name__ == '__main__':
    main()