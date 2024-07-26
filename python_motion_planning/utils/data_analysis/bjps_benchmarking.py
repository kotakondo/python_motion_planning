"""
@file: global_example.py
@breif: global planner application examples
@author: Winter
@update: 2023.3.2
"""

from python_motion_planning.utils import Grid, Map, SearchFactory
from argparse import ArgumentParser
import itertools
from datetime import datetime
from tqdm import tqdm
from python_motion_planning.utils.data_analysis.Params import Params
from python_motion_planning.utils.data_analysis.bjps_plot import main as bjps_plot_main

def main():
    
    # path searcher constructor
    search_factory = SearchFactory()

    # build environment
    start = (5, 5)
    goal = (45, 5)

    # benchmark parameters
    params = Params()
    algorithms = params.algorithms
    num_runs = params.num_runs
    bound_list = params.bound_list
    weight_list = params.weight_list
    
    # create combinations for algorithms, bounds, and weights
    combinations = list(itertools.product(algorithms, bound_list, weight_list))

    # if there is a file, delete it
    with open("benchmarking_results.txt", "w") as f:
        f.write("")

    # run benchmarking
    for search_algorithm, bound, weight in tqdm(combinations):

        env = Grid(51, 31, bound)

        # print("Algorithm: ", search_algorithm)
        # print("Bound: ", bound)
        # print("Weight: ", weight)

        # creat planner
        planner = search_factory(search_algorithm, start=start, goal=goal, env=env, bound=bound, weight=weight)

        # benchmarking
        cost, time, success_rate = planner.benchmarking_run(num_runs, save_fig=True)

        # print("Cost: ", cost)
        # print("Time: ", time)
        # print("Success Rate: ", success_rate)

        # save it to a file
        with open("benchmarking_results.txt", "a") as f:
            f.write(f"Algorithm: {search_algorithm}, Bound: {bound}, Weight: {weight}, Cost: {round(cost,2)}, Time: {round(time,2)}, Success Rate: {round(success_rate,2)}\n")

    print("Benchmarking is done.")

    # plot the results
    bjps_plot_main()

if __name__ == '__main__':
    main()