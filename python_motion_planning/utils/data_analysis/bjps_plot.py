"""
@file: bjps_plot.py
@breif: plot the benchmarking results of BJPS and Dynamic A* algorithms
@author: Kota Kondo
@update: 2024.7.21
"""

import matplotlib.pyplot as plt
from python_motion_planning.utils.data_analysis.Params import Params

class BenchmarkingResults:

    def __init__(self, algorithm, bound, weight, cost, time, success_rate):
        self.algorithm = algorithm
        self.bound = bound
        self.weight = weight
        self.cost = cost
        self.time = time
        self.success_rate = success_rate

    def __str__(self) -> str:
        return f"Algorithm: {self.algorithm}, Bound: {self.bound}, Weight: {self.weight}, Cost: {self.cost}, Time: {self.time}, Success Rate: {self.success_rate}"
    
    def __repr__(self) -> str:
        return f"Algorithm: {self.algorithm}, Bound: {self.bound}, Weight: {self.weight}, Cost: {self.cost}, Time: {self.time}, Success Rate: {self.success_rate}"

    def get_algorithm(self):
        return self.algorithm
    
    def get_bound(self):
        return self.bound
    
    def get_weight(self):
        return self.weight
        
    def get_cost(self):
        return self.cost
    
    def get_time(self):
        return self.time
    
    def get_success_rate(self):
        return self.success_rate

def plot_results(bound_list, bjps_best_nodes: set, dynamic_a_star_best_nodes: set, title: str, figure_name: str):

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    bjps_node_list = []
    dynamic_a_star_node_list = []

    for bound in bound_list:
        
        bjps_node_list.append(bjps_best_nodes[f"{bound}"])
        dynamic_a_star_node_list.append(dynamic_a_star_best_nodes[f"{bound}"])
        
    # plot the best cost for each bound
    axes[0].plot(bound_list, [node.get_cost() for node in bjps_node_list], "go-", label="BJPS")
    axes[0].plot(bound_list, [node.get_cost() for node in dynamic_a_star_node_list], "ro-", label="Dynamic A*")
    axes[0].set_title(title)

    # plot the cost for each bound
    
    axes[1].plot(bound_list, [node.get_time() for node in bjps_node_list], "go-", label="BJPS")
    axes[1].plot(bound_list, [node.get_time() for node in dynamic_a_star_node_list], "ro-", label="Dynamic A*")

    axes[1].set_xlabel("Bound")

    # set ylabel on the left side for cost
    axes[0].set_ylabel("Cost")
    
    # set ylabel on the right side for time
    axes[1].set_ylabel("Time")
    axes[0].legend()
    axes[0].legend()

    # save the figure
    plt.savefig(f"/home/kkondo/Downloads/tmp/benchmark/{figure_name}.png")

def main():
    
    # read benchmarking results
    file_path = "/home/kkondo/code/python_motion_planning/benchmarking_results.txt"

    # read the file
    with open(file_path, "r") as f:
        
        # read lines
        lines = f.readlines()
        
        # initialize the results
        list_nodes = []
        for line in lines:
            # split the line
            elements = line.split(",")
            algorithm = elements[0].split(": ")[1]
            bound = float(elements[1].split(": ")[1])
            weight = float(elements[2].split(": ")[1])
            cost = float(elements[3].split(": ")[1])
            time = float(elements[4].split(": ")[1])
            success_rate = float(elements[5].split(": ")[1])
            list_nodes.append(BenchmarkingResults(algorithm, bound, weight, cost, time, success_rate))

    # plot the results

    # colors for each algorithm
    colors = ["r", "b"]

    # list of bounds and weights
    params = Params()
    bound_list = params.bound_list
    weight_list = params.weight_list

    # for each bound, find the best weight for cost
    bjps_best_nodes_wrt_cost = {f"{bound}": None for bound in bound_list}
    dynamic_a_star_best_nodes_wrt_cost = {f"{bound}": None for bound in bound_list}
    bjps_best_nodes_wrt_time = {f"{bound}": None for bound in bound_list}
    dynamic_a_star_best_nodes_wrt_time = {f"{bound}": None for bound in bound_list}

    for bound in bound_list:

        # get nodes with the fixed bound
        fixed_bound_nodes = [node for node in list_nodes if node.get_bound() == bound]

        # separate the nodes by the algorithm
        bjps_nodes = [node for node in fixed_bound_nodes if node.get_algorithm() == "bjps"]
        dynamic_a_star_nodes = [node for node in fixed_bound_nodes if node.get_algorithm() == "dynamic_a_star"]
        assert len(bjps_nodes) == len(dynamic_a_star_nodes), "The number of nodes for BJPS and Dynamic A* should be the same."

        # find the best weight wrt cost for each bound
        bjps_best_cost = float("inf")
        dynamic_a_star_best_cost = float("inf")
        for bjps_node, dynamic_a_star_node in zip(bjps_nodes, dynamic_a_star_nodes):
            if bjps_node.get_cost() < bjps_best_cost:
                bjps_best_cost = bjps_node.get_cost()
                bjps_best_nodes_wrt_cost[f"{bound}"] = bjps_node
            if dynamic_a_star_node.get_cost() < dynamic_a_star_best_cost:
                dynamic_a_star_best_cost = dynamic_a_star_node.get_cost()
                dynamic_a_star_best_nodes_wrt_cost[f"{bound}"] = dynamic_a_star_node
        
        # find the best weight wrt time for each bound
        bjps_best_time = float("inf")
        dynamic_a_star_best_time = float("inf")
        for bjps_node, dynamic_a_star_node in zip(bjps_nodes, dynamic_a_star_nodes):
            if bjps_node.get_time() < bjps_best_time:
                bjps_best_time = bjps_node.get_time()
                bjps_best_nodes_wrt_time[f"{bound}"] = bjps_node
            if dynamic_a_star_node.get_time() < dynamic_a_star_best_time:
                dynamic_a_star_best_time = dynamic_a_star_node.get_time()
                dynamic_a_star_best_nodes_wrt_time[f"{bound}"] = dynamic_a_star_node

    # plot the results for best cost with respect to bound
    plot_results(bound_list, bjps_best_nodes_wrt_cost, dynamic_a_star_best_nodes_wrt_cost, "Best Cost and Corresponding Time for Each Bound", "best_cost")

    # plot the results for best time with respect to bound
    plot_results(bound_list, bjps_best_nodes_wrt_time, dynamic_a_star_best_nodes_wrt_time, "Best Time and Corresponding Cost for Each Bound", "best_time")


if __name__ == '__main__':
    main()