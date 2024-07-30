"""
@file: global_example.py
@breif: global planner application examples
@author: Winter
@update: 2023.3.2
"""

from python_motion_planning.utils import Grid3D, SearchFactory
from argparse import ArgumentParser

if __name__ == '__main__':
    '''
    path searcher constructor
    '''
    search_factory = SearchFactory()

    '''
    graph search
    '''

    args = ArgumentParser()
    args.add_argument("-s", "--search", type=str, default="bjps3d", help="search algorithm")
    search_algorithm = args.parse_args().search
    
    # environment parameters
    bound = 5.0 # length of each interval (jump bound in BJPS)
    is_static = False if search_algorithm == "bjps3d" or search_algorithm == "dynamic_a_star3d" or search_algorithm == "djps3dpp" else True

    # planner parameters
    weight = 1.0

    # build environment
    start = (5, 5, 1)
    goal = (45, 5, 5)
    env = Grid3D(51, 31, 21, bound, weight, is_static)

    # creat planner
    planner = search_factory(search_algorithm, start=start, goal=goal, env=env, bound=bound, weight=weight)
    # planner = search_factory("a_star", start=start, goal=goal, env=env)
    # planner = search_factory("dijkstra", start=start, goal=goal, env=env)
    # planner = search_factory("gbfs", start=start, goal=goal, env=env)
    # planner = search_factory("theta_star", start=start, goal=goal, env=env)
    # planner = search_factory("lazy_theta_star", start=start, goal=goal, env=env)
    # planner = search_factory("s_theta_star", start=start, goal=goal, env=env)
    # planner = search_factory("jps", start=start, goal=goal, env=env)
    # planner = search_factory("bjps", start=start, goal=goal, env=env)
    # planner = search_factory("d_star", start=start, goal=goal, env=env)
    # planner = search_factory("lpa_star", start=start, goal=goal, env=env)
    # planner = search_factory("d_star_lite", start=start, goal=goal, env=env)
    # planner = search_factory("voronoi", start=start, goal=goal, env=env, n_knn=4,
    #                             max_edge_len=10.0, inflation_r=1.0)

    # animation
    planner.run()