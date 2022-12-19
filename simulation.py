import matplotlib.pyplot as plt
import pprint
import numpy as np
import matplotlib.colors as colors
import pickle
import time
import itertools
from mst_prototype import map2tree, node_values, map_builder

from mst_prototype import raw_nodevalue, calculate_expected_utility

pp = pprint.PrettyPrinter(compact=False)

import models

#with open(f'__experiment_1/parsed_data/tree.pickle', 'rb') as handle:
#     TREE = pickle.load(handle)

# for a utility function, determine the best path using that utility function. Find best path to exit
def best_path(maze, raw_nodevalue_func, params):
    "returns the full trajectory of best path based on some node value function and parameter"
    TREE = map2tree(maze)
    value_summary = node_values(maze, params, raw_nodevalue_func)
    #print(value_summary)
    nid = TREE[0]
    nid=0
    path = [nid] # list of node ids

    while True:

        # # break if at leaf node
        if not TREE[nid]['children']:
            #path.append(nid) # append to path if no children -- exit found or exhausted all options 
            return path

        # non-decision (num(children) <= 1) nodes just gets added
        if len(TREE[nid]['children']) == 1:
            #add child to path
            nid = list(TREE[nid]['children'])[0]
            path.append(nid)
            continue

        # find best child dependent on raw_nodevalue (utility function)
        choose_from = []
        for cid,value in value_summary[nid].items():
            choose_from.append((value,cid))
        
        #choose best child depending on the value of the node
        best_child = max(choose_from, key = lambda x: x[0])[1]
        # update node
        nid = best_child

        # update path
        path.append(nid)

    return path




def visualize_maze(maze, ax=None):
    """
    0: hidden, 2: exit, 3: wall, 5: start, 6: open
    """
    if ax is None:
        _, ax = plt.subplots(1)

    # with open(f'__experiment_1/mazes/{world}.txt') as f:
    #     ncols = int(f.readline())
    #     nrows = int(f.readline())
    #
    #     lines = f.readlines()
    #lines=maze

    ncols= len(maze[0])
    nrows=len(maze)

    # pcolormesh anchor is at bottom left corner, so need to reverse rows
    #maze1 = [[int(cell) for cell in list(row)[:ncols]] for row in lines][::-1]
    maze1=maze[::-1]
    # short circuits at shortest nested list if table is jagged:
    #maze1=list(map(list, zip(*maze1)))
    #pp.pprint(maze1)

    # custom color map
    cmap = colors.ListedColormap(['#9c9c9c', 'white', '#d074a4', '#b0943d', 'white', '#a1c38c', 'white', '#f5f5dc'])
    boundaries = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=False)

    # draw maze
    ax.pcolormesh(maze1, edgecolors='lightgrey', linewidth=1, cmap=cmap, norm=norm)
    ax.set_aspect('equal')

    # Major ticks label (for readability of plot, (0,0) at top left)
    ax.set_xticks([x+0.5 for x in range(ncols)])
    ax.set_xticklabels([y for y in range(ncols)])
    ax.set_yticks([y+0.5 for y in range(nrows)])
    ax.set_yticklabels([y for y in range(nrows)][::-1])


def visualize_path(maze, path, ax):
    "ax already has the drawing of the maze, path is in terms of node ids"

    # with open(f'__experiment_1/mazes/{world}.txt') as f:
    #     ncols = int(f.readline())
    #     nrows = int(f.readline())
    #path=path[1:] #FOR TESTING SAKE
    ncols= len(maze[0])
    nrows=len(maze)

    def jitter(arr):
        stdev = .025 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    # draw paths
    TREE=map2tree(maze)
    #print("keys in tree:. ", len(TREE))
    for nid in path[1:]:
        c, r = zip(*[(c + 0.5, nrows - r - 0.5) for r,c in TREE[nid]['path_from_par']])
        c, r = jitter(c), jitter(r)
        ax.plot(c, r, 'o--',  markersize=4, label=nid)
        # ax.plot(x[0], y[0], 's', markersize=8, color='purple')

    ax.legend(loc='upper left', bbox_to_anchor=(0,-0.1))


def visualize_juxtaposed_best_paths(maze):

    _, axs = plt.subplots(1, 3)
    axs = axs.flat

    raw_nodevalue_func_and_params = [('Expected_Utility', raw_nodevalue, (1,1,1,1)), #output from math equation, params that go into equation
                                     ('Discounted_Utillity', raw_nodevalue, (1,1,1,1)),#params: (gamma,beta,k,tau)
                                     ('Probability_Weighted', raw_nodevalue, (1,1,1,1))]

    #compare expected, discounted, probability_weighted -- determine best path and visualize it
    for ax, (model_name, raw_nodevalue_func, params) in zip(axs, raw_nodevalue_func_and_params):

        path = best_path(maze, raw_nodevalue_func, params)
        visualize_maze(maze, ax)
        visualize_path(maze, path, ax)
        ax.set_title(model_name)

    plt.show()



if __name__ == "__main__":
    world_00 = ((3,3,3,3,3,3,3,3,3),
                (3,3,3,3,0,0,3,3,3),
                (3,6,6,6,6,6,6,6,3),
                (3,6,3,3,0,0,3,3,3),
                (3,6,3,3,0,2,3,3,3),
                (3,6,3,3,3,3,3,3,3),
                (3,6,6,6,6,5,6,6,3),
                (3,3,3,3,3,3,3,0,3),
                (3,3,3,3,3,3,3,3,3),)
    path_00 = [0, 0, 1]
    _, ax_00 = plt.subplots(1, 1)
    # random_path_00 = best_path(world_00, calculate_expected_utility, (1,1,1))

    # print(random_path_00)
    # visualize_maze(world_00, ax_00)
    # visualize_path(world_00,random_path_00,ax_00)
    # plt.show()

    world = '4ways'
    world= ((3, 3, 3, 3, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 0, 3, 0, 3, 3),
         (3, 3, 3, 3, 0, 3, 0, 3, 3),
         (3, 5, 6, 6, 6, 6, 6, 6, 3),
         (3, 6, 3, 3, 3, 3, 3, 6, 3),
         (3, 6, 6, 6, 6, 6, 6, 6, 3),
         (3, 3, 0, 0, 3, 3, 3, 3, 3),
         (3, 3, 3, 3, 3, 3, 3, 3, 3),)

    map2_test  =((3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
                (3, 0, 3, 3, 3, 0, 0, 3, 6, 6, 6, 6, 0, 0, 3),
                (3, 6, 6, 6, 3, 0, 0, 3, 6, 3, 3, 3, 0, 0, 3),
                (3, 3, 3, 6, 3, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3),
                (3, 3, 0, 6, 3, 6, 3, 3, 6, 6, 6, 6, 6, 6, 3),
                (3, 3, 0, 6, 3, 6, 3, 3, 6, 0, 0, 3, 3, 6, 3),
                (3, 3, 3, 6, 3, 6, 3, 3, 6, 0, 0, 3, 3, 6, 3),
                (3, 3, 3, 6, 6, 6, 3, 3, 5, 3, 3, 3, 3, 6, 3),
                (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3))


    ncols, nrows = 13, 9
    start = (5 ,4)

    path = {(3 ,1), (3 ,2), (3 ,3), (3 ,4), (3 ,5), (3 ,6), (3 ,7), (3 ,8),
            (5 ,1), (5 ,2), (5 ,3), (5 ,4), (5 ,5), (5 ,6), (5 ,7), (5 ,8),
            (4 ,1), (4 ,8)}

    black = {(6 ,2), (7 ,2),
             (6 ,7), (7 ,7),
             (2 ,4), (1 ,4), (1 ,5), (1 ,6),
             (4 ,9), (4 ,10), (4 ,11)}

    gen_map = map_builder(nrows, ncols, black, path, start)

    random_generated_path = best_path(gen_map, calculate_expected_utility, (1,1,1))
    # print("BEST PATH: ", random_generated_path)
         
    #
    # world= ((3, 3, 3, 3, 3, 3, 3, 3, 3),
    #      (3, 3, 3, 3, 0, 3, 0, 3, 3),
    #      (3, 3, 3, 3, 0, 3, 0, 3, 3),
    #      (3, 6, 5, 6, 6, 6, 6, 6, 3),
    #      (3, 6, 3, 3, 3, 3, 3, 6, 3),
    #      (3, 6, 6, 6, 6, 6, 6, 6, 3),
    #      (3, 3, 0, 0, 3, 3, 3, 3, 3),
    #      (3, 3, 3, 3, 3, 3, 3, 3, 3),)


    comp1 = ((3,3,3,3,3,3,3,3,3),
             (3,3,3,3,0,2,3,3,3),
             (3,0,3,0,0,0,3,0,3),
            (3,6,6,6,5,6,6,6,3),
            (3,0,3,0,0,0,3,0,3),
            (3,3,3,3,0,3,3,3,3),
             (3,3,3,3,3,3,3,3,3),)

    comp2=((3,3,3,3,3,3,3,3,3),
        (3,0,0,0,3,0,0,0,3),
           (3,0,3,0,3,0,3,0,3),
           (3,6,5,6,6,6,6,6,3),
           (3,0,3,0,3,0,3,0,3),
           (3,0,0,0,3,0,2,0,3),
           (3,3,3,3,3,3,3,3,3),)

    maze1 = ((3,3,3,3,3,3,3,3,3),
            (3,0, 0, 0, 3, 2, 0, 0,3),
            (3,3, 0, 3, 3, 3, 0, 3,3),
            (3,6, 6, 6, 5, 6, 6, 6,3),
            (3,3, 0, 3, 3, 3, 0, 3,3),
            (3,0, 0, 0, 3, 0, 0, 0,3),
            (3,3,3,3,3,3,3,3,3),)

    Maze2=((3,3,3,3,3,3,3),
           (3,0, 0, 6, 0, 0,3),
            (3,0, 3, 6, 3, 2,3),
           (3,3, 3, 5, 3, 3,3),
            (3,0, 3, 6, 3, 0,3),
            (3,0, 0, 6, 0, 0,3),
           (3,3,3,3,3,3,3),)

    Maze3=((3,3,3,3,3,3,3,3,3),
           (3,0, 0, 3, 6, 0, 0, 0,3),
           (3,0, 3, 3, 6, 3, 3, 0,3),
           (3,0, 3, 3, 6, 3, 3, 3,3),
           (3,6, 6, 6, 5, 6, 6, 6,3),
           (3,3, 3, 3, 6, 3, 3, 0,3),
           (3,2, 3, 3, 6, 3, 3, 0,3),
           (3,0, 0, 0, 6, 3, 0, 0,3),
           (3,3,3,3,3,3,3,3,3),)

    Maze4=((3,3,3,3,3,3,3,3,3),
           (3,0, 0, 0, 3, 0, 0, 0, 3),
           (3,0, 3, 0, 3, 0, 3, 0, 3),
           (3,6, 5, 6, 6, 6, 6, 6, 3),
           (3,0, 3, 0, 3, 0, 3, 0, 3),
           (3,0, 0, 0, 3, 0, 2, 0, 3),
           (3,3,3,3,3,3,3,3,3),)

    Maze5=((3,3,3,3,3,3,3),
           (3,3, 3, 5, 3, 3, 3),
           (3,0, 3, 6, 3, 0, 3),
           (3,0, 0, 6, 0, 0, 3),
           (3,3, 3, 6, 3, 3, 3),
           (3,0, 3, 6, 3, 2, 3),
           (3,0, 0, 6, 0, 0, 3),
           (3,3,3,3,3,3,3),)
           
    ##Maze 5 has two root nodes at the beginning
    #Maze 6 has a disjoint path

    Maze6=((3,3,3,3,3,3,3,3,3,3,3),
           (3,3, 3, 3, 0, 0, 3, 3, 0, 0, 3),
           (3,3, 3, 3, 0, 0, 3, 3, 0, 0, 3),
           (3,3, 3, 3, 0, 3, 3, 3, 0, 3, 3),
           (3,6, 6, 6, 6, 6, 6, 6, 6, 5, 3),
           (3,3, 0, 3, 3, 3, 0, 3, 3, 3, 3),
           (3,2, 0, 3, 3, 0, 0, 3, 3, 3, 3),
           (3,0, 0, 3, 3, 0, 0, 3, 3, 3, 3),
           (3,3,3,3,3,3,3,3,3,3,3),)

    # visualize_juxtaposed_best_paths(Maze5)

    Maze11 = ((3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 0, 3, 3, 3, 0, 0, 3, 6, 6, 6, 6, 0, 0, 3),
        (3, 6, 6, 6, 3, 0, 0, 3, 6, 3, 3, 3, 0, 0, 3),
        (3, 3, 3, 6, 3, 6, 6, 6, 6, 3, 3, 3, 3, 3, 3),
        (3, 3, 0, 6, 3, 6, 3, 3, 6, 6, 6, 6, 6, 6, 3),
        (3, 3, 0, 6, 3, 6, 3, 3, 6, 0, 0, 3, 3, 6, 3),
        (3, 3, 3, 6, 3, 6, 3, 3, 6, 0, 0, 3, 3, 6, 3),
        (3, 3, 3, 6, 6, 6, 3, 3, 5, 3, 3, 3, 3, 6, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),)

    # Maze21 = ((3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
    # (3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 3, 3, 3, 3),
    # (3, 6, 6, 6, 6, 6, 6, 6, 3, 3, 6, 3, 3, 3, 3),
    # (3, 3, 0, 0, 3, 6, 3, 3, 3, 3, 6, 6, 6, 5, 3),
    # (3, 3, 3, 3, 3, 6, 3, 3, 3, 3, 6, 3, 3, 3, 3),
    # (3, 6, 3, 3, 3, 6, 3, 0, 3, 3, 6, 3, 0, 3, 3),
    # (3, 6, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3),
    # (3, 6, 6, 6, 3, 3, 0, 0, 3, 3, 3, 6, 3, 3, 3),
    # (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),)

    # visualize_maze(Maze21)
    # plt.show()
