
# !! NOTE !!
# This tree builder only works for paths of width 1

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import pprint
#from simulation import *

global utility_functions
utility_functions = ['random', 'expected_utility', 'discounted_utility', 'pwu', 'combination', 'novel_utility']

pp = pprint.PrettyPrinter(compact=False, width=90)

def memoize(function):
    """ ram cacher """
    memo = {}
    def wrapper(*args):
        if args not in memo:
            memo[args] = function(*args)
        return memo[args]
    return wrapper


# ----------------------
# Map Builder
# ----------------------

def map_builder(nrows, ncols, black, path, start):

    return tuple(tuple(5 if (i ,j )==start else 6 if (i ,j) in path else 0 if (i ,j) in black else 3 for j in range(ncols)) for i in range(nrows))


# ----------------------
# Tree Builder
# ----------------------

def new_update_map(map_, old_pos, new_pos):
    '''
    if hidden rooms are revealed, update these changes in the map 
    returns: updated map 
    '''
    r,c = new_pos
    observations = new_get_observations(map_, r,c)

    map_updated = [[6 if (r ,c) in observations else map_[r][c]
                    for c in range(len(map_[0]))]
                   for r in range(len(map_))]

    map_updated[old_pos[0]][old_pos[1]] = 6

    map_updated = tuple(tuple(row) for row in map_updated)

    return map_updated

def new_possible_paths(map, pos):
    '''
    possible paths dtetermine the steps that we can take in any adjacent direction
    create a path to an observable room
    '''

    #obtain rows and columns
    ncols, nrows = len(map[0]), len(map)

    #initiate agenda
    agenda = [ [pos] ]
    #each path is the shortest possible path to an observation location
    paths = []

    while agenda:

        path = agenda.pop(0)
        #print("CURR PSTH: ", path)
        r_, c_ = path[-1]

        ##TAKE A STEP IN A CARDINAL DIRECTION
        for rr, cc in ((0 ,1), (0 ,-1), (1 ,0), (-1 ,0)):

            # Stay in bounds
            r, c = max(min(r_ +rr, nrows -1), 0), max(min(c_ +cc, ncols -1), 0) #assume this is working properly

            # ignore if neigh is a wall or the point already exists in the
            if map[r][c] == 3 or (r ,c) in path:
                continue

            #IF YOU'RE ABLE TO TAKE A STEP, ADD IT TO THE CURRENT PATH
            updated_path = path + [(r ,c)]
            #LOOK AROUND
            current_observations = new_get_observations(map,r,c)

            #ADD UPDATED PATH TO LIST OF TRAVERSABLE PATHS AS WELL AS WHAT IT OBSERVES
            if current_observations:
                paths.append((updated_path, current_observations))
            #IF NO OBSERVATIONS, ADD SUBPATH TO AGENDA
            if not current_observations:
                agenda.append(updated_path)

    return paths

def new_get_observations(map,r,c):
    '''
    obtains observations by taking a step in a cardinal direction. if the step results in "stepping" on a 0 (hidden room) bucket fill uncovers the room
    returns: a set of rooms that are uncovered while standing at location r,c
    '''
    ncols, nrows = len(map[0]), len(map)
    obs = set()
    for lr, lc in ((0 ,1), (0 ,-1), (1 ,0), (-1 ,0)):
        look_r, look_c = max(min(r +lr, nrows -1), 0), max(min(c +lc, ncols -1), 0)
        #if we observe a hidden room, populate it
        if map[look_r][look_c] == 0:
            unlocked_rooms = bucket_fill(map,look_r, look_c)
            obs.update(unlocked_rooms)

    return obs

def bucket_fill(map,r_loc, c_loc, revealed = None, counter =0):
    '''
    bucket fill reveals clumps of hidden rooms 
    '''
    if not revealed:
        revealed = set()
    #emergency recursion break in case somethign happens
    if counter >=10:
        return 
    #base case-- if the current location is out of bounds
    if r_loc < 0 or r_loc > len(map)-1 or c_loc < 0 or c_loc > len(map[0])-1: # ncols, nrows = len(map[0]), len(map)
        return
    #base case -- if the map at the current location is NOT a 2
    if map[r_loc][c_loc] != 0:
        return
    #base case -- the location we're looking at was already revealed
    if (r_loc,c_loc) in revealed:
        return

    revealed.add((r_loc, c_loc))
    #try to populate in every cardinal direction
    bucket_fill(map,r_loc, c_loc+1, revealed,counter=counter+1) #right
    bucket_fill(map,r_loc-1, c_loc, revealed,counter=counter+1) #up
    bucket_fill(map,r_loc, c_loc-1, revealed,counter=counter+1) #left
    bucket_fill(map,r_loc+1, c_loc, revealed, counter=counter+1) #down
    return revealed



@memoize
def map2tree(map_):
    #pp.pprint(map_)

    # determine start position
    remains = 0
    for r, row in enumerate(map_):
        for c, val in enumerate(row):
            if val == 5:
                pos = (r ,c)
            elif val == 0:
                remains += 1

    tree = {0: {'pos': pos,
                'remains': remains,
                'path_from_par': [],
                'path_from_root': [],
                'steps_from_par': 0,
                'steps_from_root': 0,
                'celldistances': set(),
                'children': set(),
                'pid': None}}

    agenda = [(0, map_)]
    while agenda: # in each loop, find and append children -- which children being every potential possible child path from current path

        node, updated_map = agenda.pop(0)
        pos = tree[node]['pos']
        
        for path, observation in new_possible_paths(updated_map, pos): #for every node add a branch"
            branch = {'pos': path[-1],
                      'remains': tree[node]['remains' ] -len(observation),
                      'path_from_par': path,
                      'path_from_root': tree[node]['path_from_root'] + path,
                      'steps_from_par': len(path) - 1,
                      'steps_from_root': tree[node]['steps_from_root'] + len(path) - 1,
                      'celldistances': observation,
                      'children': set(), #set of nodes?
                      'pid': node,
                      'map': updated_map}

            new_node = max(tree ) +1
            agenda.append((new_node, new_update_map(updated_map, path[0], path[-1]))) #current error

            tree[node]['children'].add(new_node)
            tree[new_node] = branch
    #print("tree made")
    return tree


# ----------------------
# Map & Path Visualizer
# ----------------------


def map_visualizer(maze, node=None):
    """
    0: hidden, 2: exit, 3: wall, 5: start, 6: open
    """

    nrows, ncols = len(maze), len(maze[0])

    fig = plt.figure()
    ax = fig.add_subplot(111 ,aspect='equal')

    if node: # draw path
        tree = map2tree(maze)
        path = tree[node]['path_from_root']

        maze = tree[node]['map']
        maze = new_update_map(maze, tree[node]['pos'], path[-1])

        path = [(c ,r) for r ,c in path][::-1]
        x, y = zip(*[(x + 0.5, nrows - y - 0.5) for x ,y in path])
        ax.plot(x, y, 'o--',  markersize=4, label=node)
        ax.plot(x[-1], y[-1], 's', markersize=8, color='purple')

    maze = [[int(cell) for cell in list(row)[:ncols]] for row in maze][::-1]

    # custom color maze
    cmap = colors.ListedColormap \
        (['#9c9c9c', 'white', '#d074a4', '#b0943d', 'white', '#a1c38c', 'white', '#f5f5dc', 'moccasin'])
    boundaries = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=False)

    # draw maze
    ax.pcolormesh(maze, edgecolors='lightgrey', linewidth=1, cmap=cmap, norm=norm)
    ax.set_aspect('equal')

    # Major ticks positions
    ax.set_xticks([ i +0.5 for i in list(range(ncols))])
    ax.set_yticks([ i +0.5 for i in list(range(nrows))[::-1]])

    # Major ticks label (for readability of plot, (0,0) at top left)
    ax.set_xticklabels([str(i) for i in list(range(ncols))])
    ax.set_yticklabels([str(i) for i in list(range(nrows))])

    plt.show()


# ----------------------
# Node Value Visualizer
# ----------------------

# parameters
TAUS    = np.linspace(0.1, 1, 10)
GAMMAS  = np.linspace(0.1 ,1.5 ,10)
BETAS   = np.linspace(0.1 ,2 ,10)
KAPPAS  = np.linspace(0.1 ,5 ,20)

#maps state to probability of choosing it
#maps raw utility value to probability 
def softmax(values, tau):
    """ small values are better, tau=1 is optimal
    large tau converges to random agent """
    numer = [np.exp(-v * ( 1 /tau)) for v in values]
    denom = sum(numer)
    return [ n /denom for n in numer]

def weight(p, beta):
    """ probability weighting function: convert probability p to weight """
    return np.exp( -1 * (-np.log(p) )**beta )
    # return p**beta / (p**beta + (1-p)**beta) ** (1/beta)

# raw node value function: eu, du, pwu #IGNORE
def ra_nodevalue(maze, nid, gamma=1, beta=1):
    """ return raw node value BEFORE softmax being applied """

    tree = map2tree(maze)
    cell_distances = tree[nid]["celldistances"]

    value, p_exit = 0, 0

    if tree[nid]["pid"] != "NA":

        p_exit = len(cell_distances)/tree[tree[nid]["pid"]]["remains"]

        value += weight(p_exit, beta) * (tree[nid]["steps_from_root"] + np.mean(list(cell_distances)))

    if tree[nid].get("children", []):
        min_child_value = float("inf")

        for cid in tree[nid]["children"]:
            child_value = raw_nodevalue(maze, cid, gamma, beta)
            if child_value < min_child_value:
                min_child_value = child_value

        value += gamma * weight(1-p_exit, beta) * min_child_value

    return value

#what is the unweighted probability of going in a certain way (assume random step -- random walk?)
#put other utility functions here?
### THIS FUNCTION IS NOT BEING USED, UTILITY FUNCTIONS CALLED DIRECTLY
def raw_nodevalue(maze, nid, fxn = 'random', gamma=1, beta=1, k=1):
    """ return raw node value BEFORE softmax being applied """
    global utility_functions #defined at top of script
    tree= map2tree(maze)

    if fxn in utility_functions:
        if fxn == 'random':
            return calculate_random_utility(maze,nid,gamma,beta,k)
        elif fxn == 'expected_utility':
            return calculate_expected_utility(maze,nid,gamma,beta,k)
        elif fxn == 'discounted_utility':
            return calculate_discounted_utility(maze,nid,gamma,beta,k)
        elif fxn == 'pwu':
            return calculate_pwu(maze,nid,gamma,beta,k)
        elif fxn == 'combined':
            return calculate_combined_utility(maze,nid,gamma,beta,k)
        elif fxn == 'novel_utility':
            return calculate_novel_utility(maze,nid,gamma,beta,k)
    else:
        raise NameError


def calculate_random_utility(maze,nid,gamma=1,beta=1,k=1):
    return 1

def calculate_prob_child_utility(maze,nid,gamma=1,beta=1,k=1):  
    '''node value dependent on the number of children a particular node has'''
    Tree= map2tree(maze)
    pid= Tree[nid]['pid'] #tree[node]['pid'] ?? 
    return 1/len(Tree[pid]['children'])


def calculate_expected_utility(maze, nid, gamma=1, beta=1, k=1):
    '''
    Calculates the expected utility for a specific node
    '''
    tree = map2tree(maze)
    pid = tree[nid]['pid']
    total_hidden_cells = tree[pid]['remains'] 
    observed_hidden_cells = tree[nid]['celldistances'] 

    #BASE CASE 1: all hidden cells have been enumerated
    if total_hidden_cells == 0: 
        return 0

    pi = len(observed_hidden_cells) / total_hidden_cells
    si = tree[nid]['steps_from_root']
    ei = 0
    if len(observed_hidden_cells) != 0:
        ei = get_ei (nid, tree, observed_hidden_cells) #obtain expected manhattan distance
                
    #BASE CASE 2: EXIT HAS BEEN FOUND, NO MORE CHILDREN TO ENUMERATE
    if len(tree[nid]['children']) == 0: # leaf node, but unsure how the tree calculates leaf nodes
        return pi*(si + ei)

    child_values = []
    for child in tree[nid]['children']: # recurse on children
        cur_expected_util = calculate_expected_utility(maze, child)
        child_values.append(cur_expected_util)
    
    expected_utility = pi*(si+ei) + (1-pi)*min(child_values) 
    #print(expected_utility)
    return expected_utility


def get_ei(nid, tree, observed_hidden_cells):
    '''
    assuming expected distance = p1(v1) + p2(v2) + ... 
    and p1=p2=... = 1/p and v = number of steps to a specific cell assuming the exit is at that cell
    '''
    expected_sum = 0
    for hc in observed_hidden_cells: # (x, y)
        pos_n = tree[nid]['pos']
        manhattan_dist = abs(pos_n[0] - hc[0]) + abs(pos_n[1] - hc[1])
        expected_sum  += manhattan_dist

    return expected_sum/len(observed_hidden_cells)


def calculate_discounted_utility(maze, nid, gamma = 1, beta = 1, k=1):
    #Q_du(Ni) = pi(si +ei) + γ(1−pi) min Q_du(cj)
    tree = map2tree(maze)
    pid = tree[nid]['pid']
    total_hidden_cells = tree[pid]['remains'] 
    observed_hidden_cells = tree[nid]['celldistances'] 

    #BASE CASE 1: all hidden cells have been enumerated
    if total_hidden_cells == 0: 
        return 0

    pi = len(observed_hidden_cells) / total_hidden_cells
    si = tree[nid]['steps_from_root']
    ei = 0
    if len(observed_hidden_cells) != 0:
        ei = get_ei (nid, tree, observed_hidden_cells) #obtain expected manhattan distance
        
    #BASE CASE 2: EXIT HAS BEEN FOUND, NO MORE CHILDREN TO ENUMERATE
    if len(tree[nid]['children']) == 0: # leaf node, but unsure how the tree calculates leaf nodes
        return pi*(si + ei) 
        #return 1

    child_values = []
    for child in tree[nid]['children']: # recurse on children
        cur_discounted_util = calculate_discounted_utility(maze, child,gamma,beta,k)
        child_values.append(cur_discounted_util)

    discounted_utility = pi*(si+ei) + gamma*(1-pi)*min(child_values) 
    return discounted_utility

def calculate_pwu(maze, nid, gamma = 1, beta = 1, k=1):
    import math
    import numpy as np
    probability_weighing_fxn = lambda x: np.exp(-1*(abs(np.log(x))**beta)) #pi
    
    #Q_pwu(Ni) = π(pi)(si+ei)+π(1−pi) min Q_pwu(Cj)
    tree = map2tree(maze)
    pid = tree[nid]['pid']
    total_hidden_cells = tree[pid]['remains'] 
    observed_hidden_cells = tree[nid]['celldistances'] 

    #BASE CASE 1: all hidden cells have been enumerated
    if total_hidden_cells == 0: 
        return 0

    pi = len(observed_hidden_cells) / total_hidden_cells
    si = tree[nid]['steps_from_root']
    ei = 0
    if len(observed_hidden_cells) != 0:
        ei = get_ei (nid, tree, observed_hidden_cells) #obtain expected manhattan distance
                
    #BASE CASE 2: EXIT HAS BEEN FOUND, NO MORE CHILDREN TO ENUMERATE
    if len(tree[nid]['children']) == 0: # leaf node, but unsure how the tree calculates leaf nodes
        return pi*(si + ei) 

    child_values = []
    for child in tree[nid]['children']: # recurse on children
        curr_pwu = calculate_pwu(maze, child,gamma,beta,k)
        child_values.append(curr_pwu)

    pwu = probability_weighing_fxn(pi)*(si+ei) + probability_weighing_fxn(1-pi)*min(child_values)    
    
    return pwu

def steps_heuristic(maze,nid,gamma=1,beta=1,k=1):
    tree = map2tree(maze)
    si = tree[nid]['steps_from_root']
    return si

def cells_heuristic(maze,nid,gamma=1,beta=1,k=1):
    tree = map2tree(maze)
    observed_hidden_cells = tree[nid]['celldistances'] 
    return -len(observed_hidden_cells)


def calculate_combined_utility(maze, nid, gamma = 1, beta = 1, k=1):
    import math
    probability_weighing_fxn = lambda x: math.exp(-abs(math.log(x))**beta) #pi
    
    #Q_pwu(Ni) = π(pi)(si+ei)+π(1−pi) min Q_pwu(Cj)
    tree = map2tree(maze)
    pid = tree[nid]['pid']
    total_hidden_cells = tree[pid]['remains'] 
    observed_hidden_cells = tree[nid]['celldistances'] 

    #BASE CASE 1: all hidden cells have been enumerated
    if total_hidden_cells == 0: 
        return 0

    pi = len(observed_hidden_cells) / total_hidden_cells
    si = tree[nid]['steps_from_root']
    ei = 0
    if len(observed_hidden_cells) != 0:
        ei = get_ei (nid, tree, observed_hidden_cells) #obtain expected manhattan distance
  
    #BASE CASE 2: EXIT HAS BEEN FOUND, NO MORE CHILDREN TO ENUMERATE
    if len(tree[nid]['children']) == 0: # leaf node, but unsure how the tree calculates leaf nodes
        return pi*(si + ei)

    child_values = []
    for child in tree[nid]['children']: # recurse on children
        curr_pwu = calculate_combined_utility(maze, child, gamma, beta, k)
        child_values.append(curr_pwu)
    
    combined = probability_weighing_fxn(pi)*(si+ei) + gamma* probability_weighing_fxn(1-pi)*min(child_values)    
    return combined

def calculate_novel_utility(maze,nid, gamma = 1, beta = 1, k=1):
    '''hyperbolic time discounting using expected utility'''
    #{g(D)=1/{1+kD} with D = delay in reward (steps to node), k = degree of discount
    hyperbolic_fxn = lambda x: 1/(1+k*x)

    tree = map2tree(maze)
    pid = tree[nid]['pid']
    total_hidden_cells = tree[pid]['remains'] 
    observed_hidden_cells = tree[nid]['celldistances'] 

    #BASE CASE 1: all hidden cells have been enumerated
    if total_hidden_cells == 0: 
        return 0

    pi = len(observed_hidden_cells) / total_hidden_cells
    si = tree[nid]['steps_from_root']
    ei = 0
    if len(observed_hidden_cells) != 0:
        ei = get_ei (nid, tree, observed_hidden_cells) #obtain expected manhattan distance
                
    #BASE CASE 2: EXIT HAS BEEN FOUND, NO MORE CHILDREN TO ENUMERATE
    if len(tree[nid]['children']) == 0: # leaf node, but unsure how the tree calculates leaf nodes
        return hyperbolic_fxn(si)*(pi*(si + ei))

    child_values = []
    for child in tree[nid]['children']: # recurse on children
        #cur_expected_util = calculate_novel_utility(maze, child)
        cur_expected_util = calculate_expected_utility(maze, child)
        child_values.append(cur_expected_util)
    
    expected_utility = pi*(si+ei) + (1-pi)*min(child_values) 


    #combined = probability_weighing_fxn(pi)*(si+ei) + gamma* probability_weighing_fxn(1-pi)*min(child_values)    
    return hyperbolic_fxn(si)*expected_utility 


def node_values(maze, parameters, raw_nodevalue_func):

    values_summary = {} # {nid: {cid: value, cid: value, ...}} with nid = node id and cid = child id?
    tree = map2tree(maze)

    for nid in tree:

        if nid == 'root':
            continue

        #get children of current node
        children = tree[nid]['children']

        # ignore nid if it's not a decision node
        #if there are no possible paths forward(enumerated all options)continue
        if len(children) <= 1: 
            continue

        gamma,beta,k,tau = parameters

        values_summary[nid] = {}
        #obtain node values (raw utility value) for each child of current node
        raw_values = [ raw_nodevalue_func(maze, cid, gamma,beta,k) for cid in children ] 
        #obtain node probability value 
        values = softmax(raw_values, tau)
        values_summary[nid] = {cid: val for cid ,val in zip(children, values)} #add child nodes, values to current node child dict

    return values_summary


def visualize_decision(maze, pid, ax):
    """
    0: hidden, 2: exit, 3: wall, 5: start, 6: open
    """
    tree = map2tree(maze)

    # custom color map
    cmap = colors.ListedColormap(['#9c9c9c', 'white', '#d074a4', '#b0943d', 'white', '#a1c38c', 'white', '#f5f5dc'])
    boundaries = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=False)

    # XXX fix orientation
    maze = [[int(cell) for cell in list(row)[:ncols]] for row in maze][::-1]

    # draw maze
    ax.pcolormesh(maze, edgecolors='lightgrey', linewidth=1, cmap=cmap, norm=norm)
    ax.set_aspect('equal')

    ax.set_xticks([0.5 + j for j in range(ncols)], minor=False)
    ax.set_xticklabels(list(range(ncols)))
    ax.set_yticks([0.5 + i for i in range(nrows)], minor=False)
    ax.set_yticklabels(list(range(nrows))[::-1])

    # plot path
    for nid in tree[pid]['children']:
        path = [(c ,r) for r ,c in tree[nid]['path_from_root']]
        x, y = zip(*[(x + 0.5, nrows - y - 0.5) for x ,y in path])
        ax.plot(x, y, 'o--',  markersize=4, label=nid)
        ax.plot(x[0], y[0], 's', markersize=8, color='purple')

    ax.legend(loc='upper left', bbox_to_anchor=(1 ,1))


def visualize_nodevalues(maze, pid, parameters, param_indx, model_name, raw_nodevalue_func, ax):

    tree = map2tree(maze)

    values_summary = node_values(maze, parameters, raw_nodevalue_func)

    decision_summary = {nid: [] for nid in tree[pid]['children']}

    for param in parameters:
        for nid, val in values_summary[pid][param].items():

            decision_summary[nid].append(val)

    for nid, values in decision_summary.items():
        ax.plot([param[param_indx] for param in parameters], values, 'o--', markersize=3, label=nid)

    ax.set_title(model_name)
    ax.grid()
    ax.legend()


# def visualize_decision_and_nodevalues(maze, pid):

#     _, axs = plt.subplots(2 ,2)
#     axs = axs.flat

#     # draw maze and decision paths
#     visualize_decision(maze, pid, axs[0])

#     # node value plot for each model

#     parameters = [(tau ,1 ,1) for tau in TAUS]
#     raw_nodevalue_func = raw_nodevalue
#     # raw_nodevalue_func = random_nodevalue
#     visualize_nodevalues(maze, pid, parameters, 0, 'expected utility', raw_nodevalue_func, axs[1])

#     parameters = [(1, gamma, 1) for gamma in GAMMAS]
#     raw_nodevalue_func = raw_nodevalue
#     #raw_nodevalue_func = random_nodevalue
#     visualize_nodevalues(maze, pid, parameters, 1, 'discounted utility', raw_nodevalue_func, axs[2])

#     parameters = [(1, 1, beta) for beta in BETAS]
#     raw_nodevalue_func = raw_nodevalue
#     #raw_nodevalue_func = random_nodevalue
#     visualize_nodevalues(maze, pid, parameters, 2, 'probability weighted utility', raw_nodevalue_func, axs[3])

#     # parameters = [(tau,) for tau in TAUS]
#     # raw_nodevalue_func = raw_nodevalue_h_cells
#     # visualize_nodevalues(maze, pid, parameters, 0, 'cells', raw_nodevalue_func, axs[4])

#     # parameters = [(tau,) for tau in TAUS]
#     # raw_nodevalue_func = raw_nodevalue_h_steps
#     # visualize_nodevalues(maze, pid, parameters, 0, 'steps', raw_nodevalue_func, axs[5])

#     plt.show()




if __name__ == "__main__":

    import pprint
    pp = pprint.PrettyPrinter(compact=False)

    #tester map
    map_0 = ((3,3,3,0,0,3,3),
             (6,6,6,6,6,6,6),
             (6,3,3,0,0,3,3),
             (6,3,3,0,2,3,3),
             (6,3,3,3,3,3,3),
             (6,6,6,6,5,6,6),
             (3,3,3,3,3,3,0),)
    # map 1
    map_00 = ((3,3,3,3,3,3,3,3,3),
             (3,3,3,3,0,0,3,3,3),
             (3,6,6,6,6,6,6,6,3),
             (3,6,3,3,0,0,3,3,3),
             (3,6,3,3,0,0,3,3,3),
             (3,6,3,3,3,3,3,3,3),
             (3,6,6,6,6,5,6,6,3),
             (3,3,3,3,3,3,3,0,3),
             (3,3,3,3,3,3,3,3,3),)

    path = {()}

    # map 1
    map_1 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
             (3, 3, 3, 3, 0, 3, 0, 3, 3),
             (3, 3, 3, 3, 0, 3, 0, 3, 3),
             (3, 5, 6, 6, 6, 6, 6, 6, 3),
             (3, 6, 3, 3, 3, 3, 3, 6, 3),
             (3, 6, 6, 6, 6, 6, 6, 6, 3),
             (3, 3, 0, 0, 3, 3, 3, 3, 3),
             (3, 3, 3, 3, 3, 3, 3, 3, 3),)

    # ncols, nrows = 13, 9

    # map 2
    # ncols, nrows = 13, 9
    # start = (5 ,4)

    # path = {(3 ,1), (3 ,2), (3 ,3), (3 ,4), (3 ,5), (3 ,6), (3 ,7), (3 ,8),
    #         (5 ,1), (5 ,2), (5 ,3), (5 ,4), (5 ,5), (5 ,6), (5 ,7), (5 ,8),
    #         (4 ,1), (4 ,8)}

    # black = {(6 ,2), (7 ,2),
    #          (6 ,7), (7 ,7),
    #          (2 ,4), (1 ,4), (1 ,5), (1 ,6),
    #          (4 ,9), (4 ,10), (4 ,11)}

    # map_2 = map_builder(nrows, ncols, black, path, start)
    # tree2 = map2tree(map_2)
    # print(len(tree2))
    # map 2 good

    # map 3
    # ncols, nrows = 12, 8
    # start = (1,3)

    # path = {(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10),
    #         (2,3), (3,3), (4,3), (5,3), (6,3)}

    # black = {(2,1), (3,1),
    #          (5,4), (5,5), (5,6), (6,4), (6,5), (6,6),
    #          (2,9), (2,10), (3,9), (3,10), (4,9), (4,10), (5,9), (5,10), (6,9)}

    # map_3 = map_builder(nrows, ncols, black, path, start)

    # print(len(map2tree(map_3)))
    # map 3 works 

    # map 4
    # ncols, nrows = 12, 14
    # start = (7,3)

    # path = {(6,3), (5,3), (4,3), (3,3), (3,4), (3,5), (3,6), (3,6), (3,7), (3,8), (3,9), (3,10),
    #         (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7), (7,8), (7,9), (7,10),
    #         (8,3), (9,3), (10,3), (11,3)}

    # black = {(1,7), (1,8), (1,9), (1,10), (2,7), (2,8), (2,9), (2,10), (4,7), (4,8), (4,9), (4,10), (5,7), (5,8), (5,9), (5,10),
    #          (8,1), (9,1),
    #          (11,4), (11,5), (11,6), (11,7),
    #          (8,9), (8,10), (9,9), (9,10), (10,9), (10,10), (11,9), (11,10)}

    # map_4 = map_builder(nrows, ncols, black, path, start)
    # print(len(map2tree(map_4))) # map 4 works

    # map 5
    # ncols, nrows = 13, 5
    # start = (1,8)

    # path = {(1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11)}

    # black = {(2,1), (2,2), (2,3), (3,1), (3,2), (3,3),
    #          (2,11), (3,11)}

    # map_5 = map_builder(nrows, ncols, black, path, start)
    # print(len(map2tree(map_5)))
    # len only 3, but might still wprk

    # map 6
    # ncols, nrows = 14, 11
    # start = (6,1)

    # path = {(6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11), (6,12),
    #         (7,1), (8,1), (5,3), (4,3), (3,3), (6,5), (7,5), (8,5), (9,5),
    #         (6,7), (5,7), (4,7), (3,7), (2,7), (1,7)}

    # black = {(8,2), (8,3), (4,4), (4,5), (3,4), (3,5), (8,6), (8,7), (8,8), (8,9), (9,6), (9,7), (9,8), (9,9),
    #          (1,8), (1,9), (1,10), (1,11), (2,8), (2,9), (2,10), (2,11),
    #          (3,8), (3,9), (3,10), (3,11), (4,8), (4,9), (4,10), (4,11),}

    # map_6 = map_builder(nrows, ncols, black, path, start)
    # print(len(map2tree(map_6)))
    # map 6 works 

    # map 7
    # ncols, nrows = 14, 11
    # start = (6,1)

    # path = {(6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11), (6,12),
    #         (7,1), (8,1), (5,3), (4,3), (3,3), (6,5), (7,5), (8,5), (9,5),
    #         (6,7), (5,7), (4,7), (3,7), (2,7), (1,7)}

    # black = {(8,2), (8,3), (4,4), (4,5), (3,4), (3,5), (8,6), (8,7), (8,8), (9,6), (9,7), (9,8),
    #          (3,8), (3,9), (3,10), (3,11), (4,8), (4,9), (4,10), (4,11),}

    # map_7 = map_builder(nrows, ncols, black, path, start)
    # print(len(map2tree(map_7))) # map 7 works

    # map 8
    ncols, nrows = 13, 10
    start = (6,1)

    path = {(8,1), (7,1), (6,1), (5,1), (4,1), (3,1), (2,1),
            (5,2), (5,3), (5,4),
            (2,4), (3,4), (4,4), (6,4), (7,4), (8,4),
            (8,5), (8,6),
            (7,6), (6,6), (5,6),
            (5,7), (5,7), (5,8), (5,9), (5,10), (5,11),
            (2,5), (2,6), (2,7), (2,8), (2,9), (2,10)}

    black = {(8,2), (2,2), (2,3), (3,3),
             (1,6), (1,7), (1,8), (1,9), (1,10), (3,6), (3,7), (3,8), (3,9), (3,10),
             (6,8), (6,9), (6,10), (6,11),
             (7,8), (7,9), (7,10), (7,11),
             (8,8), (8,9), (8,10), (8,11),}

    map_8 = map_builder(nrows, ncols, black, path, start)
    print(len(map2tree(map_8))) # idk if this works

