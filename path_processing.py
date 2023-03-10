from mst_prototype import map2tree
import copy 

def get_decisions(raw_decisions, mazes):
    """
    Finds decision paths in terms of node ids instead of coordinates 
    Returns: dict in the form # {sid: {world: {'nodes': [], 'path': []}}}
    """
    tree_memo = {}
    decisions = copy.deepcopy(raw_decisions)

    for p in decisions:
            for maze in decisions[p]:
                actual_maze = mazes[maze] 
                if actual_maze not in tree_memo.keys(): # memoize
                    tree_memo[maze] = map2tree(actual_maze)
        
                # For person p: get path they take on maze
                path = decisions[p][maze]
                node_path = []
                for step in path['path']:
                    for x in tree_memo[maze].keys(): 
                            if step == tree_memo[maze][x]['pos']:
                                node_path.append(x)
                                break
            
                decisions[p][maze]['nodes'] = node_path

    return decisions

# TEST DATA TO DETERMINE IF CROSS_VALIDATE AND LOGLIKES ARE WORKING
test_decisions = {0:    {'Map0': {'path': [(6,5), (6,7), (2, 4)]}, 
                        'Map1': {'path': [(3,1), (3,4), (3,6)]},
                        'Map2': {'path': [(3,4), (3,7), (4,8)]}, 
                        'Map3': {'path': [(6,3), (6,1), (2,3), (6,9), (10,7)]},
                        'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (2,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},

                  1:    {'Map0': {'path': [(6,5), (6,7), (2, 4)]}, 
                        'Map1': {'path': [(3,1), (3,4), (3,6)]},
                        'Map2': {'path': [(3,4), (3,7), (4,8)]}, 
                        'Map3': {'path': [(6,3), (6,1), (2,3), (6,9), (10,7)]},
                        'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (2,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}}
                    }

#ACTUAL HUMAN DATA 
human_decisions = { 0: {'Map0': {'path': [(6,5), (6,7), (2, 4)]}, 
                        'Map1': {'path': [(3,1), (3,4), (3,6)]},
                        'Map2': {'path': [(3,4), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (6,1), (2,3), (6,9), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (2,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},
                    
                    1: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (5,2), (3,6)]},
                        'Map2': {'path': [(3,4), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},

                    2: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (5,2), (3,6)]},
                        'Map2': {'path': [(3,4), (3,2), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (2,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,7), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},

                    3: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (3,4), (3,6)]},
                        'Map2': {'path': [(3,4), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (2,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},
                    
                    4: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (3,4), (3,6)]},
                        'Map2': {'path': [(3,4), (3,2), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (6,1), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},

                    5: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (5,2), (3,6)]},
                        'Map2': {'path': [(3,4), (3,2), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (6,1), (2,3), (6,9), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (2,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},

                    6: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (3,4), (3,6)]},
                        'Map2': {'path': [(3,4), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},

                    7: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (5,2), (3,6)]},
                        'Map2': {'path': [(3,4), (3,2), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (6,1), (2,3), (6,9), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (2,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},

                    8: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (3,4), (3,6)]},
                        'Map2': {'path': [(3,4), (3,2), (5,4), (4,8)]},
                        'Map3': {'path': [(6,3), (6,1), (2,3), (6,9), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (2,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,7), (1,3)]},
                        'Map9': {'path': [(7,7), (6,8), (2,7), (4,7)]}},
                    
                    9: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (3,4), (3,6)]},
                        'Map2': {'path': [(3,4), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (6,1), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7, 2)]}},

                    10: {'Map0': {'path': [(6,5), (6,7), (2, 4)]},
                        'Map1': {'path': [(3,1), (3,4), (3,6)]},
                        'Map2': {'path': [(3,4), (3,2), (3,7), (4,8)]},
                        'Map3': {'path': [(6,3), (6,1), (10,7)]},
                        # 'Map4': {'path': [(4,1), (2,1)]},
                        'Map5': {'path': [(4,1), (2,1), (6,3), (6,2), (6,7)]},
                        'Map6': {'path': [(4,2), (4,4), (1,3), (1,5)]},
                        'Map7': {'path': [(5,4), (5,3), (2,2), (2,6)]},
                        'Map8': {'path': [(1,8), (1,3)]},
                        'Map9': {'path': [(7, 2)]}}
                }