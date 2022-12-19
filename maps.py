import matplotlib.pyplot as plt
from simulation import visualize_maze, visualize_path, best_path
from mst_prototype import map2tree, map_builder
from mst_prototype import raw_nodevalue, calculate_random_utility, calculate_expected_utility,calculate_discounted_utility, calculate_pwu, calculate_combined_utility, calculate_novel_utility
import copy
import pprint

pp = pprint.PrettyPrinter(compact=False, width=90)


utility_functions = [calculate_random_utility,
                     calculate_expected_utility,
                     calculate_discounted_utility,
                     calculate_pwu,
                     calculate_combined_utility,
                     calculate_novel_utility]

Map0 = ((3,3,3,3,3,3,3,3,3),
        (3,3,3,3,0,0,3,3,3),
        (3,6,6,6,6,6,6,6,3),
        (3,6,3,3,0,0,3,3,3),
        (3,6,3,3,0,0,3,3,3),
        (3,6,3,3,3,3,3,3,3),
        (3,6,6,6,6,5,6,6,3),
        (3,3,3,3,3,3,3,0,3),
        (3,3,3,3,3,3,3,3,3),)

Map1 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 0, 3, 0, 3, 3),
        (3, 3, 3, 3, 0, 3, 0, 3, 3),
        (3, 5, 6, 6, 6, 6, 6, 6, 3),
        (3, 6, 3, 3, 3, 3, 3, 6, 3),
        (3, 6, 6, 6, 6, 6, 6, 6, 3),
        (3, 3, 0, 0, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3),)

Map2 = ((3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 3, 0, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3),
        (3, 3, 0, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3),
        (3, 6, 6, 6, 5, 6, 6, 6, 6, 3, 3, 3, 3),
        (3, 6, 3, 3, 3, 3, 3, 3, 6, 0, 0, 0, 3),
        (3, 6, 6, 6, 6, 6, 6, 6, 6, 3, 3, 3, 3),
        (3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3))



Map3 = ((3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 6, 0, 0, 0, 0, 3, 0, 0, 3),
        (3, 3, 3, 6, 3, 3, 3, 3, 3, 0, 0, 3),
        (3, 0, 3, 6, 3, 3, 3, 3, 3, 0, 0, 3),
        (3, 0, 3, 6, 3, 3, 3, 3, 3, 0, 0, 3),
        (3, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 3),
        (3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 6, 3, 3, 3, 0, 0, 0, 0, 3),
        (3, 3, 3, 6, 3, 3, 3, 0, 0, 0, 0, 3),
        (3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 3),
        (3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 3),
        (3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3))


Map4 = ((3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 6, 0, 0, 0, 0, 3, 3, 3, 3),
        (3, 6, 0, 0, 3, 6, 0, 0, 0, 0, 3, 3, 3, 3),
        (3, 6, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3),
        (3, 3, 3, 6, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 6, 0, 0, 3, 6, 0, 0, 0, 0, 3, 3),
        (3, 3, 3, 6, 0, 0, 3, 6, 0, 0, 0, 0, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 6, 0, 0, 0, 0, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 6, 0, 0, 0, 0, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3))

Map5 = ((3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 6, 0, 0, 0, 3, 3, 3, 3, 3),
        (3, 6, 0, 0, 3, 6, 0, 0, 0, 3, 3, 3, 3, 3),
        (3, 6, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3),
        (3, 3, 3, 6, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 6, 0, 0, 3, 6, 0, 0, 0, 0, 3, 3),
        (3, 3, 3, 6, 0, 0, 3, 6, 0, 0, 0, 0, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3))

Map6 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 0, 0, 6, 6, 6, 0, 0, 3),
        (3, 0, 0, 3, 6, 3, 0, 0, 3),
        (3, 3, 3, 3, 6, 3, 3, 3, 3),
        (3, 3, 5, 6, 6, 0, 3, 3, 3),
        (3, 3, 3, 3, 3, 0, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3))


Map7 = ((3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 0, 3, 3, 3, 3, 3, 3, 3),
        (3, 0, 6, 6, 6, 6, 6, 6, 3),
        (3, 3, 3, 6, 3, 3, 0, 0, 3),
        (3, 0, 0, 6, 3, 3, 3, 3, 3),
        (3, 0, 0, 6, 5, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3))

Map8 = ((3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 0, 0, 6, 6, 6, 6, 6, 5, 3),
        (3, 0, 0, 3, 3, 3, 3, 6, 3, 3),
        (3, 3, 3, 6, 6, 6, 6, 6, 3, 3),
        (3, 3, 3, 6, 3, 3, 0, 0, 3, 3),
        (3, 0, 0, 6, 3, 3, 3, 3, 3, 3),
        (3, 0, 0, 6, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3))


Map9 = ((3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        (3, 0, 0, 6, 6, 6, 6, 6, 3, 3, 3, 3),
        (3, 0, 0, 3, 3, 3, 3, 6, 3, 3, 3, 3),
        (3, 3, 3, 6, 6, 6, 6, 6, 3, 3, 3, 3),
        (3, 3, 3, 6, 3, 3, 3, 0, 3, 3, 3, 3),
        (3, 0, 3, 6, 3, 3, 3, 3, 6, 0, 0, 3),
        (3, 0, 6, 6, 6, 6, 6, 5, 6, 3, 3, 3),
        (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3))

MAPS = {
    "Map0": Map0,
    "Map1": Map1,
    "Map2": Map2,
    "Map3": Map3,
#     "Map4": Map4,
    "Map5": Map5,
    "Map6": Map6,
    "Map7": Map7,
    "Map8": Map8,
    "Map9": Map9
}

# utility_functions_small = [calculate_random_utility,
#                           calculate_expected_utility]

# for key,map_ in MAPS.items():
#         for utility_fxn in utility_functions:
#                 #_, ax = plt.subplots(1, 1)
#                 #visualize_maze(map_, ax)
#                 tree = map2tree(map_)

#                 generate_path = best_path(map_, utility_fxn, (1,1,1,1)) #params: (gamma,beta,k,tau)
#                 # visualize_path(map_,generate_path,ax)
#                 print("Utility function: ", str(utility_fxn))
#                 print("BEST PATH: ", generate_path)
#                 #plt.show()


   