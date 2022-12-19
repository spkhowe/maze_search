from mst_prototype import *
from cross_validation import model_preference, split_train_test_rand, split_train_test_kfold
from path_processing import *
import maps 
#import maps2
import path_processing
import models
import simulation
from loglikes import combine_all_trees, combine_path_node_values

# GET MAZES AND HUMAN DATA 
maps_ = maps.MAPS 
raw_decisisons_ = path_processing.human_decisions #CHANGE FOR REAL HUMAN DATA
decisions_ = path_processing.get_decisions(raw_decisisons_, maps_)

# GET TREE
all_trees = combine_all_trees(maps_)

# MODEL PREFERENCES AND MEDIAN PARAMETER VALUES
model_preferences, median_params = model_preference(decisions_, all_trees, maps_) # returns model: number of subjects that match this model 
print("model preferences:", model_preferences)
print("median params:", median_params)