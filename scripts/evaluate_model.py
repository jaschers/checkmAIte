import numpy as np
import pandas as pd
import os
import argparse
from utilities import *

######################################## argparse setup ########################################
script_descr="""
Evaluates the neural network
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-na", "--name", type = str, required = False, metavar = "-", help = "Name of the experiment that is going to be evaluated")

args = parser.parse_args()
##########################################################################################


# load data
print("Loading data...")
table_pred_val = pd.read_hdf(f"prediction/{args.name}/prediction_val_{args.name}.h5", key = "table")
table_examples = pd.read_hdf(f"prediction/{args.name}/examples_{args.name}.h5", key = "table")

prediction_val = table_pred_val["prediction"]
true_score_val = table_pred_val["true score"]

history = pd.read_csv(f"history/history_{args.name}.csv")

os.makedirs(f"evaluation/{args.name}/", exist_ok = True)
os.makedirs(f"evaluation/{args.name}/examples", exist_ok = True)

plot_history(history, args.name)

plot_2d_scattering(prediction_val, true_score_val, args.name)

plot_hist_difference_total(prediction_val, true_score_val, args.name)

plot_hist_difference_binned(prediction_val, true_score_val, args.name)

save_examples(table_examples, args.name)

# save_examples_gradcam(table_examples, args.name)