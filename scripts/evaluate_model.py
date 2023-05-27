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
parser.add_argument("-na", "--name", type = str, required = True, metavar = "-", help = "Name of the experiment that is going to be evaluated")

args = parser.parse_args()
##########################################################################################

max_score = 15000

# load data
print("Loading data...")
table_pred_val = pd.read_hdf(f"prediction/{args.name}/prediction_val_{args.name}.h5", key = "table")
table_examples = pd.read_hdf(f"prediction/{args.name}/examples_{args.name}.h5", key = "table")

table_pred_val["predicted score"] = table_pred_val["predicted score"] * 2 * max_score
table_pred_val["predicted score"] = table_pred_val["predicted score"] - max_score

table_pred_val["true score"] = table_pred_val["true score"] * 2 * max_score
table_pred_val["true score"] = table_pred_val["true score"] - max_score

print(np.shape(table_examples["board3d"][0]))

print(table_pred_val)
print("Number of check boards:", len(np.where(table_pred_val["true check"] == 1)[0]))
print("Number of checkmate boards:", len(np.where(table_pred_val["true checkmate"] == 1)[0]))
print("Number of stalemate boards:", len(np.where(table_pred_val["true stalemate"] == 1)[0]))

prediction_val = table_pred_val[["predicted score", "predicted check", "predicted checkmate", "predicted stalemate"]]
true_val = table_pred_val[["true score", "true check", "true checkmate", "true stalemate"]]

history = pd.read_csv(f"history/history_{args.name}.csv")

os.makedirs(f"evaluation/{args.name}/", exist_ok = True)
os.makedirs(f"evaluation/{args.name}/examples", exist_ok = True)

plot_history(history, args.name)

plot_2d_scattering(prediction_val["predicted score"], true_val["true score"], args.name)

plot_hist_difference_total(prediction_val["predicted score"], true_val["true score"], "score", args.name)

plot_hist_difference_binned(prediction_val["predicted score"], true_val["true score"], args.name)

plot_hist_difference_total(prediction_val["predicted check"], true_val["true check"], "check", args.name)

plot_hist_difference_total(prediction_val["predicted checkmate"], true_val["true checkmate"], "checkmate", args.name)

plot_hist_difference_total(prediction_val["predicted stalemate"], true_val["true stalemate"], "stalemate", args.name)

plot_hist(table_pred_val, "seventyfive moves", args.name)

save_examples(table_examples, args.name)