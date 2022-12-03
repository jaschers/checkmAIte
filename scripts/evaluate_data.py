import numpy as np
import pandas as pd
import time
# from chessboard import display
import matplotlib.pyplot as plt
import os
import sys
import argparse

np.set_printoptions(threshold=sys.maxsize)

######################################## argparse setup ########################################
script_descr="""
Evaluates the residual neural network
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-na", "--name", type = str, required = True, metavar = "-", help = "Name of this particular experiment")

args = parser.parse_args()
##########################################################################################

# load data
num_runs = 30
table = pd.DataFrame()
for run in range(num_runs):
    # run = 2 #2
    print(f"Loading data run {run}...")
    start = time.time()
    table_run = pd.read_hdf(f"data/3d/{args.name}/data{run}.h5", key = "table")
    print(f"Number of boards in run {run}:", len(table_run))
    middle = time.time()
    print(f"Data run {run} loaded in {np.round(middle-start)} sec...")
    frame = [table, table_run]
    table = pd.concat(frame)
    end = time.time()
    print(f"Tables combined in {np.round(end-middle)} sec...")

table = table.reset_index(drop = True)
print(table)
# duplicates = table.duplicated(subset=["board (int)"])
# print(duplicates)
# table = table.dropna().reset_index()
# print(table)
# table_score_none = table.loc[table["score"] == np.nan]
# table_score_none = table[pd.notna(table["score"]) == False].reset_index()
# table_score_unique = table["score"].unique()
# print(table_score_none)
# print(table_score_unique)
# board_nan = table_score_none["boards (FEN)"].iloc[1]

# print(board_nan.fen())
# display.start(board_nan.fen())

os.makedirs(f"evaluation/data/{args.name}", exist_ok = True)

# while True:
#     pass

scores = table["score"].values.tolist()
unique_scores = np.unique(scores)
unique_scores = np.sort(unique_scores)

plt.figure()
plt.hist(scores, bins = 50)
plt.xlabel("Score")
plt.ylabel("Number boards")
plt.tight_layout()
plt.savefig(f"evaluation/data/{args.name}/score_distribution.pdf")
# plt.show()

print("Unique scores:")
print(unique_scores)

# argmin = table["score"].idxmin()
# min = table["score"].iloc[argmin]
# print(min)
# board = table["boards (FEN)"].iloc[argmin]

# display.start(board.fen())

# while True:
#     pass