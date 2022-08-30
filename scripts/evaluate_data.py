import numpy as np
import pandas as pd
import time
# from chessboard import display
import matplotlib.pyplot as plt
import os
import sys
np.set_printoptions(threshold=sys.maxsize)

# load data
num_runs = 35
table = pd.DataFrame()
for run in range(num_runs):
    # run = 2 #2
    print(f"Loading data run {run}...")
    start = time.time()
    table_run = pd.read_hdf(f"data/data{run}.h5", key = "table")
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

os.makedirs("evaluation/data/", exist_ok = True)

# while True:
#     pass

scores = table["score"].values.tolist()

plt.figure()
plt.hist(scores, bins = 50)
plt.xlabel("Score")
plt.ylabel("Number boards")
plt.tight_layout()
plt.savefig("evaluation/data/score_distribution.pdf")
# plt.show()

# argmin = table["score"].idxmin()
# min = table["score"].iloc[argmin]
# print(min)
# board = table["boards (FEN)"].iloc[argmin]

# display.start(board.fen())

# while True:
#     pass