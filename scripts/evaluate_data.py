import numpy as np
from utilities import *
import pandas as pd
import time
from chessboard import display
import matplotlib.pyplot as plt

# load data
num_runs = 1
table = pd.DataFrame()
for run in range(num_runs):
    run = 2 #2
    print(f"Loading data run {run}...")
    start = time.time()
    table_run = pd.read_hdf(f"data/data{run}.h5", key = "table")
    middle = time.time()
    print(f"Data run {run} loaded in {np.round(middle-start)} sec...")
    frame = [table, table_run]
    table = pd.concat(frame)
    end = time.time()
    print(f"Tables combined in {np.round(end-middle)} sec...")

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

# os.makedirs("evaluation/data/", exist_ok = True)

# while True:
#     pass

scores = table["score"].values.tolist()
print(scores)
print(np.nanmin(scores))
print(np.nanmax(scores))

plt.figure()
plt.hist(scores, bins = 50)
# plt.show()

# argmin = table["score"].idxmin()
# min = table["score"].iloc[argmin]
# print(min)
# board = table["boards (FEN)"].iloc[argmin]

# display.start(board.fen())

# while True:
#     pass