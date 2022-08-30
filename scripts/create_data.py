import time
from utilities import boards_random
import pandas as pd
import os
import numpy as np

# to be added:
# set stockfish depth
# set stockfish skill level
# set stockfish time limit
number_runs = 35
for run in range(number_runs):
    # run = run + 36
    print(f"Processing run {run}...")
    # create random chess boards in "chess" and integer format
    start = time.time()
    boards_random_fen, boards_random_int, boards_random_score = boards_random(num_boards = 20000)

    df1 = pd.DataFrame({"boards (FEN)": boards_random_fen})
    df2 = pd.DataFrame({"boards (int)": boards_random_int})
    df3 = pd.DataFrame({"score": boards_random_score})

    table = pd.concat([df1, df2, df3], axis = 1)

    os.makedirs("data/", exist_ok = True)

    table.to_hdf(f"data/data{run}.h5", key = "table")

    end = time.time()
    print(f"Processing time for run {run}:", np.round((end - start) / 60 / 60, 2), "hours")