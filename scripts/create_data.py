import time
from utilities import boards_random
import pandas as pd
import os
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

# to be added:
# set stockfish depth
# set stockfish skill level
# set stockfish time limit
number_runs = 10 # 60
for run in range(number_runs):
    run = run + 0
    print(f"Processing run {run}...")
    # create random chess boards in "chess" and integer format
    start = time.time()
    boards_random_int, boards_random_score = boards_random(num_boards = 10000) #10000

    df1 = pd.DataFrame({"board3d (int)": boards_random_int})
    df2 = pd.DataFrame({"score": boards_random_score})

    table = pd.concat([df1, df2], axis = 1)

    print(table)
    # print(np.array(table["board3d (int)"][0]))

    os.makedirs("data/3d/digital_secrets/", exist_ok = True)

    table.to_hdf(f"data/3d/digital_secrets/data{run}.h5", key = "table")

    end = time.time()
    print(f"Processing time for run {run}:", np.round((end - start) / 60 / 60, 2), "hours")