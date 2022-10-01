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
number_runs = 20 # 60
for run in range(number_runs):
    run = run + 0
    print(f"Processing run {run}...")
    # create random chess boards in "chess" and integer format
    boards_random_int, player_move, halfmove_clock, fullmove_number, boards_random_score = boards_random(num_boards = 5000) #10000

    df1 = pd.DataFrame({"board3d": boards_random_int})
    df2 = pd.DataFrame({"player move": player_move})
    df3 = pd.DataFrame({"halfmove clock": halfmove_clock})
    df4 = pd.DataFrame({"fullmove number": fullmove_number})
    df5 = pd.DataFrame({"score": boards_random_score})

    table = pd.concat([df1, df2, df3, df4, df5], axis = 1)

    print(table)
    # print(np.array(table["board3d (int)"][0]))

    os.makedirs("data/3d/24_8_8_depth20/", exist_ok = True)

    table.to_hdf(f"data/3d/24_8_8_depth20/data{run}.h5", key = "table")

    end = time.time()
