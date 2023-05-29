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
number_runs = 500 # 60
for run in range(number_runs):
    run = run + 46
    print(f"Processing run {run}...")
    # create random chess boards in "chess" and integer format
    boards_random_int, boards_random_parameter, boards_random_score = boards_random(num_boards = 10000) #10000

    print(type(boards_random_int))

    df1 = pd.DataFrame({"board3d": boards_random_int})
    df2 = pd.DataFrame({"player move": boards_random_parameter[:,0]})
    df3 = pd.DataFrame({"halfmove clock": boards_random_parameter[:,1]})
    df4 = pd.DataFrame({"fullmove number": boards_random_parameter[:,2]})
    df5 = pd.DataFrame({"check": boards_random_parameter[:,3]})
    df6 = pd.DataFrame({"checkmate": boards_random_parameter[:,4]})
    df7 = pd.DataFrame({"stalemate": boards_random_parameter[:,5]})
    df8 = pd.DataFrame({"insufficient material white": boards_random_parameter[:,6]})
    df9 = pd.DataFrame({"insufficient material black": boards_random_parameter[:,7]})
    df10 = pd.DataFrame({"seventyfive moves": boards_random_parameter[:,8]})
    df11 = pd.DataFrame({"fivefold repetition": boards_random_parameter[:,9]})
    df12 = pd.DataFrame({"castling right king side white": boards_random_parameter[:,10]})
    df13 = pd.DataFrame({"castling right queen side white": boards_random_parameter[:,11]})
    df14 = pd.DataFrame({"castling right king side black": boards_random_parameter[:,12]})
    df15 = pd.DataFrame({"castling right queen side black": boards_random_parameter[:,13]})
    df16 = pd.DataFrame({"score": boards_random_score})

    table = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16], axis = 1)

    print(table)
    print(np.shape(table["board3d"][0]))
    # print(np.array(table["board3d (int)"][0]))

    os.makedirs("data/3d/32_8_8_depth0_mm100_ms15000/", exist_ok = True)

    table.to_hdf(f"data/3d/32_8_8_depth0_mm100_ms15000/data{run}.h5", key = "table")

    end = time.time()
