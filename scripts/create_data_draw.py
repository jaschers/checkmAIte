import pandas as pd
import os
import numpy as np
import glob

# draw by 'seventyfive moves', 'fivefold repetition'

parameters = ['seventyfive moves', 'fivefold repetition']
parameters_short_name = ['sfmoves', 'ffrep']

n_files_per_para = 3
# n_copies = len(parameters) * n_files_per_para

dir_human = "data/3d/32_8_8_depth0_ms15000_human/"
filenames = glob.glob(dir_human + "*")
filenames = np.sort(filenames)

dir_draw = "data/3d/32_8_8_draw/"
os.makedirs(dir_draw, exist_ok = True)

count = 0
for i in range(len(parameters)):
    for j in range(n_files_per_para):
        print(f"loading file {filenames[count]}")
        table = pd.read_hdf(filenames[count], key = "table")
        table[parameters[i]] = 1
        table["score"] = 0
        print("saving file", dir_draw + f"data_{parameters_short_name[i]}{j}.h5")
        table.to_hdf(dir_draw + f"data_{parameters_short_name[i]}{j}.h5", key = "table")
        count += 1
