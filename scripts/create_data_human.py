import numpy as np
import sys
import chess
from utilities import get_board_total, get_board_parameters, board_score
import pandas as pd
from tqdm import tqdm 
import os
import glob
import argparse

# data downloaded from https://rebel13.nl/download/data.html

np.set_printoptions(threshold=sys.maxsize)

######################################## argparse setup ########################################
script_descr="""
Creates training data for the neural network based on random moves. The data consists of 3D chess boards, secondary information such as the castling rights, and the stockfish evaluation of the board. The data is stored in HDF5 format in the data/ directory.
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-fid", "--file_id", type = int, required = True, metavar = "-", help = "File with file ID <fid> from which you would like to extract the data, sf-eval-<fid>.epd")
parser.add_argument("-nb", "--number_boards", type = int, metavar = "-", help = "Number of boards that are going to be extracted for each run. Default: 10000", default = 10000)
parser.add_argument("-sr", "--starting_run", type = int, metavar = "-", help = "Starting run id. If, e.g. 10 runs have already been extracted, use -sr 10 to extract avoid overwriting the old data. Default: 0", default = 0)

args = parser.parse_args()
##########################################################################################

filename = f"data/human_games/sf-eval-{args.file_id}.epd"


filename_subid = 0
filename_id = args.file_id

file = open(filename, "r")
num_lines = sum(1 for _ in open(filename))

boards_random_int = []
boards_random_parameter = []
boards_random_score = []

count = 0
count_total = 0

for line in tqdm(file):
    if int(filename_subid) >= args.starting_run:
        board_fen, _ = line.split("; ")
        board = chess.Board(board_fen)

        boards_random_int.append(get_board_total(board.copy()))
        boards_random_parameter.append(get_board_parameters(board.copy()))
        boards_random_score.append(np.int16(board_score(board.copy())))

    count += 1
    count_total += 1
    
    if count == args.number_boards:
        if int(filename_subid) >= args.starting_run:
            boards_random_parameter = np.array(boards_random_parameter)

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

            os.makedirs("data/30_8_8_depth0_ms15000_human/", exist_ok = True)

            table.to_hdf(f"data/30_8_8_depth0_ms15000_human/data{filename_id}-{filename_subid}.h5", key = "table")
            print(f"Saved table to data/30_8_8_depth0_ms15000_human/data{filename_id}-{filename_subid}.h5")
        else:
            print(f"Skip extracting file data/30_8_8_depth0_ms15000_human/data{filename_id}-{filename_subid}.h5")

        count = 0
        filename_subid += 1

        boards_random_int = []
        boards_random_parameter = []
        boards_random_score = []