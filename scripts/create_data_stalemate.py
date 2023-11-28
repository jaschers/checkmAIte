import pandas as pd
import os
import numpy as np
import glob
import chess
import random
from utilities import *
from tqdm import tqdm

# function to generate random stalemate position
def boards_stalemate(num_boards):
    """
    Creates random boards by playing games with random moves

    Args:
        num_boards (int): number of boards being created

    Returns:
        list: (N,) list including all the randomly generated boards N while playing the games
    """
    boards_stalemate_int = []
    boards_stalemate_parameter = []
    boards_stalemate_score = []

    for _ in tqdm(range(num_boards)):
        board = chess.Board()
        while not board.is_stalemate():
            if not board.is_game_over():
                move = random.choice(list(board.legal_moves))
                board.push(move)
            if board.is_stalemate():
                boards_stalemate_int.append(get_board_total(board.copy()))
                boards_stalemate_parameter.append(get_board_parameters(board.copy()))
                boards_stalemate_score.append(np.int16(board_score(board.copy())))
            elif board.is_game_over():
                board = chess.Board()

    boards_stalemate_parameter = np.array(boards_stalemate_parameter)

    return(boards_stalemate_int, boards_stalemate_parameter, boards_stalemate_score)

dir_stalemate = "data/30_8_8_draw/"
os.makedirs(dir_stalemate, exist_ok = True)

number_runs = 3
for run in range(number_runs):
    boards_stalemate_int, boards_stalemate_parameter, boards_stalemate_score = boards_stalemate(num_boards = 10000)

    df1 = pd.DataFrame({"board3d": boards_stalemate_int})
    df2 = pd.DataFrame({"player move": boards_stalemate_parameter[:,0]})
    df3 = pd.DataFrame({"halfmove clock": boards_stalemate_parameter[:,1]})
    df4 = pd.DataFrame({"fullmove number": boards_stalemate_parameter[:,2]})
    df5 = pd.DataFrame({"check": boards_stalemate_parameter[:,3]})
    df6 = pd.DataFrame({"checkmate": boards_stalemate_parameter[:,4]})
    df7 = pd.DataFrame({"stalemate": boards_stalemate_parameter[:,5]})
    df8 = pd.DataFrame({"insufficient material white": boards_stalemate_parameter[:,6]})
    df9 = pd.DataFrame({"insufficient material black": boards_stalemate_parameter[:,7]})
    df10 = pd.DataFrame({"seventyfive moves": boards_stalemate_parameter[:,8]})
    df11 = pd.DataFrame({"fivefold repetition": boards_stalemate_parameter[:,9]})
    df12 = pd.DataFrame({"castling right king side white": boards_stalemate_parameter[:,10]})
    df13 = pd.DataFrame({"castling right queen side white": boards_stalemate_parameter[:,11]})
    df14 = pd.DataFrame({"castling right king side black": boards_stalemate_parameter[:,12]})
    df15 = pd.DataFrame({"castling right queen side black": boards_stalemate_parameter[:,13]})
    df16 = pd.DataFrame({"score": boards_stalemate_score})

    table = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16], axis = 1)

    print(table)

    table.to_hdf(dir_stalemate + f"data_stalemate{run}.h5", key = "table")
