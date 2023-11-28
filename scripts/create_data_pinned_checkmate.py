import pandas as pd
import os
import numpy as np
import glob
import chess
import random
from utilities import *
from tqdm import tqdm

# function to generate random pinned_checkmate position
def boards_pinned_checkmate(num_boards):
    """
    Creates random boards by playing games with random moves

    Args:
        num_boards (int): number of boards being created

    Returns:
        list: (N,) list including all the randomly generated boards N while playing the games
    """
    boards_pinned_checkmate_int = []
    boards_pinned_checkmate_parameter = []
    boards_pinned_checkmate_score = []

    for _ in tqdm(range(num_boards)):
        board = chess.Board() #"rnbqk1nr/pppp1ppp/8/4p3/1b1P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 1 3" #"2r3nr/1p2pp2/3k3b/p6p/P1BPp1pP/1qK2P2/bB4PN/RN3QR1 w - - 0 1"
        board_pinned = get_board_pinned(board.copy())
        board_pinned_white, board_pinned_black = board_pinned[0], board_pinned[2]
        pinned_white = 1 in board_pinned_white
        pinned_black = 1 in board_pinned_black
        count = 0
        while not (board.is_checkmate() and ((pinned_white and board.turn == chess.WHITE) or (pinned_black and board.turn == chess.BLACK))):
            # if count % 5000 == 0:
            #     print(count)
            if not board.is_game_over():
                move = random.choice(list(board.legal_moves))
                board.push(move)
                board_pinned = get_board_pinned(board.copy())
                board_pinned_white, board_pinned_black = board_pinned[0], board_pinned[2]
                pinned_white = 1 in board_pinned_white
                pinned_black = 1 in board_pinned_black
                # if pinned:
                #     print("pinned")
                count += 1
            if (board.is_checkmate() and ((pinned_white and board.turn == chess.WHITE) or (pinned_black and board.turn == chess.BLACK))):
                boards_pinned_checkmate_int.append(get_board_total(board.copy()))
                boards_pinned_checkmate_parameter.append(get_board_parameters(board.copy()))
                boards_pinned_checkmate_score.append(np.int16(board_score(board.copy())))
            if not (board.is_checkmate() and ((pinned_white and board.turn == chess.WHITE) or (pinned_black and board.turn == chess.BLACK))) and board.is_game_over():
                board = chess.Board()

    boards_pinned_checkmate_parameter = np.array(boards_pinned_checkmate_parameter)

    return(boards_pinned_checkmate_int, boards_pinned_checkmate_parameter, boards_pinned_checkmate_score)

dir_pinned_checkmate = "data/30_8_8_pinned_checkmate/"
os.makedirs(dir_pinned_checkmate, exist_ok = True)

number_runs = 100
for run in range(number_runs):
    run = run + 10
    boards_pinned_checkmate_int, boards_pinned_checkmate_parameter, boards_pinned_checkmate_score = boards_pinned_checkmate(num_boards = 100)

    df1 = pd.DataFrame({"board3d": boards_pinned_checkmate_int})
    df2 = pd.DataFrame({"player move": boards_pinned_checkmate_parameter[:,0]})
    df3 = pd.DataFrame({"halfmove clock": boards_pinned_checkmate_parameter[:,1]})
    df4 = pd.DataFrame({"fullmove number": boards_pinned_checkmate_parameter[:,2]})
    df5 = pd.DataFrame({"check": boards_pinned_checkmate_parameter[:,3]})
    df6 = pd.DataFrame({"checkmate": boards_pinned_checkmate_parameter[:,4]})
    df7 = pd.DataFrame({"stalemate": boards_pinned_checkmate_parameter[:,5]})
    df8 = pd.DataFrame({"insufficient material white": boards_pinned_checkmate_parameter[:,6]})
    df9 = pd.DataFrame({"insufficient material black": boards_pinned_checkmate_parameter[:,7]})
    df10 = pd.DataFrame({"seventyfive moves": boards_pinned_checkmate_parameter[:,8]})
    df11 = pd.DataFrame({"fivefold repetition": boards_pinned_checkmate_parameter[:,9]})
    df12 = pd.DataFrame({"castling right king side white": boards_pinned_checkmate_parameter[:,10]})
    df13 = pd.DataFrame({"castling right queen side white": boards_pinned_checkmate_parameter[:,11]})
    df14 = pd.DataFrame({"castling right king side black": boards_pinned_checkmate_parameter[:,12]})
    df15 = pd.DataFrame({"castling right queen side black": boards_pinned_checkmate_parameter[:,13]})
    df16 = pd.DataFrame({"score": boards_pinned_checkmate_score})

    table = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16], axis = 1)

    print(table)

    table.to_hdf(dir_pinned_checkmate + f"data_pinned_checkmate{run}.h5", key = "table")
