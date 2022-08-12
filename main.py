import chess
import chess.engine
import numpy as np
import stockfish
import os
print(stockfish.__file__)

# define starting chess board 
board = chess.Board()

print(type(board))

def board_int(board):
    """converts chess board into 2D array with
    1: pawn
    2: knight
    3: bishop
    4: rook
    5: queen
    6: king
    positive numbers: white
    negative numbers: black

    Args:
        board ('chess.Board')

    Returns:
        numpy.ndarray: (8, 8) array of the input board  
    """
    # initialise board array
    board_arr = np.zeros((8, 8), dtype = int)

    # for loop over all piece types (pawn, knight, ...)
    for piece in chess.PIECE_TYPES:
        # for loop over all squares of white pieces
        for square in board.pieces(piece, chess.WHITE):
            # get indices of the individual piece
            board_index = np.unravel_index(square, (8, 8))
            # fill array at board_index with piece value 
            board_arr[board_index[0]][board_index[1]] = piece
        # for loop over all squares of black pieces
        for square in board.pieces(piece, chess.BLACK):
            # get indices of the individual piece
            board_index = np.unravel_index(square, (8, 8))
            # fill array at board_index with negative piece value 
            board_arr[board_index[0]][board_index[1]] = -piece
    return board_arr

print(board_int(board))


# this function will create our f(x) (score)
def stockfish(board, depth):
    path = "/Users/Jann/opt/anaconda3/envs/chessai/lib/python3.8/site-packages/stockfish/"
    # assert os.path.isfile(path)
    with chess.engine.SimpleEngine.popen_uci(path) as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result["score"].white().score()
        return score

print(stockfish(board, 100))