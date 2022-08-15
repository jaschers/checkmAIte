import chess
import chess.engine
import numpy as np
from stockfish import Stockfish
import random
from tqdm import tqdm
from tensorflow.keras import datasets, models
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D, Dense, Flatten, ReLU, Add, BatchNormalization
import tensorflow.keras.utils as utils
from tensorflow.keras.callbacks import CSVLogger

# allocate stockfish engine and specify parameters
stockfish = Stockfish("/usr/local/Cellar/stockfish/15/bin/stockfish")
stockfish.set_depth(20)
stockfish.set_skill_level(20)

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
        board (chess.Board): chess board in FEN format

    Returns:
        list: (8, 8) list of the input board with int values
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
    # board_arr = board_arr.flatten()
    board_arr = board_arr.tolist()
    return(board_arr)

def board_score(board, time_limit = 0.01):
    """Evaluates the score of a board for player white based on stockfish.

    Args:
        board (chess.Board): chess board in FEN format
        time_limit (float, optional): maximum time allocated for the calculation. Defaults to 0.001 sec.

    Returns:
        int: stockfish score of the input board
    """
    engine = chess.engine.SimpleEngine.popen_uci("/usr/local/Cellar/stockfish/15/bin/stockfish")
    result = engine.analyse(board, chess.engine.Limit(time = time_limit))
    score = result['score'].white().score()
    engine.quit()
    return(score)

def boards_random(num_games):
    """Creates random boards by playing games with random moves

    Args:
        num_games (int): number of games being played

    Returns:
        list: (N,) list including all the randomly generated boards N while playing the games
    """
    boards_random_fen = []
    boards_random_int = []
    boards_random_score = []

    for _ in tqdm(range(num_games)):
        board = chess.Board()
        while True:
            all_moves = list(board.legal_moves)
            random_move = random.choice(all_moves)
            board.push(random_move)
            boards_random_fen.append(board.copy())
            boards_random_int.append(board_int(board.copy()))
            boards_random_score.append(board_score(board.copy()))
            if board.is_game_over():
                break
    return(boards_random_fen, boards_random_int, boards_random_score)

def ResBlock(z, kernelsizes, filters, increase_dim = False):
    # https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
    # https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
    # https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

    z_shortcut = z
    kernelsize_1, kernelsize_2 = kernelsizes
    filters_1, filters_2 = filters

    fz = Conv2D(filters_1, kernelsize_1)(z)
    # fz = BatchNormalization()(fz)
    fz = ReLU()(fz)

    fz = Conv2D(filters_1, kernelsize_2, padding = "same")(fz)
    # fz = BatchNormalization()(fz)
    fz = ReLU()(fz)
    
    fz = Conv2D(filters_2, kernelsize_1)(fz)
    # fz = BatchNormalization()(fz)

    if increase_dim == True:
        z_shortcut = Conv2D(filters_2, (1, 1))(z_shortcut)
        # z_shortcut = BatchNormalization()(z_shortcut)

    out = Add()([fz, z_shortcut])
    out = ReLU()(out)
    # out = MaxPooling2D(pool_size=(3, 3), strides = 1)(out)
    
    return out