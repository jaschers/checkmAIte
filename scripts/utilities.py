import chess
import chess.engine
import chess.svg
import numpy as np
from stockfish import Stockfish
import random
from tqdm import tqdm
import os 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib import animation
import time
import pandas as pd
import sys
from keras import models
import tensorflow as tf


np.set_printoptions(threshold=sys.maxsize)

stockfish_path = os.environ.get("STOCKFISHPATH")

# # allocate stockfish engine and specify parameters
# stockfish = Stockfish(stockfish_path)
# stockfish.set_depth(20)
# stockfish.set_skill_level(20)

# get stockfish engine
stockfish_path = os.environ.get("STOCKFISHPATH")
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

best_move = None
counter = 0

def square_to_index(square):
    squares = np.linspace(0, 8*8 - 1, 8*8, dtype = int)
    squares_2d = np.reshape(squares, (8,8))
    squares_2d = np.flip(squares_2d, 0)
    index = np.where(squares_2d == square)
    return(index[0][0], index[1][0])

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

def board_3d_attack_int(board):
    """converts chess board into 3D (24, 8, 8) array with board[i] representing:
    0: all squares covered by white pawn
    1: all squares covered by white knight
    2: all squares covered by white bishop
    3: all squares covered by white rook
    4: all squares covered by white queen
    5: all squares covered by white king
    6: all squares covered by black pawn
    7: all squares covered by black knight
    8: all squares covered by black bishop
    9: all squares covered by black rook
    10: all squares covered by black queen
    11: all squares covered by black king
    12: all squares being attacked by white pawn
    13: all squares being attacked by white knight
    14: all squares being attacked by white bishop
    15: all squares being attacked by white rook
    16: all squares being attacked by white queen
    17: all squares being attacked by white king
    18: all squares being attacked by black pawn
    19: all squares being attacked by black knight
    20: all squares being attacked by black bishop
    21: all squares being attacked by black rook
    22: all squares being attacked by black queen
    23: all squares being attacked by black king

    Args:
        board (chess.Board): chess board

    Returns:
        list: (24, 8, 8) list of the input board with {1,0} int values
    """
    # initialise board array
    number_boards = 24 
    board_arr = np.zeros((number_boards, 8, 8), dtype = int)

    # for loop over all piece types (pawn, knight, ...)
    for piece in chess.PIECE_TYPES:
        # for loop over all squares of white pieces
        for square in board.pieces(piece, chess.WHITE):
            # get indices of the individual piece
            board_index = square_to_index(square)
            # fill array at board_index with piece value for each piece
            board_arr[piece - 1][board_index[0]][board_index[1]] = 1

        # for loop over all squares of black pieces
        for square in board.pieces(piece, chess.BLACK):
            # get indices of the individual piece
            board_index = square_to_index(square)
            # fill array at board_index with piece value for each piece
            board_arr[piece - 1 + 6][board_index[0]][board_index[1]] = 1


    # add attacks from each pice to an individual 8x8 subarray
    board.turn = chess.WHITE
    for valid_move in board.legal_moves:
        # get piece type that's making the move
        piece_type = board.piece_type_at(valid_move.from_square)
        # get square number that is being attacked
        square = valid_move.to_square
        # convert square number into index for (8,8) board
        board_index = square_to_index(square)
        # add +1 to the attacked square in the board_arr of corresponding piece type
        board_arr[piece_type - 1 + 12][board_index[0]][board_index[1]] += 1

    board.turn = chess.BLACK
    for valid_move in board.legal_moves:
        # get piece type that's making the move
        piece_type = board.piece_type_at(valid_move.from_square)
        # get square number that is being attacked
        square = valid_move.to_square
        # convert square number into index for (8,8) board
        board_index = square_to_index(square)
        # add +1 to the attacked square in the board_arr of corresponding piece type
        board_arr[piece_type - 1 + 18][board_index[0]][board_index[1]] += 1

    # board_arr = board_arr.flatten()
    board_arr = board_arr.tolist()
    return(board_arr)

# def board_3d_attack_int(board):
#     """converts chess board into 3D (24, 8, 8) array with board[i] representing:
#     0: all squares covered by white pawn
#     1: all squares covered by white knight
#     2: all squares covered by white bishop
#     3: all squares covered by white rook
#     4: all squares covered by white queen
#     5: all squares covered by white king
#     6: all squares covered by black pawn
#     7: all squares covered by black knight
#     8: all squares covered by black bishop
#     9: all squares covered by black rook
#     10: all squares covered by black queen
#     11: all squares covered by black king
#     12: all squares being attacked by white pawn
#     13: all squares being attacked by white knight
#     14: all squares being attacked by white bishop
#     15: all squares being attacked by white rook
#     16: all squares being attacked by white queen
#     17: all squares being attacked by white king
#     18: all squares being attacked by black pawn
#     19: all squares being attacked by black knight
#     20: all squares being attacked by black bishop
#     21: all squares being attacked by black rook
#     22: all squares being attacked by black queen
#     23: all squares being attacked by black king

#     Args:
#         board (chess.Board): chess board

#     Returns:
#         list: (24, 8, 8) list of the input board with {1,0} int values
#     """
#     # initialise board array
#     number_boards = 14 
#     board_arr = np.zeros((number_boards, 8, 8), dtype = int)

#     # for loop over all piece types (pawn, knight, ...)
#     for piece in chess.PIECE_TYPES:
#         # for loop over all squares of white pieces
#         for square in board.pieces(piece, chess.WHITE):
#             # get indices of the individual piece
#             board_index = square_to_index(square)
#             # fill array at board_index with piece value for each piece
#             board_arr[piece - 1][board_index[0]][board_index[1]] = 1

#         # for loop over all squares of black pieces
#         for square in board.pieces(piece, chess.BLACK):
#             # get indices of the individual piece
#             board_index = square_to_index(square)
#             # fill array at board_index with piece value for each piece
#             board_arr[piece - 1 + 6][board_index[0]][board_index[1]] = 1


#     # add attacks from each pice to an individual 8x8 subarray
#     board.turn = chess.WHITE
#     for valid_move in board.legal_moves:
#         # get piece type that's making the move
#         piece_type = board.piece_type_at(valid_move.from_square)
#         # get square number that is being attacked
#         square = valid_move.to_square
#         # convert square number into index for (8,8) board
#         board_index = square_to_index(square)
#         # add +1 to the attacked square in the board_arr of corresponding piece type
#         board_arr[12][board_index[0]][board_index[1]] = 1

#     board.turn = chess.BLACK
#     for valid_move in board.legal_moves:
#         # get piece type that's making the move
#         piece_type = board.piece_type_at(valid_move.from_square)
#         # get square number that is being attacked
#         square = valid_move.to_square
#         # convert square number into index for (8,8) board
#         board_index = square_to_index(square)
#         # add +1 to the attacked square in the board_arr of corresponding piece type
#         board_arr[13][board_index[0]][board_index[1]] = 1

#     # board_arr = board_arr.flatten()
#     board_arr = board_arr.tolist()
#     return(board_arr)

def board_score(board, depth = 0):
    """Evaluates the score of a board for player white based on stockfish.

    Args:
        board (chess.Board): chess board in FEN format
        time_limit (float, optional): maximum time allocated for the calculation. Defaults to 0.001 sec.

    Returns:
        int: stockfish score of the input board
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    result = engine.analyse(board.copy(), chess.engine.Limit(depth = depth))
    score = result["score"].white().score(mate_score = 10000)
    engine.quit()
    return(score)

def boards_random(num_boards):
    """Creates random boards by playing games with random moves

    Args:
        num_boards (int): number of baords being created

    Returns:
        list: (N,) list including all the randomly generated boards N while playing the games
    """
    boards_random_int = []
    player_move = []
    halfmove_clock = []
    fullmove_number = []
    boards_random_score = []

    for _ in tqdm(range(num_boards)):
        board = chess.Board()
        depth = random.randrange(0, 100) # max number of moves: 100
        for _ in range(depth):
            all_moves = list(board.legal_moves)
            random_move = random.choice(all_moves)
            board.push(random_move)
            if board.is_game_over():
                break

        boards_random_int.append(board_3d_attack_int(board.copy()))
        player_move.append(board.copy().turn)
        halfmove_clock.append(np.int16(board.copy().halfmove_clock))
        fullmove_number.append(np.int16(board.copy().fullmove_number))
        boards_random_score.append(np.int16(board_score(board.copy())))

    return(boards_random_int, player_move, halfmove_clock, fullmove_number, boards_random_score)

def ai_board_score_pred(board, model):
    board_3d_int = [board_3d_attack_int(board.copy())]
    board_3d_int = np.moveaxis(board_3d_int, 1, -1)
    parameters = np.array([[np.float32(board.copy().turn), np.float32(board.copy().halfmove_clock)]])
    prediction_score = model.predict([board_3d_int, parameters], verbose = 0)[0][0] 
    return(prediction_score)

def minimax(board, model, depth, alpha, beta, maximizing_player, verbose_minimax = False):
    # global counter
    # counter += 1
    # print(counter)
    if depth < 0 or type(depth) != int:
        raise ValueError("Depth needs to be int and greater than 0")

    if depth == 0 or board.is_game_over() == True:
        prediction = ai_board_score_pred(board.copy(), model)
        
        if verbose_minimax == True:
            print(board)
            print(f"Maximizing player == {maximizing_player}")
            print("prediction", prediction * 14863 - 7645)
            print("_____________")
        return(prediction)

    # maximizing_player == True -> AI's turn
    if maximizing_player == True:
        # print("maximizing_player == True", f", depth = {depth}")
        max_eval = - np.inf
        for valid_move in board.legal_moves:
            board.push(valid_move)
            eval = minimax(board.copy(), model, depth - 1, alpha, beta, False, verbose_minimax)
            board.pop()
            if eval > max_eval:
                max_eval = eval
            alpha = max(eval, alpha)
            if beta <= alpha:
                break
        # print("2", max_eval, best_move)
        if verbose_minimax == True:
            print(board)
            print(f"Maximizing player == {maximizing_player}")
            print("max_eval", max_eval * 14863 - 7645)
            print("_____________")
        return(max_eval)

    # maximizing_player == False -> player's turn
    else:
        # print("maximizing_player == False", f", depth = {depth}")
        min_eval = np.inf
        for valid_move in board.legal_moves:
            board.push(valid_move)
            eval = minimax(board.copy(), model, depth - 1, alpha, beta, True, verbose_minimax)
            board.pop()
            if eval < min_eval:
                min_eval = eval
            beta = min(eval, beta)
            if beta <= alpha:
                break
        #print("3", min_eval, best_move)
        if verbose_minimax == True:
            print(board)
            print(f"Maximizing player == {maximizing_player}")
            print("min_eval", min_eval * 14863 - 7645)
            print("_____________")
        return(min_eval)


def get_ai_move(board, model, depth, verbose_minimax):
    max_move = None
    max_eval = -np.inf

    for valid_move in board.legal_moves:
        board.push(valid_move)
        # maximizing_player == False -> player's move because AI's (potential) move was just pushed
        eval = minimax(board.copy(), model, depth = depth - 1, alpha = -np.inf, beta = np.inf, maximizing_player = False, verbose_minimax = verbose_minimax)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            max_move = valid_move
  
    return(max_move, max_eval)



# def minimax(board, model, depth, alpha, beta, maximizing_player, verbose_minimax = False):
#     global best_move
#     if depth == 0 or board.is_game_over() == True:
#         board_int_eval = [board_3d_attack_int(board)]
#         board_int_eval = np.moveaxis(board_int_eval, 1, -1)
#         parameters = np.array([[np.float32(board.turn), np.float32(board.halfmove_clock)]])
#         prediction = model.predict([board_int_eval, parameters], verbose = 0)[0][0]
#         # print("1", prediction, best_move)
#         if verbose_minimax == True:
#             print(board)
#             print("_____________")
#         return(prediction, best_move)

#     if maximizing_player == True:
#         # print("maximizing_player == True", f", depth = {depth}")
#         max_eval = - np.inf
#         for valid_move in board.legal_moves:
#             board.push(valid_move)
#             eval, best_move = minimax(board, model, depth - 1, alpha, beta, False, verbose_minimax)
#             board.pop()
#             if eval > max_eval:
#                 max_eval = eval
#                 best_move = valid_move
#             alpha = max(eval, alpha)
#             if beta <= alpha:
#                 break
#         # print("2", max_eval, best_move)
#         if verbose_minimax == True:
#             print(board)
#             print("_____________")
#         return(max_eval, best_move)

#     else:
#         # print("maximizing_player == False", f", depth = {depth}")
#         min_eval = np.inf
#         for valid_move in board.legal_moves:
#             board.push(valid_move)
#             eval, best_move = minimax(board, model, depth - 1, alpha, beta, True, verbose_minimax)
#             board.pop()
#             if eval < min_eval:
#                 min_eval = eval
#             beta = min(eval, beta)
#             if beta <= alpha:
#                 break
#         #print("3", min_eval, best_move)
#         if verbose_minimax == True:
#             print(board)
#             print("_____________")
#         return(min_eval, best_move)

def save_board_png(board, game_name, counter):
    """Saves the current board as png in games/{game_name}/board{counter}.png

    Args:
        board (chess.Board): chess board
        game_name (str): name of the current chess game
        counter (int): board move counter
    """
    boardsvg = chess.svg.board(board = board.copy())
    outputfile = open(f"games/{game_name}/board{counter}.svg", "w")
    outputfile.write(boardsvg)
    outputfile.close()
    os.system(f"convert -density 1200 -resize 780x780 games/{game_name}/board{counter}.svg games/{game_name}/board{counter}.png")
    os.system(f"rm games/{game_name}/board{counter}.svg")

def save_baord_gif(boards_png, game_name):
    """Loads png images of a chess game and converts it into a gif. The png images are deleted afterwards

    Args:
        boards_png (list): list of PIL.PngImagePlugin.PngImageFile images
        game_name (str): name of the current chess game
    """
    fig = plt.figure(frameon=False)
    ax  = fig.add_subplot(111)
    ims = []
    for board_png in boards_png:
        ax.axis('off')
        im = ax.imshow(board_png)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval = 1000)
    ani.save(f"games/{game_name}/baord.gif")

    os.system(f"rm games/{game_name}/*.png")

def get_valid_moves(board):
    """returns the valid moves of a board

    Args:
        board (chess.Board): chess board
    
    Returns:
        valid_moves (list): list of chess.Move valid moves
        valid_moves_str (list): list of str valid moves
    """
    valid_moves = list(board.legal_moves)
    valid_moves_str = [valid_moves[i].uci() for i in range(len(valid_moves))]
    return(valid_moves, valid_moves_str)

def get_stockfish_move(board, valid_moves, valid_moves_str, best_move_ai):
    """Get best stockfish move, stockfish score of the stockfish move, all valid moves sorted by stockfish score and ranking of the best ai move

    Args:
        board (chess.Board): chess board
        valid_moves (list): list of chess.Move valid moves
        valid_moves_str (list): list of str valid moves
        best_move_ai (chess.Move): best ai move

    Returns:
        best_move_stockfish (chess.Move): best move predicted by stockfish
        stockfish_score_stockfish_move (int): stockfish score of the best stockfish move
        stockfish_moves_sorted_by_score (numpy array): all valid moves sorted by stockfish score
        index (int): ranking of the best ai move according to stockfish
    """
    stockfish_scores = []
    for i in range(len(valid_moves)):
        board.push(valid_moves[i])

        result = engine.analyse(board.copy(), chess.engine.Limit(depth = 0))
        stockfish_score = result["score"].white().score(mate_score = 10000)
        stockfish_scores.append(stockfish_score)

        board.pop()

    stockfish_moves_sorted_by_score = sorted(zip(valid_moves_str, stockfish_scores), reverse=True)
    dtype = [("move", "U4"), ("score", int)]
    stockfish_moves_sorted_by_score = np.array(stockfish_moves_sorted_by_score, dtype = dtype)
    stockfish_moves_sorted_by_score = np.sort(stockfish_moves_sorted_by_score, order = "score")[::-1]
    best_move_stockfish = chess.Move.from_uci(stockfish_moves_sorted_by_score[0][0])
    stockfish_score_stockfish_move = stockfish_moves_sorted_by_score[0][1]
    # get ranking index of ai move according to stockfish
    index = [i for i, v in enumerate(stockfish_moves_sorted_by_score) if v[0] == best_move_ai.uci()][0]

    return(best_move_stockfish, stockfish_score_stockfish_move, stockfish_moves_sorted_by_score, index)

def convert_board_int_to_fen(board_int, number_boards_pieces, turn, castling, en_passant, halfmove_clock, fullmove_number):
    """Converts a n-dimensional list of the chess board back to its FEN format

    Args:
        board_int (np.array): (n, 8, 8) list of the input board with {1,0} int values
        number_boards_pieces (int): number of boards that describe the positions of the chess pieces
        turn (bool): white trun (True) or black turn (False)
        castling (str): string describing the castling rights of the game, e.g. Kqkq
        en_passant (str): possible en passant targets, e.g. e3
        halfmove_clock (int): number of moves both players have made since the last pawn advance or piece capture
        fullmove_number (int): number of completed turns in the game

    Returns:
        board_fen (str): chess board in FEN format
    """
    dict_pieces = np.array(["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"])
    dict_turn = np.array(["b", "w"]) # or vice versa?
    board_int_pieces = board_int[0:number_boards_pieces]

    board_fen = ""
    for row in range(8):
        blank_square_counter = 0
        for column in range(8):
            piece_array = board_int_pieces[:,row,column]
            piece_index = np.where(piece_array == 1)[0]

            if piece_index.size > 0 and blank_square_counter == 0:
                piece_str = dict_pieces[piece_index][0]
                board_fen += f"{piece_str[0]}"
                blank_square_counter = 0
            elif piece_index.size > 0 and blank_square_counter != 0:
                board_fen += f"{blank_square_counter}"
                piece_str = dict_pieces[piece_index][0]
                board_fen += f"{piece_str[0]}"
                blank_square_counter = 0
            else:
                blank_square_counter += 1

        if blank_square_counter != 0:
            board_fen += f"{blank_square_counter}/"
        else:
            board_fen += "/"

    board_fen = board_fen[:-1]    
    board_fen += f" {dict_turn[int(turn)]}"
    
    if castling != None:
        board_fen += f" {castling}"
    else:
        board_fen += f" -"

    if en_passant != None:
        board_fen += f" {en_passant}"
    else:
        board_fen += f" -"
    
    board_fen += f" {halfmove_clock}"
    board_fen += f" {fullmove_number}"
    
    return(board_fen)

def plot_history(history, name):
    print("Plotting history...")
    plt.figure()
    plt.plot(history["loss"], label="Training")
    plt.plot(history["val_loss"], label = "Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"evaluation/{name}/history_{name}.pdf")
    # plt.show()
    plt.close()

def plot_2d_scattering(prediction_val, true_score_val, name):
    print("Plotting 2D scattering...")
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    newcolors[:1, :] = white
    newcmp = ListedColormap(newcolors)

    plt.figure()
    plt.hist2d(true_score_val, prediction_val, bins = (50, 50), cmap = newcmp, norm = LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Number of boards')
    plt.plot(np.linspace(np.min(true_score_val), np.max(true_score_val), 100), np.linspace(np.min(true_score_val), np.max(true_score_val), 100), color = "black")
    # plt.ylim(np.min(true_score_val), np.max(true_score_val))
    plt.xlabel("True score")
    plt.ylabel("Predicted score")
    plt.tight_layout()
    plt.savefig(f"evaluation/{name}/2Dscattering_{name}.pdf")
    # plt.show()
    plt.close()

def plot_hist_difference_total(prediction_val, true_score_val, name):
    print("Plotting histogram difference total...")
    difference = prediction_val - true_score_val
    mean = np.mean(difference)
    median = np.median(difference)
    std = np.std(difference)
    plt.figure()
    plt.hist(difference, bins = 50, label = f"$\mu = {np.round(mean*1e4, 2)} \cdot 10^{{-4}}$ \nmedian $={np.round(median*1e4, 2)} \cdot 10^{{-4}}$ \n$\sigma={np.round(std*1e4, 2)} \cdot 10^{{-4}}$")
    plt.xlabel("pred. score - true score")
    plt.ylabel("Number of boards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"evaluation/{name}/hist_difference_total_{name}.pdf")
    # plt.show()
    plt.close()

def plot_hist_difference_binned(prediction_val, true_score_val, name):
    print("Plotting histogram difference binned...")
    true_score_min, true_score_max = np.min(true_score_val), np.max(true_score_val)
    bins = np.linspace(true_score_min, true_score_max, 5)

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    for subplot in range(4):
        indices = np.where((true_score_val >= bins[subplot]) & (true_score_val <= bins[subplot+1]))[0]
        true_score_val_binned = true_score_val[indices]
        prediction_val_binned = prediction_val[indices]
        difference = prediction_val_binned - true_score_val_binned
        mean = np.mean(difference)
        median = np.median(difference)
        std = np.std(difference)
        number_boards = len(difference)
        ax[subplot].set_title(f"{np.round(bins[subplot], 2)} < true score < {np.round(bins[subplot+1], 2)}")
        ax[subplot].hist(difference, bins = 50, label = f"$\mu = {np.round(mean*1e4, 1)} \cdot 10^{{-4}}$ \nmedian $={np.round(median*1e4, 1)} \cdot 10^{{-4}}$ \n$\sigma={np.round(std*1e4, 1)} \cdot 10^{{-4}}$ \n# boards = {number_boards}")
        ax[subplot].legend()
        ymin, ymax = ax[subplot].get_ylim()
        ax[subplot].set_ylim(ymin, ymax * 2.0)
        # plt.xlabel("pred. score - true score")
        # plt.ylabel("Number of boards")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"evaluation/{name}/hist_difference_binned_{name}.pdf")
    # plt.show()
    plt.close()

def save_examples(table, name):
    print("Saving examples...")
    os.system(f"rm evaluation/{name}/examples/*")

    model = models.load_model(f"model/model_{name}.h5")

    for layer in model.layers:
        if "conv" in layer.name:
            last_conv_layer_name = layer.name

    for i in range(len(table)):
        board = chess.Board(table["board (FEN)"][i])
        boardsvg = chess.svg.board(board = board.copy())

        path = f"evaluation/{name}/examples/board_diff_{np.round(table['difference'][i], 2):.2f}_ts_{np.round(table['true score'][i], 2):.2f}_ps_{np.round(table['prediction'][i], 2):.2f}"

        path = path_uniquify(path)

        outputfile = open(path +".svg", "w")
        outputfile.write(boardsvg)
        outputfile.close()
        os.system("convert -density 1200 -resize 780x780 " + path + ".svg " + path + ".png")
        os.system("rm " + path + ".svg")

        X_board3d = board_3d_attack_int(board.copy())
        X_board3d = np.array([np.moveaxis(X_board3d, 0, -1)])
        X_parameter = np.array([[board.copy().turn, board.copy().halfmove_clock]])

        heatmap = make_gradcam_heatmap([X_board3d, X_parameter], model, last_conv_layer_name)

        # save heatmap
        plt.figure()
        plt.matshow(heatmap, cmap = "gnuplot")
        plt.axis("off")
        plt.savefig(path + "_heatmap.png", bbox_inches = "tight", pad_inches = 0.15, dpi = 194.2)
        plt.close()

        plt.figure()
        img_heatmap = plt.imread(path + "_heatmap.png")
        img_board = plt.imread(path + ".png")
        plt.imshow(img_board, interpolation = "nearest")
        plt.imshow(img_heatmap, alpha = 0.7, interpolation = "nearest")
        plt.axis("off")
        plt.savefig(path + "_gradcam.png", bbox_inches="tight", pad_inches = 0, dpi = 211.2)

        os.system("rm " + path + "_heatmap.png")

        plt.close("all")

        print(f"Board {i} saved...")

def path_uniquify(path):
    filename = path
    counter = 1

    while os.path.exists(path + ".png"):
        path = filename + "_" + str(counter)
        counter += 1

    return(path)

def make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    # print(grads)
    # print(np.min(grads), np.max(grads))
    # print(np.shape(grads))

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return(heatmap.numpy())

# def save_examples_gradcam(table, name):
#     print("Saving examples gradcam...")
#     model = models.load_model(f"model/model_{name}.h5")

#     for layer in model.layers:
#         if "conv" in layer.name:
#             last_conv_layer_name = layer.name

#     for i in range(len(table)):
#         board = chess.Board(table["board (FEN)"][i])
#         X_board3d = board_3d_attack_int(board.copy())
#         X_board3d = np.array([np.moveaxis(X_board3d, 0, -1)])
#         X_parameter = np.array([[board.copy().turn, board.copy().halfmove_clock]])

#         heatmap = make_gradcam_heatmap([X_board3d, X_parameter], model, last_conv_layer_name)

#         path = f"evaluation/{name}/examples/board_diff_{np.round(table['difference'][i], 2):.2f}_ts_{np.round(table['true score'][i], 2):.2f}_ps_{np.round(table['prediction'][i], 2):.2f}"

#         path = path_uniquify(path)

#         # save heatmap
#         plt.figure()
#         plt.matshow(heatmap, cmap = "gnuplot")
#         plt.axis("off")
#         plt.savefig(path + "_heatmap.png", bbox_inches = "tight", pad_inches = 0.15, dpi = 194.2)
#         plt.close()

#         plt.figure()
#         img_heatmap = plt.imread(path + "_heatmap.png")
#         img_board = plt.imread(path + ".png")
#         plt.imshow(img_board, interpolation = "nearest")
#         plt.imshow(img_heatmap, alpha = 0.7, interpolation = "nearest")
#         plt.axis("off")
#         plt.savefig(path + "_gradcam.png", bbox_inches="tight", pad_inches = 0, dpi = 211.2)

#         os.system("rm " + path + "_heatmap.png")

#         plt.close("all")

#         print(f"GradCam Board {i} saved...")

        