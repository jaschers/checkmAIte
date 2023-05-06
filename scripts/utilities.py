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
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import cairosvg
from PIL import Image, ImageTk
import tkinter as tk
import io
import logging

np.set_printoptions(threshold=sys.maxsize)

stockfish_path = os.environ.get("STOCKFISHPATH")

# get stockfish engine
stockfish_path = os.environ.get("STOCKFISHPATH")
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

best_move = None
counter = 0
score_max = 15000

def square_to_index(square):
    """
    converts square number to 2D index
    Args:
        square (int): square number
    Returns:
        list: 2D index
    """
    squares = np.linspace(0, 8*8 - 1, 8*8, dtype = int)
    squares_2d = np.reshape(squares, (8,8))
    squares_2d = np.flip(squares_2d, 0)

    if type(square) == int:
        index = np.where(squares_2d == square)
        row_index, column_index = index[0][0], index[1][0]
        return(row_index, column_index)
    elif type(square) == list:
        indices = []
        for sqr in square:
            index = np.where(squares_2d == sqr)
            indices.append([index[0][0], index[1][0]])
        return(indices) 

def board_int(board):
    """
    converts chess board into 2D list with
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
    """
    converts chess board into 3D (24, 8, 8) array with board[i] representing:
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

def board_score(board, depth = 0):
    """
    Evaluates the score of a board for player white based on stockfish.

    Args:
        board (chess.Board): chess board in FEN format
        depth (int, optional): stockfish depth. Default 0

    Returns:
        int: stockfish score of the input board
    """
    score_dict = {"15000": 15000, "14999": 14000, "14998": 13000, "14997": 12000, "14996": 11000, "14995": 10000, "14994": 9000, "14993": 8000, "-15000": -15000, "-14999": -14000, "-14998": -13000, "-14997": -12000, "-14996": -11000, "-14995": -10000, "-14994": -9000, "-14993": -8000}
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    result = engine.analyse(board.copy(), chess.engine.Limit(depth = depth))
    score = result["score"].white().score(mate_score = score_max)
    if str(score) in score_dict:
        score = score_dict[f"{score}"]
    engine.quit()
    return(score)

def boards_random(num_boards):
    """
    Creates random boards by playing games with random moves

    Args:
        num_boards (int): number of boards being created

    Returns:
        list: (N,) list including all the randomly generated boards N while playing the games
    """
    boards_random_int = []
    boards_random_parameter = []
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

        boards_random_int.append(get_board_total(board.copy()))
        boards_random_parameter.append(get_board_parameters(board.copy()))
        boards_random_score.append(np.int16(board_score(board.copy())))

    boards_random_parameter = np.array(boards_random_parameter)

    return(boards_random_int, boards_random_parameter, boards_random_score)

def ai_board_score_pred(board, model):
    """
    Predicts the score of a board on the CNN model.
    Args:
        board (chess.Board): chess board
        model (keras.model): CNN model
    Returns:
        float: predicted score of the input board
    """
    board_3d_int = [get_board_total(board.copy())]
    board_3d_int = np.moveaxis(board_3d_int, 1, -1)
    parameters = np.array([get_model_input_parameter(board.copy())])
    prediction_score = model.predict([board_3d_int, parameters], verbose = 0)[0][0] * 2 * score_max - score_max
    return(prediction_score)


def minimax_parallel(board, model, depth, alpha, beta, maximizing_player, transposition_table, verbose_minimax = False):
    if depth < 0 or type(depth) != int:
        raise ValueError("Depth needs to be int and greater than 0")

    # Check if the current game state is already in the transposition table
    hash_value = board.fen()
    if hash_value in transposition_table:
        return(transposition_table[hash_value])

    if depth == 0 or board.is_game_over() == True:
        prediction = ai_board_score_pred(board.copy(), model)
 
        # Add the current game state and its evaluation to the transposition table
        transposition_table[hash_value] = prediction
        return(prediction)

    # maximizing_player == True -> AI's turn
    if maximizing_player == True:
        evals = []
        for valid_move in board.legal_moves:
            board.push(valid_move)
            eval = dask.delayed(minimax)(board.copy(), model, depth - 1, alpha, beta, False, transposition_table, verbose_minimax)
            board.pop()
            evals.append(eval)
        
        evals = dask.compute(*evals, num_workers = 3)
        # print("ai turn evals", evals)
        argmax = np.argmax(evals)
        max_eval = evals[argmax]
        
        # Add the current game state and its evaluation to the transposition table
        transposition_table[hash_value] = max_eval
        return(max_eval)

    # maximizing_player == False -> player's turn
    else:
        evals = []
        for valid_move in board.legal_moves:
            board.push(valid_move)
            eval = dask.delayed(minimax)(board.copy(), model, depth - 1, alpha, beta, True, transposition_table, verbose_minimax)
            board.pop()
            evals.append(eval)            
        
        evals = dask.compute(*evals, num_workers = 3)
        # print("players turn evals", evals)
        argmin = np.argmin(evals)
        min_eval = evals[argmin]
        # Add the current game state and its evaluation to the transposition table
        transposition_table[hash_value] = min_eval
        return(min_eval)


def minimax(board, model, depth, alpha, beta, maximizing_player, transposition_table, best_move = None, verbose_minimax = False):
    """
    Minimax algorithm with alpha-beta pruning, transposition table and move ordering
    Args:
        board (chess.Board): chess board
        model (keras.model): CNN model
        depth (int): depth of the search tree
        alpha (float): alpha value for alpha-beta pruning
        beta (float): beta value for alpha-beta pruning
        maximizing_player (bool): True if AI is playing, False if player is playing
        transposition_table (dict): transposition table
        best_move (chess.Move): best move
        verbose_minimax (bool): True if you want to print the progress of the minimax algorithm, False if not
    Returns:
        float: evaluation of the board
        chess.Move: best move
    """
    # print("alpha", alpha, "beta", beta)
    if depth < 0 or type(depth) != int:
        raise ValueError("Depth needs to be int and greater than 0")

    # # Check if this position is already in the transposition table
    hash_value = board.fen()[:-4] # ignore halmove clock and fullmove number
    if hash_value in transposition_table:
        entry = transposition_table[hash_value]
        # print("hash_value", hash_value)
        # print("transposition_table[hash_value]", transposition_table[hash_value])
        # Check if the stored depth is greater than or equal to the current depth
        if entry["depth"] >= depth:
            # Use the stored evaluation and best move
            if entry["flag"] == "exact":
                return entry["eval"], entry["best_move"]
            elif entry["flag"] == "lower_bound":
                alpha = max(alpha, entry["eval"])
            elif entry["flag"] == "upper_bound":
                beta = min(beta, entry["eval"])
            if alpha >= beta:
                return entry["eval"], entry["best_move"]

    if depth == 0 or board.is_game_over():
        prediction = ai_board_score_pred(board.copy(), model)
        # analyse_stockfish = engine.analyse(board, chess.engine.Limit(depth = 0))
        # prediction = analyse_stockfish["score"].white().score(mate_score = score_max)
        # Add the current game state and its evaluation to the transposition table
        transposition_table[hash_value] = {"depth": depth, "flag": "exact", "eval": prediction, "ancient": len(transposition_table), "best_move": None}
        return(prediction, None)

    if maximizing_player:
        max_eval = -np.inf
        ordered_moves = order_moves(board, transposition_table)
        if verbose_minimax == True:
            ordered_moves = tqdm(ordered_moves)
        for move in ordered_moves:
            board.push(move)
            eval, _ = minimax(board.copy(), model, depth - 1, alpha, beta, False, transposition_table, best_move, verbose_minimax = False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                ("alpha-beta pruning")
                break

        # Add the current game state and its evaluation to the transposition table
        if max_eval <= alpha:
            flag = "upper_bound"
        elif max_eval >= beta:
            flag = "lower_bound"
        else:
            flag = "exact"
        transposition_table[hash_value] = {"depth": depth, "flag": flag, "eval": max_eval, "ancient": len(transposition_table), "best_move": best_move}

        return(max_eval, best_move)

    else:
        # # Null move pruning
        # if depth >= 2:
        #     null_move = chess.Move.null()
        #     board.push(null_move)
        #     eval, _ = minimax(board.copy(), model, depth - 3, alpha, beta, True, transposition_table, best_move, verbose_minimax = False)
        #     board.pop()
        #     print("Null move pruning attempt")
        #     print("beta", beta, "eval", eval)

        #     if eval >= beta:
        #         print("Null move pruning")
        #         return beta
        
        min_eval = np.inf
        ordered_moves = order_moves(board, transposition_table)
        for move in ordered_moves:
            board.push(move)
            eval, _ = minimax(board.copy(), model, depth - 1, alpha, beta, True, transposition_table, best_move, verbose_minimax = False)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                ("alpha-beta pruning")
                break

        # Store the evaluation, best move, and flag in the transposition table
        if min_eval <= alpha:
            flag = "upper_bound"
        elif min_eval >= beta:
            flag = "lower_bound"
        else:
            flag = "exact"
        transposition_table[hash_value] = {"depth": depth, "flag": flag, "eval": min_eval, "ancient": len(transposition_table), "best_move": best_move}

        return (min_eval, best_move)


def get_ai_move(board, model, depth, transposition_table, verbose_minimax):
    """
    Get the best move for the AI
    Args:
        board (chess.Board): chess board
        model (keras.Model): neural network model
        depth (int): depth of the minimax algorithm
        transposition_table (dict): transposition table
        verbose_minimax (bool): True if you want to print the progress of the minimax algorithm, False if not
    Returns:
        chess.Move: best move
        float: evaluation of the board
    """
    max_eval, max_move = minimax(board.copy(), model, depth = depth, alpha = -np.inf, beta = np.inf, maximizing_player = True, transposition_table = transposition_table, best_move = None, verbose_minimax = verbose_minimax)

    return(max_move, max_eval)

def get_ai_move_parallel(board, model, depth, transposition_table, verbose_minimax):
    max_move = None
    max_eval = -np.inf
    evals = []
    start = time.time()
    for valid_move in list(board.legal_moves):
        board.push(valid_move)
        # maximizing_player == False -> player's move because AI's (potential) move was just pushed
        eval = dask.delayed(minimax)(board.copy(), model, depth = depth - 1, alpha = -np.inf, beta = np.inf, maximizing_player = False, transposition_table = transposition_table, verbose_minimax = verbose_minimax)
        board.pop()
        evals.append(eval)
    with ProgressBar():
        evals = dask.compute(*evals, num_workers = 3)
    argmax = np.argmax(evals)
    max_eval = evals[argmax]
    max_move = list(board.legal_moves)[argmax]

    print("time needed", time.time() - start)
    return(max_move, max_eval)

def save_board_png(board, game_name, counter):
    """
    Saves the current board as png in games/{game_name}/board{counter}.png

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

def delete_board_png(game_name, counter):
    """
    deletes the board saved as games/{game_name}/board{counter}.png

    Args:
        game_name (str): name of the current chess game
        counter (int): board move counter
    """
    os.system(f"rm games/{game_name}/board{counter}.png")

def save_board_gif(boards_png, game_name):
    """
    Loads png images of a chess game and converts it into a gif. The png images are deleted afterwards

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
    ani.save(f"games/{game_name}/board.gif")

    os.system(f"rm games/{game_name}/*.png")

def get_valid_moves(board):
    """
    returns the valid moves of a board

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
    """
    Get best stockfish move, stockfish score of the stockfish move, all valid moves sorted by stockfish score and ranking of the best ai move

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
        result = engine.analyse(board, chess.engine.Limit(depth = 0))
        stockfish_score = result["score"].white().score(mate_score = score_max)
        stockfish_scores.append(stockfish_score)

        board.pop()

    stockfish_moves_sorted_by_score = sorted(zip(valid_moves_str, stockfish_scores), reverse=True)
    dtype = [("move", "U8"), ("score", int)]
    stockfish_moves_sorted_by_score = np.array(stockfish_moves_sorted_by_score, dtype = dtype)
    stockfish_moves_sorted_by_score = np.sort(stockfish_moves_sorted_by_score, order = "score")[::-1]
    best_move_stockfish = chess.Move.from_uci(stockfish_moves_sorted_by_score[0][0])
    stockfish_score_stockfish_move = stockfish_moves_sorted_by_score[0][1]
    # get ranking index of ai move according to stockfish
    index = [i for i, v in enumerate(stockfish_moves_sorted_by_score) if v[0] == best_move_ai.uci()][0]

    return(best_move_stockfish, stockfish_score_stockfish_move, stockfish_moves_sorted_by_score, index)

def convert_board_int_to_fen(board_int, number_boards_pieces, turn, castling, en_passant, halfmove_clock, fullmove_number):
    """
    Converts a n-dimensional list of the chess board back to its FEN format

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
    """
    Plot training and validation loss
    Args:
        history (dict): history of the training
        name (str): name of the model
    """
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
    """
    Plot 2D scattering of the true and predicted scores
    Args:
        prediction_val (np.array): (n,) array of the predicted scores
        true_score_val (np.array): (n,) array of the true scores
        name (str): name of the model
    """
    print("Plotting 2D scattering...")
    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    newcolors[:1, :] = white
    newcmp = ListedColormap(newcolors)

    plt.figure()
    plt.grid(alpha = 0.3)
    plt.hist2d(true_score_val, prediction_val, bins = (200, 200), cmap = newcmp, norm = LogNorm())
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

def plot_hist_difference_total(prediction, true, parameter, name):
    """
    Plot histogram of the difference between the true and predicted scores
    Args:
        prediction (np.array): (n,) array of the predicted scores
        true (np.array): (n,) array of the true scores 
        parameter (str): name of the parameter
        name (str): name of the model
    """
    print(f"Plotting {parameter} histogram difference total...")
    difference = prediction - true
    mean = np.mean(difference)
    median = np.median(difference)
    std = np.std(difference)
    plt.figure()
    if parameter != "score":
        plt.hist(difference, bins = 25, range = (-1, 1), label = f"$\mu = {np.round(mean*1e4, 2)} \cdot 10^{{-4}}$ \nmedian $={np.round(median*1e4, 2)} \cdot 10^{{-4}}$ \n$\sigma={np.round(std*1e4, 2)} \cdot 10^{{-4}}$")
    else:
        plt.hist(difference, bins = 25, range = (-15000, 15000), label = f"$\mu = {np.round(mean, 2)}$ \nmedian $={np.round(median, 2)}$ \n$\sigma={np.round(std, 2)}$")
    # plt.hist(difference, bins = 50, label = "$\mu = {0}$ \nmedian $={1}$ \n$\sigma={2}$".format(mean, median, std))
    plt.xlabel(f"pred. {parameter} - true {parameter}")
    plt.ylabel("Number of boards")
    if parameter != "score":
        plt.xlim(-1, 1)
    else:
        plt.xlim(-30000, 30000)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"evaluation/{name}/{parameter}_hist_difference_total_{name}.pdf")
    # plt.show()
    plt.close()

def plot_hist_difference_binned(prediction_val, true_score_val, name):
    """
    Plot histogram of the difference between the true and predicted scores binned
    Args:
        prediction_val (np.array): (n,) array of the predicted scores
        true_score_val (np.array): (n,) array of the true scores
        name (str): name of the model
    """
    print("Plotting score histogram difference binned...")
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
        ax[subplot].hist(difference, bins = 25, label = f"$\mu = {np.round(mean*1e4, 1)} \cdot 10^{{-4}}$ \nmedian $={np.round(median*1e4, 1)} \cdot 10^{{-4}}$ \n$\sigma={np.round(std*1e4, 1)} \cdot 10^{{-4}}$ \n# boards = {number_boards}")
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

def plot_hist(table, parameter, name):
    print(f"Plot {parameter} histogram")
    table_score = table[table[parameter] == 1].reset_index(drop = True)
    table_score = table_score["predicted score"]

    mean = np.mean(table_score)
    median = np.median(table_score)
    std = np.std(table_score)

    parameter = parameter.replace(" ", "")

    plt.figure()
    plt.hist(table_score, bins = 25, range = (-15000, 15000), label = f"$\mu = {np.round(mean, 2)}$ \nmedian $={np.round(median, 2)}$ \n$\sigma={np.round(std, 2)}$")
    plt.xlabel(f"predicted score")
    plt.ylabel("Number of boards")
    # if parameter != "score":
    plt.yscale("log")
    # plt.xlim(-1, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"evaluation/{name}/{parameter}_hist_score_total_{name}.pdf")
    # plt.show()
    plt.close()


def save_examples(table, name):
    """
    Save examples of the boards
    Args:
        table (pd.DataFrame): table with the boards and the scores
        name (str): name of the model
    """
    print("Saving examples...")
    os.system(f"rm evaluation/{name}/examples/*")

    model = models.load_model(f"model/model_{name}.h5")

    for layer in model.layers:
        if "conv" in layer.name:
            last_conv_layer_name = layer.name

    for i in range(len(table)):
        board = chess.Board(table["board (FEN)"][i])
        boardsvg = chess.svg.board(board = board.copy())

        # difference = table['difference'][i] * 30000 - 15000
        true_score = table['true score'][i] * 2 * score_max - score_max
        predicted_score = table['predicted score'][i] * 2 * score_max - score_max
        difference = predicted_score - true_score
        if int(table['turn'][i]) == 0:
            turn = "black"
        else:
            turn = "white"

        path = f"evaluation/{name}/examples/board_diff_{difference:.0f}_ts_{true_score:.0f}_ps_{predicted_score:.0f}_turn_{turn}"

        print(path)
        path = path_uniquify(path)
        print(board.fen())

        outputfile = open(path +".svg", "w")
        outputfile.write(boardsvg)
        outputfile.close()
        os.system("convert -density 1200 -resize 780x780 " + path + ".svg " + path + ".png")
        os.system("rm " + path + ".svg")

        X_board3d = get_board_total(board.copy())
        X_board3d = np.array([np.moveaxis(X_board3d, 0, -1)])
        X_parameter = np.array([get_model_input_parameter(board.copy())])

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
    """
    Add a number to the end of the path if the path already exists
    Args:
        path (str): path to the file
    Returns:
        path (str): path to the file
    """
    filename = path
    counter = 1

    while os.path.exists(path + ".png"):
        path = filename + "_" + str(counter)
        counter += 1

    return(path)

def make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    """
    Make a heatmap of the gradient of the output neuron
    Args:
        img (np.array): input image
        model (keras.model): model
        last_conv_layer_name (str): name of the last convolutional layer
        pred_index (int): index of the output neuron
    Returns:
        heatmap (np.array): heatmap
    """
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

def get_board_parameters(board):
    """
    Returns board parameters from a given board.

    Args:
        board (chess.Board): chess board
    
    Returns:
        tuple: (14,) of:
            bool: side to move (True = white, False = black)
            int: halfmove clock number
            int: fullmove number
            bool: checks if the current side to move is in check
            bool: checks if the current side to move is in checkmate
            bool: checks if the current side to move is in stalemate
            bool: checks if white has insufficient winning material
            bool: checks if black has insufficient winning material
            bool: checks seventy-five-move rule
            bool: checks fivefold repetition
            bool: checks castling right king side of white
            bool: checks castling right queen side of white
            bool: checks castling right king side of black
            bool: checks castling right queen side of black
    """

    turn = board.turn
    halfmove_clock = board.halfmove_clock
    fullmove_number = board.fullmove_number
    check = board.is_check()
    checkmate = board.is_checkmate()
    stalemate = board.is_stalemate()
    insufficient_material_white = board.has_insufficient_material(chess.WHITE)
    insufficient_material_black = board.has_insufficient_material(chess.BLACK)
    seventyfive_moves = board.is_seventyfive_moves()
    fivefold_repetition = board.is_fivefold_repetition()
    # threefold_repetition = board.is_repetition()
    castling_right_king_side_white = board.has_kingside_castling_rights(chess.WHITE)
    castling_right_queen_side_white = board.has_queenside_castling_rights(chess.WHITE)
    castling_right_king_side_black = board.has_kingside_castling_rights(chess.BLACK)
    castling_right_queen_side_black = board.has_queenside_castling_rights(chess.BLACK)

    return(
        turn,
        halfmove_clock,
        fullmove_number,
        check, 
        checkmate, 
        stalemate, 
        insufficient_material_white, 
        insufficient_material_black, 
        seventyfive_moves, 
        fivefold_repetition, 
        castling_right_king_side_white, 
        castling_right_queen_side_white, 
        castling_right_king_side_black, 
        castling_right_queen_side_black
        )

def get_board_pinned(board):
    """
    Returns board of pinned black and white pieces

    Args:
        board (chess.Board): chess board

    Returns:
        list (8, 8): list of a board with pinned black and white pieces
    """
    board_pinned = np.zeros((8, 8), dtype = int)
    for square in chess.SQUARES:
        if (board.is_pinned(chess.WHITE, square) == True) or (board.is_pinned(chess.BLACK, square) == True):
            board_index = square_to_index(square)
            board_pinned[board_index[0]][board_index[1]] = 1
    # board_pinned = board_pinned.tolist()

    return(board_pinned)

def get_board_en_passant(board):
    """
    Returns board of possible en passant move

    Args:
        board (chess.Board): chess board

    Returns:
        list (8, 8): list of a board with square that can be attacked by en passant move
    """
    board_en_passant = np.zeros((8, 8), dtype = int)
    if board.has_legal_en_passant() == True:
        board_index = square_to_index(board.ep_square)
        board_en_passant[board_index[0]][board_index[1]] = 1
    # board_en_passant = board_en_passant.tolist()

    return(board_en_passant)

def get_board_3d_pieces(board):
    """
    converts chess board into 3D (12, 8, 8) list with board[i] representing:
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
    Args:
        board (chess.Board): chess board

    Returns:
        list: (12, 8, 8) list of the input board with {1,0} int values
    """
    # initialise board array
    number_boards = 12 
    board_pieces = np.zeros((number_boards, 8, 8), dtype = int)

    # for loop over all piece types (pawn, knight, ...)
    for piece in chess.PIECE_TYPES:
        # for loop over all squares of white pieces
        for square in board.pieces(piece, chess.WHITE):
            # get indices of the individual piece
            board_index = square_to_index(square)
            # fill array at board_index with piece value for each piece
            board_pieces[piece - 1][board_index[0]][board_index[1]] = 1

        # for loop over all squares of black pieces
        for square in board.pieces(piece, chess.BLACK):
            # get indices of the individual piece
            board_index = square_to_index(square)
            # fill array at board_index with piece value for each piece
            board_pieces[piece - 1 + 6][board_index[0]][board_index[1]] = 1

    # board_pieces = board_pieces.tolist()
    return(board_pieces)

def get_board_3d_attacks(board):
    """
    converts chess board into 3D (12, 8, 8) list with board[i] representing:
    0: all squares being attacked/defended by white pawn
    1: all squares being attacked/defended by white knight
    2: all squares being attacked/defended by white bishop
    3: all squares being attacked/defended by white rook
    4: all squares being attacked/defended by white queen
    5: all squares being attacked/defended by white king
    6: all squares being attacked/defended by black pawn
    7: all squares being attacked/defended by black knight
    8: all squares being attacked/defended by black bishop
    9: all squares being attacked/defended by black rook
    10: all squares being attacked/defended by black queen
    11: all squares being attacked/defended by black king
    Args:
        board (chess.Board): chess board

    Returns:
        list: (12, 8, 8) list of the input board with values n (int) where n represents the number of attacks per piece
    """
    # initialise board array
    number_boards = 12 
    board_attacks = np.zeros((number_boards, 8, 8), dtype = int)

    # get king positions
    king_pos_white, king_pos_black = board.king(chess.WHITE), board.king(chess.BLACK)
    # print("king_pos_white, king_pos_black")
    # print(king_pos_white, king_pos_black)
    # for loop over all piece types (pawn, knight, ...)
    for piece in chess.PIECE_TYPES:
        # for loop over all squares of white pieces
        for square in board.pieces(piece, chess.WHITE):
            # make black king "invisible" by removing him from the board
            # print("before remove white")
            # print(board)
            # print("_____________")
            board.remove_piece_at(king_pos_black)
            # print("after remove white")
            # print(board)
            # print("_____________")
            # get squares that are attacked by the piece
            attacks_squares = list(board.attacks(square))
            for attack_square in attacks_squares:
                board_index = square_to_index(attack_square)
                board_attacks[piece - 1][board_index[0]][board_index[1]] += 1
            # add black king back to the board
            # print("before set piece at white")
            # print(board)
            # print("_____________")
            board.set_piece_at(chess.Square(king_pos_black), chess.Piece(chess.KING, chess.BLACK))
            # print("after set piece at white")
            # print(board)
            # print("_____________")
        # for loop over all squares of black pieces
        for square in board.pieces(piece, chess.BLACK):
            # make white king "invisible" by removing him from the board
            # print("before remove black")
            # print(board)
            # print("_____________")
            board.remove_piece_at(king_pos_white)
            # print("after remove black")
            # print(board)
            # print("_____________")
            # get squares that are attacked by the piece
            attacks_squares = list(board.attacks(square))
            for attack_square in attacks_squares:
                board_index = square_to_index(attack_square)
                board_attacks[piece - 1 + 6][board_index[0]][board_index[1]] += 1
            # add black white back to the board
            # print("before set piece at black")
            # print(board)
            # print("_____________")
            board.set_piece_at(chess.Square(king_pos_white), chess.Piece(chess.KING, chess.WHITE))
            # print("after set piece at black")
            # print(board)
            # print("_____________")
    # board_attacks = board_attacks.tolist()
    return(board_attacks)

def get_board_3d_2nd_attacks(board):
    """converts chess board into 3D (12, 8, 8) list with board[i] representing:
    0: all squares being potentially attacked/defended in the next move by white pawn
    1: all squares being potentially attacked/defended in the next move by white knight
    2: all squares being potentially attacked/defended in the next move by white bishop
    3: all squares being potentially attacked/defended in the next move by white rook
    4: all squares being potentially attacked/defended in the next move by white queen
    5: all squares being potentially attacked/defended in the next move by white king
    6: all squares being potentially attacked/defended in the next move by black pawn
    7: all squares being potentially attacked/defended in the next move by black knight
    8: all squares being potentially attacked/defended in the next move by black bishop
    9: all squares being potentially attacked/defended in the next move by black rook
    10: all squares being potentially attacked/defended in the next move by black queen
    11: all squares being potentially attacked/defended in the next move by black king

    Args:
        board (chess.Board): chess board

    Returns:
        list: (12, 8, 8) list of the input board with values {0, 1, 2} which represent the number of attacks per piece
    """
    # print("get_board_3d_2nd_attacks beginning")
    # print(board)
    # initialise board array
    number_boards = 12
    board_2nd_attacks = np.zeros((number_boards, 8, 8), dtype = int)

    # get original board turn
    turn = board.turn
    # print("original turn")
    # print(turn)

    # get 2nd attacks for white
    board.turn = chess.WHITE

    valid_moves = list(board.legal_moves)
    # print("valid_moves white")
    # print(valid_moves)
    for move in valid_moves:
        # print("get_board_3d_2nd_attacks before move white")
        # print(board)
        # print("move white")
        # print(move)
        # print(board.is_legal(move))
        board.push(move)
        # print("get_board_3d_2nd_attacks after move white")
        # print(board)
        # print("Is board valid?", board.is_valid())
        if board.is_valid() == False:
            board.pop()
        else:
            board_attacks_move_i = np.array(get_board_3d_attacks(board.copy()))

            board_attacks_move_i = board_attacks_move_i[:int(len(board_attacks_move_i) / 2)]
            board_attacks_move_i[board_attacks_move_i > 1] = - 1000 # arbitrary high negative number

            board_2nd_attacks[:6] = board_2nd_attacks[:6] + board_attacks_move_i

            board.pop()

    # get 2nd attacks for black
    board.turn = chess.BLACK

    valid_moves = list(board.legal_moves)
    # print("valid_moves black")
    # print(valid_moves)
    for move in valid_moves:
        # print("get_board_3d_2nd_attacks before move black")
        # print(board)
        # print("move black")
        # print(move)
        # print(board.is_legal(move))
        board.push(move)
        # print("get_board_3d_2nd_attacks after move black")
        # print(board)
        # print("Is board valid?", board.is_valid())
        if board.is_valid() == False:
            board.pop()
        else:
            board_attacks_move_i = np.array(get_board_3d_attacks(board.copy()))

            board_attacks_move_i = board_attacks_move_i[int(len(board_attacks_move_i) / 2):]
            board_attacks_move_i[board_attacks_move_i > 1] = - 1000 # arbitrary high negative number

            board_2nd_attacks[6:] = board_2nd_attacks[6:] + board_attacks_move_i

            board.pop()

    board_2nd_attacks[board_2nd_attacks > 0] = 1
    board_2nd_attacks[board_2nd_attacks < 0] = 2
    # board_2nd_attacks = board_2nd_attacks.tolist()

    # put turn back to original 
    board.turn = turn

    # print("get_board_3d_2nd_attacks end")
    # print(board)

    return(board_2nd_attacks)

def get_board_3d_pawn_move(board):
    """
    converts chess board into 3D (2, 8, 8) list with board[i] representing:
    0: all squares being a potential move by white pawns
    1: all squares being a potential move by black pawns

    Args:
        board (chess.Board): chess board
    
    Returns:
        list: (2, 8, 8) list of the input board with {0,1} values
    """
    # initialise board array
    number_boards = 2
    board_pawn_move = np.zeros((number_boards, 8, 8), dtype = int)

    board.turn = chess.WHITE
    for valid_move in board.legal_moves:
        piece_type = board.piece_type_at(valid_move.from_square)
        if piece_type == chess.PAWN:
            square = valid_move.to_square
            board_index = square_to_index(square)
            board_pawn_move[0][board_index[0]][board_index[1]] += 1

    board.turn = chess.BLACK
    for valid_move in board.legal_moves:
        piece_type = board.piece_type_at(valid_move.from_square)
        if piece_type == chess.PAWN:
            square = valid_move.to_square
            board_index = square_to_index(square)
            board_pawn_move[1][board_index[0]][board_index[1]] += 1
    
    # board_pawn_move = board_pawn_move.tolist()
    return(board_pawn_move)

def get_board_total(board):
    """
    converts chess board into 3D (40, 8, 8) list with board[i] representing:
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
    12: all squares being attacked/defended by white pawn
    13: all squares being attacked/defended by white knight
    14: all squares being attacked/defended by white bishop
    15: all squares being attacked/defended by white rook
    16: all squares being attacked/defended by white queen
    17: all squares being attacked/defended by white king
    18: all squares being attacked/defended by black pawn
    19: all squares being attacked/defended by black knight
    20: all squares being attacked/defended by black bishop
    21: all squares being attacked/defended by black rook
    22: all squares being attacked/defended by black queen
    23: all squares being attacked/defended by black king
    24: all squares being potentially attacked/defended in the next move by white pawn
    25: all squares being potentially attacked/defended in the next move by white knight
    26: all squares being potentially attacked/defended in the next move by white bishop
    27: all squares being potentially attacked/defended in the next move by white rook
    28: all squares being potentially attacked/defended in the next move by white queen
    29: all squares being potentially attacked/defended in the next move by white king
    30: all squares being potentially attacked/defended in the next move by black pawn
    31: all squares being potentially attacked/defended in the next move by black knight
    32: all squares being potentially attacked/defended in the next move by black bishop
    33: all squares being potentially attacked/defended in the next move by black rook
    34: all squares being potentially attacked/defended in the next move by black queen
    35: all squares being potentially attacked/defended in the next move by black king
    36: all squares being a potential move by white pawns
    37: all squares being a potential move by black pawns
    38: all squares being pinned by black or white
    39: all squares being possible en passant moves

    Args:
        board (chess.Board): chess board

    Returns:
        list: (40, 8, 8) list of the input board
    """
    board_pieces = get_board_3d_pieces(board.copy())
    board_pawn_move = get_board_3d_pawn_move(board.copy())
    board_pinned = np.array([get_board_pinned(board.copy())])
    board_en_passant = np.array([get_board_en_passant(board.copy())])
    board_attacks = get_board_3d_attacks(board.copy())
    board_2nd_attacks = get_board_3d_2nd_attacks(board.copy())

    board_total = np.concatenate(
        (board_pieces,
        board_attacks,
        board_2nd_attacks,
        board_pawn_move,
        board_pinned,
        board_en_passant)
    )
    
    board_total = board_total.tolist()
    return(board_total)

def get_model_input_parameter(board):
    """
    Returns neural network input parameters from a given board.

    Args:
        board (chess.Board): chess board
    
    Returns:
        tuple: (10,) of:
            bool: side to move (True = white, False = black)
            int: halfmove clock number
            bool: checks if white has insufficient winning material
            bool: checks if black has insufficient winning material
            bool: checks seventy-five-move rule
            bool: checks fivefold repetition
            bool: checks castling right king side of white
            bool: checks castling right queen side of white
            bool: checks castling right king side of black
            bool: checks castling right queen side of black
    """
    X_parameter = get_board_parameters(board.copy())
    X_parameter = X_parameter[:2] + X_parameter[6:]
    return(X_parameter)


def setup_logging(dt_string):
    """
    Set up logging for the program. If dt_string is not None, then the log file will be saved in the games folder with the name of the folder being the date and time of the game. If dt_string is None, then the log file will not be saved.
    Args:
        dt_string (str): date and time of the game
    Returns:
        logger (logging.Logger): logger object
    """
    # set up logger
    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a stream handler to print the log messages to the console
    console_handler = logging.StreamHandler()

    # Define the log message format
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)

    # Create a file handler to save the log messages to a file
    if dt_string != None:
        log_file = f"games/{dt_string}/logging.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add the file handler and the stream handler to the logger
    logger.addHandler(console_handler)
    
    return(logger)

def order_moves(board, transposition_table):
    """
    Orders the moves in the following order:
    1. Moves that are in the transposition table
    2. Moves that give check
    3. Moves that capture a piece
    4. Other moves by piece position (centre is favoured)
    Args: 
        board (chess.Board): chess board
        transposition_table (dict): transposition table
    Returns:
        list: list of moves in the order described above
    """
    moves = list(board.legal_moves)

    data = []
    # Sort the moves by the value of the captured piece minus the value of the capturing piece
    for move in moves:
        board_temp = board.copy()
        board_temp.push(move)
        hash_value = board_temp.fen()[:-4]
        
        if hash_value in transposition_table:
            score = transposition_table[hash_value]["eval"]
            data.append([move, True, False, False, False, score])

        elif board.gives_check(move):
            data.append([move, False, True, False, False, 1])

        elif board.is_capture(move):
            if board.is_en_passant(move):
                data.append([move, False, False, True, False, 0])
            else:
                score = capture_score(board, move)
                data.append([move, False, False, True, False, score])

        else:
            score = position_score(board, move)
            data.append([move, False, False, False, True, score])

    table = pd.DataFrame(data = data, columns = ["move", "TT", "check", "capture", "position", "score"])
    table = table.sort_values(by = ["TT", "check", "capture", "position", "score"], ascending = False)
    moves = table["move"].to_numpy()
    return(moves)

# Define a function to get the value of a piece based on its type
def piece_value(piece):
    """
    Returns the value of a chess piece based on its type
    Args:
        piece (chess.Piece): chess piece
    Returns:
        int: value of the chess piece
    """
    # Define the value of each chess piece for the MVV-LVA heuristic
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    return(piece_values.get(piece.piece_type, 0))

def capture_score(board, move):
    """
    Returns the value of the captured piece minus the value of the capturing piece
    Args:
        board (chess.Board): chess board
        move (chess.Move): chess move
    Returns:
        int: value of the captured piece minus the value of the capturing piece
    """
    captured_piece_value = piece_value(board.piece_at(move.to_square))
    attacking_piece_value = piece_value(board.piece_at(move.from_square))
    capture_score = captured_piece_value - attacking_piece_value
    return(capture_score)

def position_score(board, move):
    """
    Returns the position score of a move
    Args:
        board (chess.Board): chess board
        move (chess.Move): chess move
    Returns:
        int: position score of the move
    """
    position_score = 0
    if board.gives_check(move):
        position_score += 6
    if board.is_capture(move):
        position_score += 5
    if move.uci()[2] in ["d", "e"]:
        position_score += 4
    if move.uci()[2] in ["c", "f"]:
        position_score += 3
    if move.uci()[2] in ["b", "g"]:
        position_score += 2
    if move.uci()[2] in ["a", "h"]:
        position_score += 1

    return(position_score)