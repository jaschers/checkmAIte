import chess
import chess.engine
import numpy as np
from stockfish import Stockfish
import random
from tqdm import tqdm
import os 

stockfish_path = os.environ.get("STOCKFISHPATH")

# allocate stockfish engine and specify parameters
stockfish = Stockfish(stockfish_path)
stockfish.set_depth(20)
stockfish.set_skill_level(20)

best_move = None

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
    result = engine.analyse(board, chess.engine.Limit(depth = depth))
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


def minimax(board, model, depth, alpha, beta, maximizing_player, verbose_minimax = False):
    if depth < 0 or type(depth) != int:
        raise ValueError("Depth needs to be int and greater than 0")

    if depth == 0 or board.is_game_over() == True:
        board_int_eval = [board_3d_attack_int(board)]
        board_int_eval = np.moveaxis(board_int_eval, 1, -1)
        parameters = np.array([[np.float32(board.turn), np.float32(board.halfmove_clock)]])
        prediction = model.predict([board_int_eval, parameters], verbose = 0)[0][0]
        # print("1", prediction, best_move)
        if verbose_minimax == True:
            print(board)
            print("_____________")
        return(prediction)

    if maximizing_player == True:
        # print("maximizing_player == True", f", depth = {depth}")
        max_eval = - np.inf
        for valid_move in board.legal_moves:
            board.push(valid_move)
            eval = minimax(board, model, depth - 1, alpha, beta, False, verbose_minimax)
            board.pop()
            if eval > max_eval:
                max_eval = eval
            alpha = max(eval, alpha)
            if beta <= alpha:
                break
        # print("2", max_eval, best_move)
        if verbose_minimax == True:
            print(board)
            print("_____________")
        return(max_eval)

    else:
        # print("maximizing_player == False", f", depth = {depth}")
        min_eval = np.inf
        for valid_move in board.legal_moves:
            board.push(valid_move)
            eval = minimax(board, model, depth - 1, alpha, beta, True, verbose_minimax)
            board.pop()
            if eval < min_eval:
                min_eval = eval
            beta = min(eval, beta)
            if beta <= alpha:
                break
        #print("3", min_eval, best_move)
        if verbose_minimax == True:
            print(board)
            print("_____________")
        return(min_eval)


def get_ai_move(board, model, depth, verbose_minimax):
    max_move = None
    max_eval = -np.inf

    for valid_move in board.legal_moves:
        board.push(valid_move)
        eval = minimax(board, model, depth = depth - 1, alpha = -np.inf, beta = np.inf, maximizing_player = False, verbose_minimax = verbose_minimax)
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