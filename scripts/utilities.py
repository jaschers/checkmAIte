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

def board_score(board, depth = 15):
    """Evaluates the score of a board for player white based on stockfish.

    Args:
        board (chess.Board): chess board in FEN format
        time_limit (float, optional): maximum time allocated for the calculation. Defaults to 0.001 sec.

    Returns:
        int: stockfish score of the input board
    """
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    result = engine.analyse(board, chess.engine.Limit(depth = depth))
    score = result["score"].white().score(mate_score = 15000)
    engine.quit()
    return(score)

def boards_random(num_boards):
    """Creates random boards by playing games with random moves

    Args:
        num_boards (int): number of baords being created

    Returns:
        list: (N,) list including all the randomly generated boards N while playing the games
    """
    boards_random_fen = []
    boards_random_int = []
    boards_random_score = []

    for _ in tqdm(range(num_boards)):
        board = chess.Board()
        depth = random.randrange(1, 101) # max number of moves: 100
        for _ in range(depth):
            all_moves = list(board.legal_moves)
            random_move = random.choice(all_moves)
            board.push(random_move)
            if board.is_game_over():
                break

        boards_random_fen.append(board.copy())
        boards_random_int.append(board_int(board.copy()))
        boards_random_score.append(board_score(board.copy()))

    return(boards_random_fen, boards_random_int, boards_random_score)

def minimax(board, model, depth, alpha, beta, maximizing_player):
    """_summary_

    Args:
        board (chess.Board): chess board in FEN format
        model (_type_): _description_
        depth (_type_): _description_
        alpha (_type_): _description_
        beta (_type_): _description_
        maximizing_player (_type_): _description_
    """
    if depth == 0 or board.is_game_over() == True:
        board_int_eval = [board_int(board)]
        # board_int_eval_shape = np.shape(board_int_eval)
        # board_int_eval = np.reshape(board_int_eval, (board_int_eval_shape[0], board_int_eval_shape[1], board_int_eval_shape[2], 1))
        prediction = model.predict(board_int_eval, verbose=0)[0][0] * 15000
        return(prediction)

        # result = engine.analyse(board, chess.engine.Limit(depth = 10))
        # score_stockfish = result["score"].white().score()
        # return(score_stockfish)

    if maximizing_player == True:
        max_eval = - np.inf
        for valid_move in board.legal_moves:
            board.push(valid_move)
            eval = minimax(board, model, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(eval, max_eval)
            alpha = max(eval, alpha)
            if beta <= alpha:
                break
        return(max_eval)

    else:
        min_eval = np.inf
        for valid_move in board.legal_moves:
            board.push(valid_move)
            eval = minimax(board, model, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(eval, min_eval)
            beta = min(eval, beta)
            if beta <= alpha:
                break
        return(min_eval)