import chess
from keras import models
import os
from utilities import *
import chess.svg
from chessboard import display
from datetime import datetime
import readline # allows to use arrow keys while being asked for user input
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
# import time

# get stockfish engine
stockfish_path = os.environ.get("STOCKFISHPATH")
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
score_max = 15000

# avoid printing tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

######################################## argparse setup ########################################
# add script description
script_descr="""
Plays a game against the AI
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-d", "--depth", type = int, required = True, metavar = "-", help = "Depth of the minimax algorithm")
parser.add_argument("-v", "--verbose", type = bool, required = True, metavar = "-", help = "Verbose True or False")

args = parser.parse_args()
##########################################################################################

# get current date and time
dt_string = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

# load neural network model
model = models.load_model("model/model_34_8_8_depth0_mm100_ms15000_ResNet512_sc9000-14000_r150_exp3.h5") # model_24_8_8_depth0_mm100_ms10000_s9900_resnet128_lossmsle_exp1

# initialise game
board = chess.Board()

# save chess board as svg
os.makedirs(f"games/{dt_string}", exist_ok = True)
save_board_png(board = board.copy(), game_name = dt_string, counter = 1)

# display chess board
display.start(board.fen())
print("________________")

user_input = None

board_counter = 2
while True:
    # check if game is over
    if board.is_game_over():
        print("Game over!")
        print(board.outcome())

        # save game as gif
        boards_png = [Image.open(f"games/{dt_string}/board{i}.png", mode='r') for i in range(1, board_counter)]

        save_baord_gif(boards_png = boards_png, game_name = dt_string)

        break

    if user_input == "undo":
        # user move
        valid_moves, valid_moves_str = get_valid_moves(board.copy())
        valid_moves_str.append("undo")
        print("Valid moves for user:\n", valid_moves_str)
        user_input = input("Enter your move: ")

    else:
        # get all valid moves
        valid_moves, valid_moves_str = get_valid_moves(board.copy())

        # get best move ai
        # start_time = time.time()
        best_move_ai, prediction_minimax = get_ai_move(board.copy(), model, depth = args.depth, verbose_minimax = False)
        # end_time = time.time()
        # print(end_time - start_time)
        # get best move stockfish and ranking of all valid moves
        best_move_stockfish, stockfish_score_stockfish_move, stockfish_moves_sorted_by_score, index = get_stockfish_move(board.copy(), valid_moves, valid_moves_str, best_move_ai)

        # push best stockfish move
        board.push(best_move_stockfish)

        # determine predicted ai score of stockfish move
        prediction_score_stockfish_move = ai_board_score_pred(board.copy(), model)

        # reset last move
        board.pop()

        # push best ai move
        board.push(best_move_ai)
        
        # determine predicted ai score of ai move
        prediction_score_ai_move = ai_board_score_pred(board.copy(), model)

        # determine stockfish score of ai move
        analyse_stockfish = engine.analyse(board.copy(), chess.engine.Limit(depth = 0))
        stockfish_score_ai_move = analyse_stockfish["score"].white().score(mate_score = score_max)

        # reset last move
        board.pop()

        # print results
        print("AI / SF best move:", best_move_ai, "/", best_move_stockfish)
        print("AI / SF pred. score (ai move):", np.round(prediction_score_ai_move), "/", stockfish_score_ai_move)
        print("AI / SF pred. score (sf move):", np.round(prediction_score_stockfish_move), "/", stockfish_score_stockfish_move)
        print("SF top 3 moves:", stockfish_moves_sorted_by_score[:3])
        print("SF ranking of AI's best move:", f"{index + 1} / {len(stockfish_moves_sorted_by_score)} ({np.round((index + 1) / len(stockfish_moves_sorted_by_score) * 100, 1)} %)")

        print("________________")
        board.push(best_move_ai)

        save_board_png(board = board.copy(), game_name = dt_string, counter = board_counter)

        board_counter += 1

        display.start(board.fen())

        if board.is_game_over():
            print("Game over!")
            print(board.outcome())

            # save game as gif
            boards_png = [Image.open(f"games/{dt_string}/board{i}.png", mode='r') for i in range(1, board_counter)]

            save_baord_gif(boards_png = boards_png, game_name = dt_string)

            break

        # user move
        valid_moves, valid_moves_str = get_valid_moves(board.copy())
        valid_moves_str.append("undo")
        print("Valid moves for user:\n", valid_moves_str)
        user_input = input("Enter your move: ")

    while not (user_input in valid_moves_str):
        print("Invalid move!")
        print("Valid moves for user:\n", valid_moves_str)
        user_input = input("Enter your move: ")

    if user_input == "undo":
        board.pop()
        board.pop()
    else:
        user_move = chess.Move.from_uci(user_input)
        board.push(user_move)

    print("________________")

    save_board_png(board = board.copy(), game_name = dt_string, counter = board_counter)

    board_counter += 1

    display.start(board.fen())


