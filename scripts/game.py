import chess
from keras import models
import os
from utilities import *
import chess.svg
from chessboard import display
import sys
from datetime import datetime
import readline # allows to use arrow keys while being asked for user input
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation

stockfish_path = os.environ.get("STOCKFISHPATH")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

# get current date and time
dt_string = datetime.now().strftime("%b-%d-%Y_%H.%M.%S")

# load neural network model
model = models.load_model("model/model_24_8_8_depth0_mm100_ms10000_resnet128.h5") # score9998

# initialise game
board = chess.Board()

# save chess board as svg
os.makedirs(f"games/{dt_string}", exist_ok = True)
boardsvg = chess.svg.board(board = board)
outputfile = open(f"games/{dt_string}/board1.svg", "w")
outputfile.write(boardsvg)
outputfile.close()
os.system(f"convert -density 1200 -resize 780x780 games/{dt_string}/board1.svg games/{dt_string}/board1.png")
os.system(f"rm games/{dt_string}/board1.svg")

# display chess board
display.start(board.fen())
print("________________")

user_input = None

move_n = 2
while True:
    # check if game is over
    if board.is_game_over():
        print("Game over!")
        print(board.outcome())

        # save game as gif
        boards_png = [Image.open(f"games/{dt_string}/board{i}.png", mode='r') for i in range(1, move_n)]

        fig = plt.figure(frameon=False)
        ax  = fig.add_subplot(111)
        ims = []
        for board_png in boards_png:
            ax.axis('off')
            im = ax.imshow(board_png)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval = 1000)
        ani.save(f"games/{dt_string}/baord.gif")

        os.system(f"rm games/{dt_string}/*.png")

        break

    if user_input == "undo":
        # user move
        valid_moves = list(board.legal_moves)
        valid_moves_str = [valid_moves[i].uci() for i in range(len(valid_moves))]
        valid_moves_str.append("undo")
        print("Valid moves for user:\n", valid_moves_str)
        user_input = input("Enter your move: ")

    else:
        # get all valid moves
        valid_moves = list(board.legal_moves)
        valid_moves_str = [valid_moves[i].uci() for i in range(len(valid_moves))]

        # get best move ai        
        best_move_ai, prediction_minimax = get_ai_move(board, model, depth = 2, verbose_minimax = False)

        # get best move stockfish and ranking of all valid moves
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

        # push best stockfish move
        board.push(best_move_stockfish)

        # determine predicted ai score of stockfish move
        board_3d_int = [board_3d_attack_int(board.copy())]
        board_3d_int = np.moveaxis(board_3d_int, 1, -1)
        parameters = np.array([[np.float32(board.copy().turn), np.float32(board.copy().halfmove_clock)]])
        prediction_score_stockfish_move = model.predict([board_3d_int, parameters], verbose = 0)[0][0]  

        # reset last move
        board.pop()
        # push best ai move
        board.push(best_move_ai)

        # determine predicted ai score of ai move
        board_3d_int = [board_3d_attack_int(board.copy())]
        board_3d_int = np.moveaxis(board_3d_int, 1, -1)
        parameters = np.array([[np.float32(board.copy().turn), np.float32(board.copy().halfmove_clock)]])
        prediction_score_ai_move = model.predict([board_3d_int, parameters], verbose = 0)[0][0]  

        # determine stockfish score of ai move
        analyse_stockfish = engine.analyse(board.copy(), chess.engine.Limit(depth = 0))
        stockfish_score_ai_move = analyse_stockfish["score"].white().score(mate_score = 10000)

        # reset last move
        board.pop()

        # print results
        print("AI / SF best move:", best_move_ai, "/", best_move_stockfish)
        print("AI / SF pred. score (ai move):", np.round(prediction_score_ai_move * 20000 - 10000), "/", stockfish_score_ai_move)
        print("AI / SF pred. score (sf move):", np.round(prediction_score_stockfish_move * 20000 - 10000), "/", stockfish_score_stockfish_move)
        print("SF top 3 moves:", stockfish_moves_sorted_by_score[:3])
        print("SF ranking of AI's best move:", f"{index + 1} / {len(stockfish_moves_sorted_by_score)} ({np.round((index + 1) / len(stockfish_moves_sorted_by_score) * 100, 1)} %)")

        print("________________")
        board.push(best_move_ai)

        boardsvg = chess.svg.board(board = board)
        outputfile = open(f"games/{dt_string}/board{move_n}.svg", "w")
        outputfile.write(boardsvg)
        outputfile.close()
        os.system(f"convert -density 1200 -resize 780x780 games/{dt_string}/board{move_n}.svg games/{dt_string}/board{move_n}.png")
        os.system(f"rm games/{dt_string}/board{move_n}.svg")


        move_n += 1

        display.start(board.fen())

        # user move
        valid_moves = list(board.legal_moves)
        valid_moves_str = [valid_moves[i].uci() for i in range(len(valid_moves))]
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

    boardsvg = chess.svg.board(board = board)
    outputfile = open(f"games/{dt_string}/board{move_n}.svg", "w")
    outputfile.write(boardsvg)
    outputfile.close()
    os.system(f"convert -density 1200 -resize 780x780 games/{dt_string}/board{move_n}.svg games/{dt_string}/board{move_n}.png")
    os.system(f"rm games/{dt_string}/board{move_n}.svg")

    move_n += 1

    display.start(board.fen())


