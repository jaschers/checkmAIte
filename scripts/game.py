import chess
from keras import models
import os
from utilities import *
import chess.svg
from chessboard import display
import readline # allows to use arrow keys while being asked for user input

stockfish_path = os.environ.get("STOCKFISHPATH")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

# load neural network model
model = models.load_model("model/model_24_8_8_depth15_no_cm_resnet128.h5")

# initialise game
board = chess.Board()
# print(board.apply_transform(chess.flip_vertical))
# print(chess.svg.board(board, size=350))
display.start(board.fen())
print("________________")

while True:
    # check if game is over
    if board.is_game_over():
        print("Game over!")
        print(board.outcome())
        break

    # AI move
    valid_moves = list(board.legal_moves)
    valid_moves_str = [valid_moves[i].uci() for i in range(len(valid_moves))]
    print("Valid moves for AI:\n", valid_moves_str)
    valid_boards = []
    boards_parameters = []
    stockfish_scores = []
    for i in range(len(valid_moves)):
        board.push(valid_moves[i])
        valid_boards.append(board_3d_attack_int(board.copy()))
        boards_parameters.append([np.float32(board.copy().turn), np.float32(board.copy().halfmove_clock)])

        result = engine.analyse(board.copy(), chess.engine.Limit(depth = 15))
        stockfish_score = result["score"].white().score(mate_score = 11000)
        stockfish_scores.append(stockfish_score)

        board.pop()

    valid_boards = np.moveaxis(valid_boards, 1, -1)
    boards_parameters = np.array(boards_parameters)
    prediction = model.predict([valid_boards, boards_parameters], verbose=0)
    # print("AI all predicted scores: ", prediction)
    argmax_chessai = np.argmax(prediction)
    best_move_chessai = valid_moves[argmax_chessai]
    print("AI best move:", best_move_chessai)
    print("AI predicted score:", prediction[argmax_chessai][0] * 22000 - 11000)

    argmax_stockfish = np.argmax(stockfish_scores)
    best_move_stockfish = valid_moves[argmax_stockfish]
    print("Stockfish best move:", best_move_stockfish)
    print("Stockfish predicted score:", stockfish_scores[argmax_stockfish])
    print("Stockfish predicted score chessai move:", stockfish_scores[argmax_chessai])

    board.push(best_move_chessai)
    # result = engine.analyse(board, chess.engine.Limit(depth = 20))
    # score_stockfish = result["score"].white().score(mate_score = 11000)
    # print("Stockfish score:", score_stockfish)
    # print(board)
    print("________________")
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

    print(board)
    print("________________")
    display.start(board.fen())


