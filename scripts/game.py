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
model = models.load_model("model/model.h5")

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
    for i in range(len(valid_moves)):
        board.push(valid_moves[i])
        valid_boards.append(board_int(board.copy()))
        board.pop()
    prediction = model.predict(valid_boards, verbose=0) * -15000
    print("AI all predicted scores: ", prediction)
    argmax = np.argmax(prediction)
    best_move = valid_moves[argmax]
    print("AI move:", best_move)
    print("AI predicted score:", prediction[argmax])
    board.push(best_move)
    result = engine.analyse(board, chess.engine.Limit(depth = 20))
    score_stockfish = result["score"].white().score()
    print("Stockfish score:", score_stockfish)
    print(board)
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


