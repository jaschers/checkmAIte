import chess
import chess.engine
import os
from utilities import *
import chess.svg
from chessboard import display
import readline # allows to use arrow keys while being asked for user input
from stockfish import Stockfish

# allocate stockfish engine and specify parameters
stockfish = Stockfish("/usr/local/Cellar/stockfish/15/bin/stockfish")
engine = chess.engine.SimpleEngine.popen_uci("/usr/local/Cellar/stockfish/15/bin/stockfish")
stockfish.set_depth(20)
stockfish.set_skill_level(20)

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

    # stockfish move
    valid_moves = list(board.legal_moves)
    scores = []
    for i in range(len(valid_moves)):
        board.push(valid_moves[i])
        # valid_boards.append(board_int(board.copy()))
        result = engine.analyse(board, chess.engine.Limit(time = 0.01))
        score = result['score'].white().score()
        scores.append(score)
        board.pop()
    scores = [0 if i is None else i for i in scores] # replace None with 0
    argmax = np.argmax(scores)
    best_move = valid_moves[argmax]
    print("AI move:", best_move)
    board.push(best_move)
    print(board)
    print("________________")
    display.start(board.fen())

    # user move
    valid_moves = list(board.legal_moves)
    valid_moves_str = [valid_moves[i].uci() for i in range(len(valid_moves))]
    valid_moves_str.append("undo")
    print("Valid moves:\n", valid_moves_str)
    user_input = input("Enter your move: ")

    while not (user_input in valid_moves_str):
        print("Invalid move!")
        print("Valid moves:\n", valid_moves_str)
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


