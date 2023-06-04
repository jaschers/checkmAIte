import tkinter as tk
import chess.svg
import io
import cairosvg
import chess.engine
from PIL import Image, ImageTk
import os
import time
import argparse
from utilities import *
from datetime import datetime
import psutil
from playsound import playsound

# get stockfish engine
stockfish_path = os.environ.get("STOCKFISHPATH")
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
parser.add_argument("-v", "--verbose", type = int, required = True, metavar = "-", help = "Verbose 0 (off) or 1 (on)")
parser.add_argument("-s", "--save", type = int, required = True, metavar = "-", help = "Save 0 (no) or 1 (yes)")
parser.add_argument("-f", "--flipped", type = int, metavar = "-", default = 0, help = "Flip board 0 (no) or 1 (yes)")

args = parser.parse_args()
##########################################################################################

class ChessApp:
    def __init__(self, master):
        self.master = master
        master.title("Chess")

        # Create canvas for displaying chess board
        self.width, self.height = 390, 390
        self.width_border, self.height_border = 14, 14# self.width / (80/3), self.height / (80/3)
        self.true_width, self.true_height = self.width - (2 * self.width_border), self.height - (2 * self.height_border)
        self.canvas = tk.Canvas(master, width = self.width, height = self.height)
        self.canvas.pack()

        # Initialize chess engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        # Draw initial chess board
        self.board = chess.Board()
        # self.board = chess.Board("8/8/8/8/5K2/8/5p1Q/4k3 w - - 2 5")
        # self.board = chess.Board("8/8/8/5K2/8/8/4kp1Q/8 w - - 0 4")
        # Create button to undo move
        self.button = tk.Button(master, text="Undo move", command=self.undo_move)
        self.button.pack()
        self.draw_board()
        playsound("sounds/start.m4a")

        self.model = models.load_model("model/model_32_8_8_depth0_mm100_ms15000_ResNet512o1_sc9000-14000_r150_rh150_rd9_rp50_exp1.h5") # model/model_40_8_8_depth0_mm100_ms15000_ResNet512_sc9000-14000_r450_exp1.h5

        # initialsie transportation table
        self.transposition_table = {}

        # empty list for ai accuracy
        self.ai_accuracy = []

        self.setup_save()

        self.ai_move()

        # Bind mouse events to canvas
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.selected_square = None
        self.selected_piece = None


    def draw_board(self):
        """
        Draw the chess board on the canvas

        Args:
            self (ChessApp): An instance of ChessApp.
        """
        # Create SVG image of chess board using chess.svg.board module
        if args.flipped == 1:
            svg_board = chess.svg.board(self.board, flipped=True).encode("utf-8")
        else:
            svg_board = chess.svg.board(self.board).encode("utf-8")

        # Convert SVG to PNG image using cairosvg library
        png_data = cairosvg.svg2png(bytestring=svg_board)

        # Display PNG image on canvas using PIL and Tkinter
        self.canvas.delete("all")
        image = Image.open(io.BytesIO(png_data))
        tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=tk_image, anchor="nw")
        self.canvas.tk_image = tk_image

        self.master.update()

    def on_click(self, event):
        """
        Handle mouse click event
        Args: 
            self (ChessApp): An instance of ChessApp.
        """
        # Get the square that was clicked on
        # consider 15x15 border of the board
        col = int((event.x - self.width_border) / (self.true_width / 8)) # 400 (chess board width) / 8 (number of squares be col)
        row = int((event.y - self.height_border) / (self.true_width / 8)) # 400 (chess board height) / 8 (number of squares be row)
        if args.flipped == 1:
            square = chess.square(7 - col, row)
        else:
            square = chess.square(col, 7 - row)

        # Get the piece that is on the clicked square
        piece = self.board.piece_at(square)

        # If there is a piece on the clicked square, select it
        if piece is not None:
            self.selected_square = square
            self.selected_piece = piece

    def on_release(self, event):
        """
        Handle mouse release event
        Args: 
            self (ChessApp): An instance of ChessApp.
        """
        # If a piece is selected, attempt to move it to the released square
        if self.selected_piece is not None:
            # Get the square that was released on
            col = int((event.x - self.width_border) / (self.true_width / 8)) # 400 (chess board width) / 8 (number of squares be col)
            row = int((event.y - self.height_border) / (self.true_width / 8)) # 400 (chess board height) / 8 (number of squares be row)
            if args.flipped == 1:
                self.release_square = chess.square(7 - col, row)
            else:
                self.release_square = chess.square(col, 7 - row)

            # Attempt to make the move
            self.move = chess.Move(self.selected_square, self.release_square)


            if self.move in self.board.legal_moves:
                self.board.push(self.move)
                self.draw_board()
                self.play_sound(board = self.board, move = self.move)
                


                if args.save == 1:
                    save_board_png(board = self.board.copy(), game_name = self.dt_string, counter = self.board_counter)
                    self.board_counter += 1

                self.check_game_over()
                self.ai_move()
            elif self.selected_piece.piece_type == chess.PAWN and (self.release_square < 8 or self.release_square > 55) and (self.board.is_check() == False):
                self.text = tk.Text(self.master, height = 1, width = 2)
                self.text.pack()
                self.button = tk.Button(self.master, text="Enter promotion piece (q/r/k/b)", command=self.promotion)
                self.button.pack()
            else:
                self.logger.info("Illegal move! Valid moves: \n %s", get_valid_moves(self.board)[1])

            self.selected_square = None
            self.selected_piece = None
    
    def undo_move(self):
        """
        Undo the last move
        Args: 
            self (ChessApp): An instance of ChessApp.
        """
        if self.board.fullmove_number < 2:
            self.logger.info("A move has to be played first")
        else:
            self.board.pop()
            self.board.pop()
            self.draw_board()
            if args.save == 1:
                delete_board_png(self.dt_string, self.board_counter - 1)
                delete_board_png(self.dt_string, self.board_counter - 2)
                self.board_counter -= 2

    def promotion(self):
        """
        Promote pawn to queen, rook, knight or bishop
        Args:
            self (ChessApp): An instance of ChessApp.
        """
        inp = self.text.get(1.0, "end-1c")
        if inp in ["q", "r", "k", "b"]:
            move_uci = self.move.uci()
            move_uci += inp
            self.move = chess.Move.from_uci(move_uci)
            self.board.push(self.move)
            self.draw_board()
            self.play_sound(board = self.board, move = self.move)
            

            if args.save == 1:
                save_board_png(board = self.board.copy(), game_name = self.dt_string, counter = self.board_counter)
                self.board_counter += 1

            self.check_game_over()
            self.button.pack_forget()
            self.text.pack_forget()
            self.ai_move()
        else:
            self.logger.info("Invalid piece type")

    def ai_move(self):
        """
        Make AI move
        Args:
            self (ChessApp): An instance of ChessApp.
        """
        # get all valid moves
        board_fen_previous = self.board.fen()
        valid_moves, valid_moves_str = get_valid_moves(self.board.copy())
        best_move_ai, _ = get_ai_move(self.board.copy(), self.model, depth = args.depth, transposition_table = self.transposition_table, verbose_minimax = True)
        self.board.push(best_move_ai)
        
        if args.save == 1:
            save_board_png(board = self.board.copy(), game_name = self.dt_string, counter = self.board_counter)
            self.board_counter += 1

        # print results
        if args.verbose == 1:
            self.board.pop()    
            best_move_stockfish, stockfish_score_stockfish_move, stockfish_moves_sorted_by_score, index = get_stockfish_move(self.board.copy(), valid_moves, valid_moves_str, best_move_ai)

            # push best stockfish move
            self.board.push(best_move_stockfish)

            # determine predicted ai score of stockfish move
            prediction_score_stockfish_move = ai_board_score_pred(self.board.copy(), self.model)

            # reset last move
            self.board.pop()

            # push best ai move
            self.board.push(best_move_ai)
            
            # determine predicted ai score of ai move
            prediction_score_ai_move = ai_board_score_pred(self.board.copy(), self.model)

            # determine stockfish score of ai move
            analyse_stockfish = engine.analyse(self.board.copy(), chess.engine.Limit(depth = 0))
            stockfish_score_ai_move = analyse_stockfish["score"].white().score(mate_score = score_max)

            self.ai_accuracy.append((1 - (index / len(stockfish_moves_sorted_by_score))) * 100)

            self.logger.info("AI / SF best move: %s / %s", best_move_ai, best_move_stockfish)
            self.logger.info("AI / SF pred. score (ai move): %s / %s", np.round(prediction_score_ai_move), stockfish_score_ai_move)
            self.logger.info("AI / SF pred. score (sf move): %s / %s", np.round(prediction_score_stockfish_move), stockfish_score_stockfish_move)
            self.logger.info("SF top 3 moves: %s", stockfish_moves_sorted_by_score[:3])
            self.logger.info("SF accuracy of AI's best move: %s", f"{index + 1} / {len(stockfish_moves_sorted_by_score)} ({np.round(self.ai_accuracy[-1], 1)} %)")
            self.logger.info("AI's mean SF accuracy: %s %%", np.round(np.mean(self.ai_accuracy), 1))
            self.logger.info("Lentgh transposition table: %s", len(self.transposition_table))
            self.logger.info("Previous board FEN: %s", board_fen_previous)
            self.logger.info("Board FEN: %s", self.board.fen())
            self.logger.info("Memory usage: %s GB", np.round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, 1))
            
        else:
            self.logger.info("AI move: %s", best_move_ai)

        self.draw_board()
        self.play_sound(board = self.board, move = best_move_ai)
        

        self.logger.info("--------------------------------------------------------------------------")
        self.check_game_over()
    
    def setup_save(self):
        """
        Setup saving of game
        Args: 
            self (ChessApp): An instance of ChessApp.
        """
        if args.save == 0:
            # setup logger
            self.logger = setup_logging(None)
        else:
            # get current date and time
            self.dt_string = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

            # save chess board as svg
            os.makedirs(f"games/{self.dt_string}", exist_ok = True)
            # setup logger
            self.logger = setup_logging(self.dt_string)
            save_board_png(board = self.board.copy(), game_name = self.dt_string, counter = 1)
            self.board_counter = 2

    def check_game_over(self):
        """
        Check if game is over
        Args:
            self (ChessApp): An instance of ChessApp.
        """
        # check if game is over
        if self.board.is_game_over():
            self.logger.info("Game over!")
            self.logger.info("%s", self.board.outcome())

            if args.save == 1:
                # save game as gif
                boards_png = [Image.open(f"games/{self.dt_string}/board{i}.png", mode='r') for i in range(1, self.board_counter)]

                save_board_gif(boards_png = boards_png, game_name = self.dt_string)

            exit()
    
    def play_sound(self, board, move):
        board_current = self.board.copy()
        board_previous = self.board.copy()
        board_previous.pop()

        if board_current.is_checkmate() or board_current.is_stalemate() or board_current.can_claim_draw():
                playsound("sounds/end.m4a")
        elif board_current.is_check():
            playsound("sounds/check.mp3")
        elif board_previous.is_capture(move):
            playsound("sounds/capture.mp3")
        elif board_previous.is_castling(move):
            playsound("sounds/castle.mp3")
        else:
            playsound("sounds/move.mp3")
        

def main():
    # Create GUI and start event loop
    root = tk.Tk()
    app = ChessApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()