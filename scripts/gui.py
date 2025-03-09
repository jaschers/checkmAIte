import tkinter as tk
import chess
from PIL import Image, ImageTk
import os

class ChessGUI:
    def __init__(self, square_size=60):
        self.square_size = square_size
        self.board_size = 8
        self.light_color = "#F0D9B5"
        self.dark_color = "#B58863"
        self.piece_images = {}
        self.selected_piece_square = None  # Track the selected piece
        self.dragged_piece = None  # Track the piece being dragged
        self.drag_start_pos = None  # Track the starting position of the drag

        self.window = tk.Tk()
        self.window.title("Chess GUI")

        self.canvas = tk.Canvas(
            self.window,
            width=self.board_size * self.square_size,
            height=self.board_size * self.square_size
        )
        self.canvas.pack()

        self.board = chess.Board()
        self.load_piece_images()
        self.draw_board()
        self.draw_pieces()

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_square_click)
        self.canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drop)

    def load_piece_images(self):
        piece_folder = "visuals/development/chess_pieces"
        piece_map = {
            "P": "Chess_plt45.png",
            "N": "Chess_nlt45.png",
            "B": "Chess_blt45.png",
            "R": "Chess_rlt45.png",
            "Q": "Chess_qlt45.png",
            "K": "Chess_klt45.png",
            "p": "Chess_pdt45.png",
            "n": "Chess_ndt45.png",
            "b": "Chess_bdt45.png",
            "r": "Chess_rdt45.png",
            "q": "Chess_qdt45.png",
            "k": "Chess_kdt45.png",
        }

        for symbol, filename in piece_map.items():
            path = os.path.join(piece_folder, filename)
            image = Image.open(path)

            # Resize each image to fit within the square size
            image = image.resize((self.square_size, self.square_size), Image.Resampling.LANCZOS)
            
            # Store the resized image
            self.piece_images[symbol] = ImageTk.PhotoImage(image)

    def draw_board(self):
        for row in range(self.board_size):
            for col in range(self.board_size):
                color = self.light_color if (row + col) % 2 == 0 else self.dark_color
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                image = self.piece_images[symbol]
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)  # Invert for GUI coordinates
                x = col * self.square_size
                y = row * self.square_size
                self.canvas.create_image(x, y, anchor=tk.NW, image=image)

    def draw_move_highlights(self, possible_moves):
        """Draw the highlighted squares where the selected piece can move"""
        for move in possible_moves:
            target_square = move.to_square
            col = chess.square_file(target_square)
            row = 7 - chess.square_rank(target_square)  # Invert for GUI coordinates
            x = col * self.square_size + self.square_size / 2
            y = row * self.square_size + self.square_size / 2
            self.canvas.create_oval(
                x - self.square_size / 8,
                y - self.square_size / 8,
                x + self.square_size / 8,
                y + self.square_size / 8,
                fill="grey", outline="grey", width=2
            )

    def on_square_click(self, event):
        """Handle square clicks to select a piece and highlight possible moves"""
        col = event.x // self.square_size
        row = event.y // self.square_size
        clicked_square = chess.square(col, 7 - row)  # Invert row to match GUI coordinates

        piece = self.board.piece_at(clicked_square)

        # If a piece is clicked, highlight its possible moves
        if piece:
            self.selected_piece_square = clicked_square
            possible_moves = [move for move in self.board.legal_moves if move.from_square == clicked_square]
            self.draw_board()  # Redraw the board
            self.draw_pieces()  # Redraw the pieces
            self.draw_move_highlights(possible_moves)  # Highlight possible moves

    def on_drag_motion(self, event):
        """Handle the motion of the dragged piece"""
        if self.selected_piece_square:
            # Calculate the new position for the dragged piece
            col = event.x // self.square_size
            row = event.y // self.square_size
            self.dragged_piece = self.canvas.create_image(
                event.x, event.y, anchor=tk.NW, image=self.piece_images[self.board.piece_at(self.selected_piece_square).symbol()]
            )

    def on_drop(self, event):
        """Handle the drop of the dragged piece"""
        if self.selected_piece_square:
            col = event.x // self.square_size
            row = event.y // self.square_size
            target_square = chess.square(col, 7 - row)

            # Check if the move is legal
            move = chess.Move(self.selected_piece_square, target_square)
            if move in self.board.legal_moves:
                self.board.push(move)  # Make the move on the board
                self.selected_piece_square = None  # Reset selected piece
                self.dragged_piece = None  # Remove dragged piece

                self.draw_board()  # Redraw the board
                self.draw_pieces()  # Redraw the pieces

            else:
                # If the move is illegal, just reset the dragged piece
                self.selected_piece_square = None
                self.canvas.delete(self.dragged_piece)
                self.dragged_piece = None

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()