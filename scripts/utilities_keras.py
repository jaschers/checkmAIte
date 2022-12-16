from keras.layers import Conv2D, ReLU, Add, BatchNormalization
import pandas as pd
import time
import numpy as np
import os

def ResBlock(z, kernelsizes, filters, increase_dim = False):
    # https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
    # https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
    # https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

    z_shortcut = z
    # z_shortcut = BatchNormalization()(z_shortcut)
    kernelsize_1, kernelsize_2 = kernelsizes
    filters_1, filters_2 = filters

    fz = Conv2D(filters_1, kernelsize_1)(z)
    # fz = BatchNormalization()(fz)
    fz = ReLU()(fz)

    fz = Conv2D(filters_1, kernelsize_2, padding = "same")(fz)
    # fz = BatchNormalization()(fz)
    fz = ReLU()(fz)
    
    fz = Conv2D(filters_2, kernelsize_1)(fz)
    # fz = BatchNormalization()(fz)

    if increase_dim == True:
        z_shortcut = Conv2D(filters_2, (1, 1))(z_shortcut)
        # z_shortcut = BatchNormalization()(z_shortcut)

    out = Add()([fz, z_shortcut])
    # out = BatchNormalization()(out)
    out = ReLU()(out)
    # out = MaxPooling2D(pool_size=(3, 3), strides = 1)(out)
    
    return out

def load_data(num_runs, name_data, score_cut):
    # load data
    table = pd.DataFrame()
    for run in range(num_runs):
        print(f"Loading data run {run}...")
        start = time.time()
        table_run = pd.read_hdf(f"data/3d/{name_data}/data{run}.h5", key = "table")
        middle = time.time()
        frame = [table, table_run]
        table = pd.concat(frame)
        end = time.time()
        print(f"Data run {run} loaded in {np.round(middle-start, 1)} sec...")

    if score_cut != None:
        if len(score_cut) == 1:
            table = table.reset_index(drop = True)
            table = table.drop(table[abs(table.score) >= score_cut[0]].index)
            table = table.reset_index(drop = True)
        else:
            table = table.reset_index(drop = True)
            mask = (abs(table.score) >= score_cut[0]) & (abs(table.score) <= score_cut[1])
            mask = ~mask
            table = table[mask]
            table = table.reset_index(drop = True)
    print(table)

    X_board3d = np.array(table["board3d"].values.tolist())
    X_parameter = np.array(table[["player move", "halfmove clock", "insufficient material white", "insufficient material black", "seventyfive moves", "fivefold repetition", "castling right queen side white", "castling right king side white", "castling right queen side black", "castling right king side black"]].values.tolist())
    Y = np.array(table[["score", "check", "checkmate", "stalemate"]].values.tolist())

    return(X_board3d, X_parameter, Y)

def get_extreme_predictions(prediction_val, true_score_val, X_board3d, X_parameter, name):
    difference = prediction_val - true_score_val
    indices_min = difference.argsort()[:10]
    indices_max = difference.argsort()[-10:][::-1]
    indices_zero = (np.abs(difference)).argsort()[:10]

    indices = np.concatenate((indices_min, indices_max, indices_zero))

    X_board3d_extreme, X_parameter_extreme, difference_extreme = np.array(X_board3d)[indices], np.array(X_parameter)[indices], difference[indices]

    X_board3d_extreme = np.moveaxis(X_board3d_extreme, -1, 1)

    X_board_extreme = []
    for i in range(len(X_board3d_extreme)):
        X_board_extreme.append(convert_board_int_to_fen(X_board3d_extreme[i], 12, X_parameter_extreme[i][0], None, None, X_parameter_extreme[i][1], 1))

    table_baord = pd.DataFrame({"board (FEN)": X_board_extreme})
    table_baord3d = pd.DataFrame({"board3d": list(X_board3d_extreme)})
    table_pred_val = pd.DataFrame({"predicted score": prediction_val[indices]}).reset_index(drop = True)
    table_true_val = pd.DataFrame({"true score": true_score_val[indices]}).reset_index(drop = True)
    table_difference = pd.DataFrame({"difference": difference_extreme}).reset_index(drop = True)
    table_turn = pd.DataFrame({"turn": X_parameter_extreme[:, 0]})

    table = pd.concat([table_baord, table_baord3d, table_pred_val, table_true_val, table_difference, table_turn], axis = 1)

    print(table)

    # os.makedirs(f"evaluation/{name}/examples", exist_ok = True)
    table.to_hdf(f"prediction/{name}/examples_{name}.h5", key = "table")

    # table.to_hdf(f"evaluation/{name}/examples/examples_{name}.h5", key = "table")

def convert_board_int_to_fen(board_int, number_boards_pieces, turn, castling, en_passant, halfmove_clock, fullmove_number):
    """Converts a n-dimensional list of the chess board back to its FEN format

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