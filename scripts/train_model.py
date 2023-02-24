import numpy as np
from utilities_keras import *
from tensorflow.keras.optimizers import Adam
from keras import models, optimizers
from keras.layers import Conv2D, GlobalMaxPooling2D, Dense, Flatten, Input, BatchNormalization, Add, Activation, Concatenate, Dropout
# from keras.applications import EfficientNetV2L, ResNet152V2, VGG19, EfficientNetV2M, EfficientNetV2S, ResNet50, MobileNet, InceptionV3, EfficientNetV2B0
import keras.utils as utils
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import os
import time
import argparse
import sys
import psutil
# np.set_printoptions(threshold=sys.maxsize)

######################################## argparse setup ########################################
script_descr="""
Trains the residual neural network
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-r", "--runs", type = int, required = False, metavar = "-", help = "Number of runs used for training")
parser.add_argument("-nd", "--name_data", type = str, required = False, metavar = "-", default = "40_8_8_depth0_mm100_ms15000", help = "name of the data folder")
parser.add_argument("-sc", "--score_cut", type = float, required = False, nargs='+', metavar = "-", default = None, help = "score cut applied on the data (default: None)")
parser.add_argument("-ne", "--name_experiment", type = str, required = True, metavar = "-", help = "Name of this particular experiment")
parser.add_argument("-e", "--epochs", type = int, required = False, metavar = "-", default = 1000, help = "Number of epochs for the training")
parser.add_argument("-bs", "--batch_size", type = int, required = False, metavar = "-", default = 1024, help = "Batch size used for training")
parser.add_argument("-g", "--generator", type = str, required = False, metavar = "-", default = "y", help = "Use of generator for training (required for large data sets and limited RAM)")

args = parser.parse_args()
##########################################################################################

# load data
print("Memory usage before loading data:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MiB")

if args.generator == "n":
    X_board3d, X_parameter, Y = load_data(args.runs, args.name_data, args.score_cut)

    X_board3d = np.moveaxis(X_board3d, 1, -1)
    X_board3d_shape = np.shape(X_board3d)
    X_parameter_shape = np.shape(X_parameter)

    print("X board3d shape:", np.shape(X_board3d))
    print("X parameter shape:", np.shape(X_parameter))
    print("Y score shape:", np.shape(Y))

    # norm Y data between -1 and 1
    print("Y_score_min, Y_score_max before normalisation:", np.min(Y[:,0]), np.max(Y[:,0]))
    print("Y_score unique", np.unique(Y[:,0]))
    Y = Y.astype("float")
    Y[:,0] = Y[:,0] - np.min(Y[:,0])
    Y[:,0] = Y[:,0] / np.max(Y[:,0])
    print("Y_score_min, Y_score_max after normalisation:", np.min(Y[:,0]), np.max(Y[:,0]))
    # Y = Y[:,0]

    X_board3d_train, X_board3d_val, X_board3d_test = np.split(X_board3d, [-int(len(X_board3d) / 5), -int(len(X_board3d) / 10)]) 
    X_parameter_train, X_parameter_val, X_parameter_test = np.split(X_parameter, [-int(len(X_parameter) / 5), -int(len(X_parameter) / 10)]) 
    Y_train, Y_val, Y_test = np.split(Y, [-int(len(Y) / 5), -int(len(Y) / 10)]) 

    print("Number of training, validation and test data:", len(X_board3d_train), len(X_board3d_val), len(X_board3d_test))

    print("Memory usage after loading data:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MiB")

elif args.generator == "y":
    # load data
    print("in elif args.generator")
    table_dd = dd.read_hdf(os.path.join("data", "3d", args.name_data, "data45*.h5"), key = "table")
    # table_dd = dd.read_hdf("data/3d/40_8_8_depth0_mm100_ms15000_dd/data0.h5", key = "table")

    # chunksize = args.batch_size * 1

    # get total dataset_size after score cut
    dataset_size = len(table_dd.index)
    print("Total number of boards:", dataset_size)

    # apply score cut
    if args.score_cut != None:
        if len(args.score_cut) == 1:
            mask = (abs(table_dd.score) >= args.score_cut[0])
        else:
            mask = (abs(table_dd.score) >= args.score_cut[0]) & (abs(table_dd.score) <= args.score_cut[1])
        mask = ~mask
        table_dd = table_dd[mask]
        table_dd = table_dd.reset_index(drop = True)

    # get total dataset_size after score cut
    dataset_size = len(table_dd.index)
    print("Total number of boards after score cut:", dataset_size)

    # norm score values between 0 and 1
    table_dd = table_dd.astype({"score": "float"})

    score_max = 15000
    table_dd["score"] = table_dd["score"] + score_max
    table_dd["score"] = table_dd["score"] / (2 * score_max)

    dataset_split_frac = np.array([0.8, 0.1, 0.1])
    batch_size = args.batch_size #1024 # 32

    table_dd_train, table_dd_val, table_dd_test = table_dd.random_split(dataset_split_frac)  
    # print("Total number of train/val/test boards after score cut:", dataset_size * dataset_split_frac)

    board3d_columns = [f"sq{counter}" for counter in range(2560)]
    X_board3d_train, X_board3d_val, X_board3d_test = table_dd_train[board3d_columns].to_dask_array(), table_dd_val[board3d_columns].to_dask_array(), table_dd_test[board3d_columns].to_dask_array()

    X_board3d_train.compute_chunk_sizes()
    X_board3d_val.compute_chunk_sizes()
    X_board3d_test.compute_chunk_sizes()

    print("X_board3d_train chunks before reshape", X_board3d_train.chunks)

    X_board3d_train, X_board3d_val, X_board3d_test = X_board3d_train.reshape(-1, 40, 8, 8), X_board3d_val.reshape(-1, 40, 8, 8), X_board3d_test.reshape(-1, 40, 8, 8)
    X_board3d_train, X_board3d_val, X_board3d_test = np.moveaxis(X_board3d_train, 1, -1), np.moveaxis(X_board3d_val, 1, -1), np.moveaxis(X_board3d_test, 1, -1)

    print("X_board3d_train chunks after reshape", X_board3d_train.chunks)
    print("X_board3d_val chunks after reshape", X_board3d_val.chunks)
    print("X_board3d_test chunks after reshape", X_board3d_test.chunks)

    # X_board3d_train = X_board3d_train.rechunk({0: chunksize})
    # X_board3d_val = X_board3d_val.rechunk({0: chunksize})

    # print("X_board3d_train chunks after rechunk", X_board3d_train.chunks)

    train_size, val_size, test_size = len(X_board3d_train), len(X_board3d_val), len(X_board3d_test)
    steps_per_epoch = np.floor(np.array([train_size, val_size, test_size]) / batch_size)
    print("steps_per_epoch", steps_per_epoch)
    
    print("Total number of train/val/test boards after score cut:", train_size, val_size, test_size)

    X_parameter_train = table_dd_train[["player move", "halfmove clock", "insufficient material white", "insufficient material black", "seventyfive moves", "fivefold repetition", "castling right queen side white", "castling right king side white", "castling right queen side black", "castling right king side black"]].to_dask_array()
    X_parameter_val = table_dd_val[["player move", "halfmove clock", "insufficient material white", "insufficient material black", "seventyfive moves", "fivefold repetition", "castling right queen side white", "castling right king side white", "castling right queen side black", "castling right king side black"]].to_dask_array()
    X_parameter_test = table_dd_test[["player move", "halfmove clock", "insufficient material white", "insufficient material black", "seventyfive moves", "fivefold repetition", "castling right queen side white", "castling right king side white", "castling right queen side black", "castling right king side black"]].to_dask_array()

    X_parameter_train.compute_chunk_sizes()
    X_parameter_val.compute_chunk_sizes()
    X_parameter_test.compute_chunk_sizes()

    # X_parameter_train = X_parameter_train.rechunk({0: chunksize})
    # X_parameter_val = X_parameter_val.rechunk({0: chunksize})

    print("X_parameter_train chunks", X_parameter_train.chunks)
    print("X_parameter_val chunks", X_parameter_val.chunks)
    print("X_parameter_test chunks", X_parameter_test.chunks)

    Y_train = table_dd_train[["score", "check", "checkmate", "stalemate"]].to_dask_array()
    Y_val = table_dd_val[["score", "check", "checkmate", "stalemate"]].to_dask_array()
    Y_test = table_dd_test[["score", "check", "checkmate", "stalemate"]].to_dask_array()

    Y_train.compute_chunk_sizes()
    Y_val.compute_chunk_sizes()
    Y_test.compute_chunk_sizes()

    # Y_train = Y_train.rechunk({0: chunksize})
    # Y_val = Y_val.rechunk({0: chunksize})

    print("Y_train chunks", Y_train.chunks)
    print("Y_val chunks", Y_val.chunks)
    print("Y_test chunks", Y_test.chunks)

    print("Memory usage after loading data:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MiB")

    # hard coded - needs to be updated
    X_board3d_shape = (0, 8, 8, 40)
    X_parameter_shape = (0, 10)
    board3d_columns = [f"sq{counter}" for counter in range(2560)]



# # # define model
# ############################################################################
# model_input_board3d = Input(shape = X_board3d_shape[1:])
# model_board3d = Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(model_input_board3d)
# model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
# model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
# model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
# model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
# model_board3d = Flatten()(model_board3d)

# model_input_parameter = Input(shape = X_parameter_shape[1:])
# # model_parameter = Dense(128, activation = "relu")(model_input_parameter)

# model = Concatenate()([model_board3d, model_input_parameter])

# model = Dense(128, activation = "relu")(model)
# model = Dense(64, activation = "relu")(model)
# model = Dense(4, name = "score-check-checkmate-stalemate")(model)
# # model = Dense(1, name = "score-check-checkmate-stalemate")(model)
# ############################################################################

######################################################
model_input_board3d = Input(shape = X_board3d_shape[1:])
model_board3d = Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "same")(model_input_board3d)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 256], increase_dim = True)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 256])
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
model_board3d = Flatten()(model_board3d)

model_input_parameter = Input(shape = X_parameter_shape[1:])
# model_parameter = Dense(128, activation = "relu")(model_input_parameter)

model = Concatenate()([model_board3d, model_input_parameter])

model = Dense(256, activation = "relu")(model)
# model = Dropout(0.05)(model)
model = Dense(128, activation = "relu")(model)
# model = Dropout(0.05)(model)
model = Dense(4, name = "score-check-checkmate-stalemate")(model)
# model = Dense(1, name = "score-check-checkmate-stalemate")(model)
############################################################################

# ############################################################################
# model_input_board3d = Input(shape = X_board3d_shape[1:])

# model_board3d = EfficientNetV2S(weights = None, include_top = True, classes = 256, classifier_activation = "relu", input_tensor = model_input_board3d) #, input_shape = (8, 8, 34))

# model_input_parameter = Input(shape = X_parameter_shape[1:])
# # model_parameter = Dense(128, activation = "relu")(model_input_parameter)

# print("______________________")
# print(model_board3d)
# print(model_board3d.output)
# print(model_input_parameter)
# print("______________________")

# model = Concatenate()([model_board3d.output, model_input_parameter])

# model = Dense(128, activation = "relu")(model)
# # model = Dense(64, activation = "relu")(model)
# model = Dense(4, name = "score-check-checkmate-stalemate")(model)

# model = models.Model(inputs = [model_input_board3d, model_input_parameter], outputs = model)

# model.summary()

# model.compile(optimizer=optimizers.Adam(5e-4), loss="mse") # mean_squared_logarithmic_error
# checkpointer = ModelCheckpoint(filepath = f"model/model_{args.name_experiment}.h5", verbose = 1, save_best_only = True)
# os.makedirs("history/", exist_ok = True)
# model.fit([X_board3d_train, X_parameter_train], Y_train, epochs = args.epochs, batch_size = 32, validation_data=([X_board3d_val, X_parameter_val], Y_val), callbacks = [checkpointer, CSVLogger(f"history/history_{args.name_experiment}.csv"), ReduceLROnPlateau(monitor="val_loss", patience=20, min_delta=1e-7), EarlyStopping(monitor="val_loss", patience=40, min_delta=1e-7)], verbose = 2)
# ############################################################################

model = models.Model(inputs = [model_input_board3d, model_input_parameter], outputs = model)
model.summary()

os.makedirs("model/", exist_ok = True)

# compile model
# model.compile(optimizer=optimizers.Adam(5e-4), loss="mse")
# model.compile(optimizer = "adam", loss = "mse", learning_rate = 1e-4) # mean_squared_logarithmic_error
model.compile(optimizer = Adam(learning_rate = 1e-4), loss="mse") # mean_squared_logarithmic_error
checkpointer = ModelCheckpoint(filepath = f"model/model_{args.name_experiment}.h5", verbose = 1, save_best_only = True)
os.makedirs("history/", exist_ok = True)

if args.generator == "n":
    model.fit([X_board3d_train, X_parameter_train], Y_train, epochs = args.epochs, batch_size = args.batch_size, validation_data = ([X_board3d_val, X_parameter_val], Y_val), callbacks = [checkpointer, CSVLogger(f"history/history_{args.name_experiment}.csv"), ReduceLROnPlateau(monitor = "val_loss", patience = 20, min_delta = 1e-7), EarlyStopping(monitor = "val_loss", patience = 40, min_delta = 1e-7)], verbose = 2)
elif args.generator == "y":
    # model.fit(dask_data_generator(X_board3d_train, X_parameter_train, Y_train, train_size, batch_size), steps_per_epoch = steps_per_epoch[0], epochs = args.epochs, validation_data = dask_data_generator(X_board3d_val, X_parameter_val, Y_val, val_size, batch_size), validation_steps = steps_per_epoch[1], callbacks = [checkpointer, CSVLogger(f"history/history_{args.name_experiment}.csv"), ReduceLROnPlateau(monitor = "val_loss", patience = 20, min_delta = 1e-7), EarlyStopping(monitor = "val_loss", patience = 40, min_delta = 1e-7)], verbose = 1)
    model.fit(dask_data_generator(X_board3d_train, X_parameter_train, Y_train, train_size, batch_size), steps_per_epoch = steps_per_epoch[0], epochs = args.epochs, validation_data = dask_data_generator(X_board3d_val, X_parameter_val, Y_val, val_size, batch_size), validation_steps = steps_per_epoch[1], verbose = 1)


# model.save(f"model/model_{args.name_experiment}.h5")

# # save model predictions on training an validation data
# os.makedirs("prediction/", exist_ok = True)

# # prediction_train = model.predict([X_board3d_train, X_parameter_train])
# # prediction_train = np.reshape(prediction_train, (np.shape(prediction_train)[0]))

# prediction_val = model.predict([X_board3d_val, X_parameter_val])
# prediction_val = np.reshape(prediction_val, (np.shape(prediction_val)[0]))

# # table_pred_train = pd.DataFrame({"prediction": prediction_train})
# # table_true_train = pd.DataFrame({"true score": Y_train})

# table_pred_val = pd.DataFrame({"prediction": prediction_val})
# table_true_val = pd.DataFrame({"true score": Y_val})

# # table_pred_train = pd.concat([table_pred_train, table_true_train], axis = 1)
# table_pred_val = pd.concat([table_pred_val, table_true_val], axis = 1)

# # print(table_pred_train)
# print(table_pred_val)

# # table_pred_train.to_hdf(f"prediction/prediction_train_{args.name_experiment}.h5", key = "table")
# table_pred_val.to_hdf(f"prediction/prediction_val_{args.name_experiment}.h5", key = "table")