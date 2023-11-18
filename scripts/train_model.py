import numpy as np
from utilities_keras import *
from tensorflow.keras.optimizers import Adam
from keras import models, optimizers
from keras.layers import Conv2D, GlobalMaxPooling2D, Dense, Flatten, Input, BatchNormalization, Add, Activation, Concatenate, Dropout, ReLU, MaxPooling2D, PReLU, ELU, LeakyReLU, Softmax
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
import glob
import itertools
import tensorflow as tf

# np.set_printoptions(threshold=sys.maxsize)

######################################## argparse setup ########################################
script_descr="""
Trains the residual neural network
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-r", "--runs", type = int, required = True, metavar = "-", help = "Number of runs used for training")
parser.add_argument("-nd", "--name_data", type = str, required = False, metavar = "-", default = "30_8_8_depth0+5", help = "name of the data folder")
parser.add_argument("-sc", "--score_cut", type = float, required = False, nargs='+', metavar = "-", default = None, help = "score cut applied on the data (default: None)")
parser.add_argument("-ne", "--name_experiment", type = str, required = True, metavar = "-", help = "Name of this particular experiment")
parser.add_argument("-rh", "--read_human", type = int, required = True, metavar = "-", help = "Number of human data runs used for training")
parser.add_argument("-rd", "--read_draw", type = str, required = False, metavar = "-", default = "y", help = "Should draw data be read? [y/n], default: y")
parser.add_argument("-rp", "--read_pinned", type = str, required = False, metavar = "-", default = "y", help = "Should pinned checkmate data be read? [y/n], default: y")
parser.add_argument("-e", "--epochs", type = int, required = False, metavar = "-", default = 1000, help = "Number of epochs for the training")
parser.add_argument("-bs", "--batch_size", type = int, required = False, metavar = "-", default = 32, help = "Batch size used for training")
parser.add_argument("-af", "--activation_function", type = str, required = False, metavar = "-", default = "relu", help = "Activation function used for the model")
parser.add_argument("-l", "--loss_function", type = str, required = False, metavar = "-", default = "mse", help = "Loss function used for training")
parser.add_argument("-dr", "--dropout", type = float, required = False, metavar = "-", default = 0, help = "Dropout used for training")
parser.add_argument("-g", "--generator", type = str, required = False, metavar = "-", default = "n", help = "Use of generator for training (required for large data sets and limited RAM)")
parser.add_argument("-v", "--verbose", type = int, required = False, metavar = "-", default = 2, help = "verbose level for training [0, 1, 2], default: 2")

args = parser.parse_args()
##########################################################################################
max_score = 15000
num_boards_per_file = 10000
dir_af = {"relu": ReLU(), "leakyrelu": LeakyReLU(), "elu": ELU(), "softmax": Softmax()}

# load data
print("Memory usage before loading data:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MiB")

if args.generator == "n":
    X_board3d, X_parameter, Y = load_data(args.runs, args.name_data, args.score_cut, args.read_human, args.read_draw, args.read_pinned)

    # X_board3d = np.concatenate((X_board3d[:,:24], X_board3d[:,36:]), axis = 1)

    X_board3d = np.moveaxis(X_board3d, 1, -1)
    X_board3d_shape = np.shape(X_board3d)
    X_parameter_shape = np.shape(X_parameter)

    # X_board3d = np.moveaxis(X_board3d, -1, 1)

    # print("X board3d shape:", np.shape(X_board3d))
    # print("X parameter shape:", np.shape(X_parameter))
    # print("Y score shape:", np.shape(Y))

    # print(np.shape(X_board3d[0]))
    # print(np.shape(X_parameter[0]))
    # print(convert_board_int_to_fen(X_board3d[0], 12, X_parameter[0][0], None, None, X_parameter[0][1], 1))
    # exit()


    # norm Y data between -1 and 1
    print("Y_score_min, Y_score_max before normalisation:", np.min(Y[:,0]), np.max(Y[:,0]))
    print("Y_score unique", np.unique(Y[:,0]))
    Y = Y.astype("float")
    Y[:,0] = Y[:,0] + max_score
    Y[:,0] = Y[:,0] / (2 * max_score)
    print("Y_score_min, Y_score_max after normalisation:", np.min(Y[:,0]), np.max(Y[:,0]))
    # Y = Y[:,0]

    X_board3d_train, X_board3d_val, X_board3d_test = np.split(X_board3d, [-int(len(X_board3d) / 5), -int(len(X_board3d) / 10)]) 
    X_parameter_train, X_parameter_val, X_parameter_test = np.split(X_parameter, [-int(len(X_parameter) / 5), -int(len(X_parameter) / 10)]) 
    Y_train, Y_val, Y_test = np.split(Y, [-int(len(Y) / 5), -int(len(Y) / 10)]) 

    print("Number of training, validation and test data:", len(X_board3d_train), len(X_board3d_val), len(X_board3d_test))

    print("Memory usage after loading data:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MiB")

elif args.generator == "y":
    # TO DO: add shuffle
    dir_random = f"data/3d/{args.name_data}_mm100_ms15000/"
    filenames_random = glob.glob(dir_random + "*")
    filenames_random = np.random.choice(filenames_random, args.runs)
    filenames_random_train, filenames_random_test, filenames_random_val = np.split(filenames_random, [int(len(filenames_random) * 0.8), int(len(filenames_random) * 0.9)])

    dir_human = f"data/3d/{args.name_data}_ms15000_human/"
    filenames_human = glob.glob(dir_human + "*")
    filenames_human = np.random.choice(filenames_human, args.read_human)
    filenames_human_train, filenames_human_test, filenames_human_val = np.split(filenames_human, [int(len(filenames_human) * 0.8), int(len(filenames_human) * 0.9)])

    if args.read_draw == "y":
        dir_draw = "data/3d/30_8_8_draw/"
        filenames_draw = glob.glob(dir_draw + "*")
        filenames_draw = np.sort(filenames_draw)
        filenames_draw_train = filenames_draw[0::3]
        filenames_draw_test = filenames_draw[1::3]
        filenames_draw_val = filenames_draw[2::3]

    else:
        filenames_draw_train, filenames_draw_val, filenames_draw_test = [], [], []
    
    if args.read_pinned == "y":
        dir_pinned = "data/3d/30_8_8_pinned_checkmate/"
        filenames_pinned = glob.glob(dir_pinned + "*")
        filenames_pinned = np.sort(filenames_pinned)
        filenames_pinned_train = filenames_pinned[0::3]
        filenames_pinned_test = filenames_pinned[1::3]
        filenames_pinned_val = filenames_pinned[2::3]
    else:
        filenames_pinned_train, filenames_pinned_val, filenames_pinned_test = [], [], []

    filenames_train = np.concatenate((filenames_random_train, filenames_human_train, filenames_draw_train, filenames_pinned_train), axis = 0)
    filenames_val = np.concatenate((filenames_random_val, filenames_human_val, filenames_draw_val, filenames_pinned_val), axis = 0)
    filenames_test = np.concatenate((filenames_random_test, filenames_human_test, filenames_draw_test, filenames_pinned_test), axis = 0)

    print("filenames_train", filenames_train)
    print("filenames_val", filenames_val)

    print("Number of training, validation and test data:", len(filenames_train) * num_boards_per_file, len(filenames_val) * num_boards_per_file, len(filenames_test) * num_boards_per_file)

    # TO DO: add warning for number of validation files

    def data_generator(filenames, training = True):
        while True:
            # np.random.shuffle(filenames)
            for filename in filenames:
                table = pd.read_hdf(filename, 'table')
                if args.score_cut != None:
                    if len(args.score_cut) == 1:
                        table = table.drop(table[abs(table.score) >= args.score_cut[0]].index)
                    else:
                        table = table.reset_index(drop = True)
                        mask = (abs(table.score) >= args.score_cut[0]) & (abs(table.score) <= args.score_cut[1])
                        mask = ~mask
                        table = table[mask]
                        table = table.reset_index(drop = True)

                if training == True:
                    table = table.sample(frac=1).reset_index(drop=True)

                num_boards = len(table)
                for i in range(0, num_boards, args.batch_size):
                    batch_boards = table["board3d"][i:i+args.batch_size].values.tolist()
                    # print(np.shape(batch_boards))
                    batch_boards = np.reshape(batch_boards, (-1, 8, 8, 30))
                    # print(np.shape(batch_boards))

                    batch_parameter = np.array(table[["player move", "halfmove clock", "check", "checkmate", "stalemate", "insufficient material white", "insufficient material black", "seventyfive moves", "fivefold repetition", "castling right queen side white", "castling right king side white", "castling right queen side black", "castling right king side black"]][i:i+args.batch_size].values.tolist())

                    batch_score = np.array(table[["score"]][i:i+args.batch_size].values.tolist())
                    batch_score = batch_score + max_score
                    batch_score = batch_score / (2 * max_score)

                    # print(convert_board_int_to_fen(batch_boards[0], 12, batch_parameter[0][0], None, None, batch_parameter[0][1], 1))

                    # get_extreme_predictions(batch_score, batch_score, batch_boards, batch_parameter, "test")

                    # exit()

                    yield [batch_boards, batch_parameter], batch_score

    generator_train = data_generator(filenames_train)
    generator_val = data_generator(filenames_val)
    generator_test = data_generator(filenames_test)


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
model_input_board3d = Input(shape = (8, 8, 30))
model_board3d = Conv2D(64, kernel_size = (3, 3), padding = "same")(model_input_board3d)
model_board3d = dir_af[args.activation_function](model_board3d)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 256], increase_dim = True, dir_af = dir_af, activation_function = args.activation_function)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 256], dir_af = dir_af, activation_function = args.activation_function)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True, dir_af = dir_af, activation_function = args.activation_function)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], dir_af = dir_af, activation_function = args.activation_function)
# model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [512, 1028], increase_dim = True, dir_af = dir_af, activation_function = args.activation_function)
# model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [512, 1028], dir_af = dir_af, activation_function = args.activation_function)
model_board3d = Flatten()(model_board3d)

model_input_parameter = Input(shape = (13,))
# model_parameter = Dense(128, activation = "relu")(model_input_parameter)

model = Concatenate()([model_board3d, model_input_parameter])

model = Dense(256)(model)
model = dir_af[args.activation_function](model)
model = Dropout(args.dropout)(model)
model = Dense(128)(model)
model = dir_af[args.activation_function](model)
# model = Dropout(0.05)(model)
model = Dense(1, name = "score")(model)
# model = Dense(4, name = "score-check-checkmate-stalemate")(model)
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
# checkpointer = ModelCheckpoint(filepath = f"model/model_{args.name_experiment}.h5", verbose = args.verbose, save_best_only = True)
# os.makedirs("history/", exist_ok = True)
# model.fit([X_board3d_train, X_parameter_train], Y_train, epochs = args.epochs, batch_size = 32, validation_data=([X_board3d_val, X_parameter_val], Y_val), callbacks = [checkpointer, CSVLogger(f"history/history_{args.name_experiment}.csv"), ReduceLROnPlateau(monitor="val_loss", patience=20, min_delta=1e-7), EarlyStopping(monitor="val_loss", patience=40, min_delta=1e-7)], verbose = args.verbose)
# ############################################################################

model = models.Model(inputs = [model_input_board3d, model_input_parameter], outputs = model)
# model = tf.function(model, jit_compile = True)
model.summary()

os.makedirs("model/", exist_ok = True)

# compile model
# model.compile(optimizer=optimizers.Adam(5e-4), loss="mse")
# model.compile(optimizer = "adam", loss = "mse", learning_rate = 1e-4) # mean_squared_logarithmic_error
model.compile(optimizer = Adam(learning_rate = 1e-4), loss = args.loss_function) # mean_squared_logarithmic_error
checkpointer = ModelCheckpoint(filepath = f"model/model_{args.name_experiment}.h5", verbose = args.verbose, save_best_only = True, monitor = "val_loss")
os.makedirs("history/", exist_ok = True)

if args.generator == "n":
    model.fit([X_board3d_train, X_parameter_train], Y_train, epochs = args.epochs, batch_size = args.batch_size, validation_data = ([X_board3d_val, X_parameter_val], Y_val), callbacks = [checkpointer, CSVLogger(f"history/history_{args.name_experiment}.csv"), ReduceLROnPlateau(monitor = "val_loss", patience = 15, factor = 0.5, min_delta = 1e-7), EarlyStopping(monitor = "val_loss", patience = 30, min_delta = 1e-7)], verbose = args.verbose)
elif args.generator == "y":
    # model.fit(dask_data_generator(X_board3d_train, X_parameter_train, Y_train, train_size, batch_size), steps_per_epoch = steps_per_epoch[0], epochs = args.epochs, validation_data = dask_data_generator(X_board3d_val, X_parameter_val, Y_val, val_size, batch_size), validation_steps = steps_per_epoch[1], callbacks = [checkpointer, CSVLogger(f"history/history_{args.name_experiment}.csv"), ReduceLROnPlateau(monitor = "val_loss", patience = 20, min_delta = 1e-7), EarlyStopping(monitor = "val_loss", patience = 40, min_delta = 1e-7)], verbose = args.verbose)
    # model.fit(dask_data_generator(X_board3d_train, X_parameter_train, Y_train, train_size, batch_size), steps_per_epoch = steps_per_epoch[0], epochs = args.epochs, validation_data = dask_data_generator(X_board3d_val, X_parameter_val, Y_val, val_size, batch_size), validation_steps = steps_per_epoch[1], verbose = args.verbose)
    steps_per_epoch_train = np.floor(len(filenames_train) * num_boards_per_file / args.batch_size)
    steps_per_epoch_val = np.floor(len(filenames_val) * num_boards_per_file / args.batch_size)
    print("steps_per_epoch_train", steps_per_epoch_train)
    print("steps_per_epoch_val", steps_per_epoch_val)
    model.fit(generator_train, steps_per_epoch = steps_per_epoch_train, epochs = args.epochs, validation_data = generator_val, validation_steps = steps_per_epoch_val, callbacks = [checkpointer, CSVLogger(f"history/history_{args.name_experiment}.csv"), ReduceLROnPlateau(monitor = "val_loss", patience = 15, factor = 0.5, min_delta = 1e-7), EarlyStopping(monitor = "val_loss", patience = 30, min_delta = 1e-7)], verbose = args.verbose)

# # predict data
model = models.load_model(f"model/model_{args.name_experiment}.h5")

# save model predictions on training an validation data
os.makedirs(f"prediction/{args.name_experiment}", exist_ok = True)

if args.generator == "n":
    prediction_val = model.predict([X_board3d_val, X_parameter_val])

else:
    val_steps = int(np.floor(len(filenames_val) * num_boards_per_file / args.batch_size))
    generator_val = data_generator(filenames_val, training = False)
    prediction_val = model.predict(generator_val, steps = val_steps, verbose = args.verbose)

    # Initialize empty lists to store the values
    X_board3d_val = []
    X_parameter_val = []
    Y_val = []
    filenames_val_ordered = []  # Store filenames in the order of data generation
    
    # Iterate over the generator and capture the values
    generator_val = data_generator(filenames_val, training = False)
    for _ in range(val_steps):
        X_batch, Y_batch = next(generator_val)
        X_board3d_val.append(X_batch[0])
        X_parameter_val.append(X_batch[1])
        Y_val.append(Y_batch)

        print(type(X_batch))
        # print(np.shape(X_batch))
        print(np.shape(X_batch[0]))
        print(np.shape(X_batch[1]))
        print(np.shape(X_batch[0][0]))
        print(np.shape(X_batch[1][0]))
        print("##############################")

        X_batch[0] = np.moveaxis(X_batch[0], -1, 1)

        print(np.shape(X_batch[0]))
        print(np.shape(X_batch[1]))
        print(np.shape(X_batch[0][0]))
        print(np.shape(X_batch[1][0]))

        print(convert_board_int_to_fen(X_batch[0][0], 12, X_batch[1][0][0], None, None, X_batch[1][0][1], 1))

        # TO DO: check if generator reshapes data correctly

        exit()

    # Convert the lists to numpy arrays

    # print("shape X_board3d_val[0]", np.shape(X_board3d_val[0][0]))
    # print("shape X_parameter_val[0][0]", np.shape(X_parameter_val[0][0][0]))
    # print(convert_board_int_to_fen(X_board3d_val[0][0], 12, X_parameter_val[0][0][0], None, None, X_parameter_val[0][0][1], 1))

    # print("shape X_board3d_val", np.shape(X_board3d_val))
    X_board3d_val = np.concatenate(X_board3d_val)
    # print("shape X_board3d_val", np.shape(X_board3d_val))
    # print("shape X_parameter_val", np.shape(X_parameter_val))
    X_parameter_val = np.concatenate(X_parameter_val)
    # print("shape X_parameter_val", np.shape(X_parameter_val))
    Y_val = np.concatenate(Y_val)

    # print(convert_board_int_to_fen(X_board3d_val[0], 12, X_parameter_val[0][0], None, None, X_parameter_val[0][1], 1))

    # exit()


df1 = pd.DataFrame({"predicted score": prediction_val[:,0]})
df2 = pd.DataFrame({"true score": Y_val[:,0]})
df9 = pd.DataFrame({"player move": X_parameter_val[:,0]})
df10 = pd.DataFrame({"seventyfive moves": X_parameter_val[:,4]})
df11 = pd.DataFrame({"fivefold repetition": X_parameter_val[:,5]})

table_pred_val = pd.concat([df1, df2, df9, df10, df11], axis = 1)

print(table_pred_val)

table_pred_val.to_hdf(f"prediction/{args.name_experiment}/prediction_val_{args.name_experiment}.h5", key = "table")

print("shape X_board3d_val", np.shape(X_board3d_val))

get_extreme_predictions(table_pred_val["predicted score"], table_pred_val["true score"], X_board3d_val, X_parameter_val, args.name_experiment)
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