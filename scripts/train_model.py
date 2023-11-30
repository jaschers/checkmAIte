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
Trains the convolutional neural network
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
parser.add_argument("-v", "--verbose", type = int, required = False, metavar = "-", default = 2, help = "verbose level for training [0, 1, 2], default: 2")

args = parser.parse_args()
##########################################################################################
max_score = 15000
num_boards_per_file = 10000
dir_af = {"relu": ReLU(), "leakyrelu": LeakyReLU(), "elu": ELU(), "softmax": Softmax()}

# load data
print("Memory usage before loading data:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MiB")

X_board3d, X_parameter, Y = load_data(args.runs, args.name_data, args.score_cut, args.read_human, args.read_draw, args.read_pinned)

X_board3d = np.moveaxis(X_board3d, 1, -1)
X_board3d_shape = np.shape(X_board3d)
X_parameter_shape = np.shape(X_parameter)

# norm Y data between -1 and 1
print("Y_score_min, Y_score_max before normalisation:", np.min(Y[:,0]), np.max(Y[:,0]))
print("Y_score unique", np.unique(Y[:,0]))
Y = Y.astype("float")
Y[:,0] = Y[:,0] + max_score
Y[:,0] = Y[:,0] / (2 * max_score)
print("Y_score_min, Y_score_max after normalisation:", np.min(Y[:,0]), np.max(Y[:,0]))

X_board3d_train, X_board3d_val, X_board3d_test = np.split(X_board3d, [-int(len(X_board3d) / 5), -int(len(X_board3d) / 10)]) 
X_parameter_train, X_parameter_val, X_parameter_test = np.split(X_parameter, [-int(len(X_parameter) / 5), -int(len(X_parameter) / 10)]) 
Y_train, Y_val, Y_test = np.split(Y, [-int(len(Y) / 5), -int(len(Y) / 10)]) 

print("Number of training, validation and test data:", len(X_board3d_train), len(X_board3d_val), len(X_board3d_test))

print("Memory usage after loading data:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MiB")

######################################################
model_input_board3d = Input(shape = (8, 8, 30))
model_board3d = Conv2D(64, kernel_size = (3, 3), padding = "same")(model_input_board3d)
model_board3d = dir_af[args.activation_function](model_board3d)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 256], increase_dim = True, dir_af = dir_af, activation_function = args.activation_function)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 256], dir_af = dir_af, activation_function = args.activation_function)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True, dir_af = dir_af, activation_function = args.activation_function)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], dir_af = dir_af, activation_function = args.activation_function)
model_board3d = Flatten()(model_board3d)

model_input_parameter = Input(shape = (13,))

model = Concatenate()([model_board3d, model_input_parameter])

model = Dense(256)(model)
model = dir_af[args.activation_function](model)
model = Dropout(args.dropout)(model)
model = Dense(128)(model)
model = dir_af[args.activation_function](model)
model = Dense(1, name = "score")(model)

model = models.Model(inputs = [model_input_board3d, model_input_parameter], outputs = model)
model.summary()

os.makedirs("model/", exist_ok = True)

# compile model
model.compile(optimizer = Adam(learning_rate = 1e-4), loss = args.loss_function) # mean_squared_logarithmic_error
checkpointer = ModelCheckpoint(filepath = f"model/model_{args.name_experiment}.h5", verbose = args.verbose, save_best_only = True, monitor = "val_loss")
os.makedirs("history/", exist_ok = True)

model.fit([X_board3d_train, X_parameter_train], Y_train, epochs = args.epochs, batch_size = args.batch_size, validation_data = ([X_board3d_val, X_parameter_val], Y_val), callbacks = [checkpointer, CSVLogger(f"history/history_{args.name_experiment}.csv"), ReduceLROnPlateau(monitor = "val_loss", patience = 15, factor = 0.5, min_delta = 1e-7), EarlyStopping(monitor = "val_loss", patience = 30, min_delta = 1e-7)], verbose = args.verbose)

# # predict data
model = models.load_model(f"model/model_{args.name_experiment}.h5")

# save model predictions on training an validation data
os.makedirs(f"prediction/{args.name_experiment}", exist_ok = True)

if args.generator == "n":
    prediction_val = model.predict([X_board3d_val, X_parameter_val])

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