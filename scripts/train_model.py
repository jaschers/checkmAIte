import numpy as np
from utilities_keras import *
from tensorflow.keras.optimizers import Adam
from keras import models, optimizers
from keras.layers import Conv2D, GlobalMaxPooling2D, Dense, Flatten, Input, BatchNormalization, Add, Activation, Concatenate, Dropout
# from keras.applications import EfficientNetV2L, ResNet152V2, VGG19, EfficientNetV2M, EfficientNetV2S, ResNet50, MobileNet, InceptionV3, EfficientNetV2B0
import keras.utils as utils
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import pandas as pd
import os
import time
import argparse
import sys
# np.set_printoptions(threshold=sys.maxsize)

######################################## argparse setup ########################################
script_descr="""
Trains the residual neural network
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-r", "--runs", type = int, required = True, metavar = "-", help = "Number of runs used for training")
parser.add_argument("-nd", "--name_data", type = str, required = False, metavar = "-", default = "34_8_8_depth0_mm100_ms15000", help = "name of the data folder")
parser.add_argument("-sc", "--score_cut", type = float, required = False, nargs='+', metavar = "-", default = None, help = "score cut applied on the data (default: None)")
parser.add_argument("-ne", "--name_experiment", type = str, required = True, metavar = "-", help = "Name of this particular experiment")
parser.add_argument("-e", "--epochs", type = int, required = False, metavar = "-", default = 1000, help = "Number of epochs for the training")

args = parser.parse_args()
##########################################################################################

# load data
X_board3d, X_parameter, Y = load_data(args.runs, args.name_data, args.score_cut)

# print(np.shape(X_board3d))
# X_board3d_1 = X_board3d[:,:24]
# X_board3d_2 = X_board3d[:,30:]
# print(np.shape(X_board3d_1))
# print(np.shape(X_board3d_2))
# X_board3d = np.concatenate((X_board3d_1, X_board3d_2), axis = 1)
# print(np.shape(X_board3d))

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

# print(np.shape([X_board3d, X_parameter]), np.shape(X_val), np.shape(X_test))
# print(np.shape(Y_train), np.shape(Y_val), np.shape(Y_test))
print("Number of training, validation and test data:", len(X_board3d_train), len(X_board3d_val), len(X_board3d_test))

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
model.fit([X_board3d_train, X_parameter_train], Y_train, epochs = args.epochs, batch_size = 32, validation_data = ([X_board3d_val, X_parameter_val], Y_val), callbacks = [checkpointer, CSVLogger(f"history/history_{args.name_experiment}.csv"), ReduceLROnPlateau(monitor = "val_loss", patience = 20, min_delta = 1e-7), EarlyStopping(monitor = "val_loss", patience = 40, min_delta = 1e-7)], verbose = 2)

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