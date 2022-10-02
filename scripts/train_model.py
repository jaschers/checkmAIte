import numpy as np
from utilities_keras import *
from keras import models, optimizers
from keras.layers import Conv2D, GlobalMaxPooling2D, Dense, Flatten, Input, BatchNormalization, Add, Activation, Concatenate
import keras.utils as utils
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
import pandas as pd
import os
import time
import argparse
import sys
np.set_printoptions(threshold=sys.maxsize)

######################################## argparse setup ########################################
script_descr="""
Trains the residual neural network
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-na", "--name", type = str, required = True, metavar = "-", help = "Name of this particular experiment")
parser.add_argument("-e", "--epochs", type = int, required = True, metavar = "-", help = "Number of epochs for the training")

args = parser.parse_args()
##########################################################################################


# load data
num_runs = 50 #19
table = pd.DataFrame()
for run in range(num_runs):
    print(f"Loading data run {run}...")
    start = time.time()
    table_run = pd.read_hdf(f"data/3d/24_8_8_depth15/data{run}.h5", key = "table")
    middle = time.time()
    frame = [table, table_run]
    table = pd.concat(frame)
    end = time.time()
    print(f"Data run {run} loaded within {np.round(middle-start, 1)} sec...")

table = table.reset_index(drop = True)
table = table.drop(table[abs(table.score) > 10900].index)
table = table.reset_index(drop = True)
print(table)

X_board3d = table["board3d"].values.tolist()
X_parameter = table[["player move", "halfmove clock"]].values.tolist()
Y = table["score"].values.tolist()

# prepare data for neural network
# X_board3d = np.zeros((len(X_board3d_aux), 14, 8, 8), dtype = int)
# X_board3d[:,:12] = np.array(X_board3d_aux)[:,:12]
# X_board3d[:,12] = np.sum(np.array(X_board3d_aux)[0,12:18], axis = 0)
# X_board3d[:,13] = np.sum(np.array(X_board3d_aux)[0,18:], axis = 0)

X_board3d = np.moveaxis(X_board3d, 1, -1)
X_board3d_shape = np.shape(X_board3d)
X_parameter_shape = np.shape(X_parameter)

print("X board3d shape:", np.shape(X_board3d))
print("X parameter shape:", np.shape(X_parameter))
print("Y score shape:", np.shape(Y))

# norm Y data between -1 and 1
print("Ymin, Ymax before normalisation:", np.min(Y), np.max(Y))
Y = Y - np.min(Y)
Y = Y / np.max(Y)
# Y = np.asarray(Y / abs(np.array(Y)).max() / 2 + 0.5, dtype=np.float32) # normalization (0 - 1)
print("Ymin, Ymax after normalisation:", np.min(Y), np.max(Y))

X_board3d_train, X_board3d_val, X_board3d_test = np.split(X_board3d, [-int(len(X_board3d) / 5), -int(len(X_board3d) / 10)]) 
X_parameter_train, X_parameter_val, X_parameter_test = np.split(X_parameter, [-int(len(X_parameter) / 5), -int(len(X_parameter) / 10)]) 
Y_train, Y_val, Y_test = np.split(Y, [-int(len(Y) / 5), -int(len(Y) / 10)]) 

# print(np.shape([X_board3d, X_parameter]), np.shape(X_val), np.shape(X_test))
# print(np.shape(Y_train), np.shape(Y_val), np.shape(Y_test))
print("Number of training, validation and test data:", len(X_board3d_train), len(X_board3d_val), len(X_board3d_test))

# # # define model
# ############################################################################
# # model_input = Input(shape = X_shape[1:])
# # model = Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(model_input)
# # model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
# # model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
# # model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
# # model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
# # # model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
# # # model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
# # # model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
# # # model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])
# # # model = ResBlock(model,)GlobalMaxPooling2D()(model)
# # model = Flatten()(model)
# # model = Dense(128, activation = "relu")(model)
# # model = Dense(1, name = "score")(model)
# ############################################################################


# ############################################################################
# model_input = Input(shape = X_shape[1:])
# model = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model_input)
# model = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model)
# model = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model)
# model = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model)
# model = Flatten()(model)
# model = Dense(64, activation = "relu")(model)
# model = Dense(1, name = "score")(model)
# ############################################################################


# ############################################################################
# model_input_board3d = Input(shape = X_board3d_shape[1:])
# model_board3d = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model_input_board3d)
# model_board3d = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model_board3d)
# model_board3d = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model_board3d)
# model_board3d = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model_board3d)
# model_board3d = Flatten()(model_board3d)

# model_input_parameter = Input(shape = X_parameter_shape[1:])
# # model_parameter = Dense(128, activation = "relu")(model_input_parameter)

# model = Concatenate()([model_board3d, model_input_parameter])

# model = Dense(128, activation = "relu")(model)
# model = Dense(64, activation = "relu")(model)
# model = Dense(1, name = "score")(model)
# ############################################################################

############################################################################
model_input_board3d = Input(shape = X_board3d_shape[1:])
model_board3d = Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(model_input_board3d)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
model_board3d = ResBlock(model_board3d, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
model_board3d = Flatten()(model_board3d)

model_input_parameter = Input(shape = X_parameter_shape[1:])
# model_parameter = Dense(128, activation = "relu")(model_input_parameter)

model = Concatenate()([model_board3d, model_input_parameter])

model = Dense(128, activation = "relu")(model)
model = Dense(64, activation = "relu")(model)
model = Dense(1, name = "score")(model)
############################################################################

model = models.Model(inputs = [model_input_board3d, model_input_parameter], outputs = model)
model.summary()

os.makedirs("model/", exist_ok = True)

# compile model
# model.compile(optimizer=optimizers.Adam(5e-4), loss="mse")
model.compile(optimizer = "adam", loss="mse")

os.makedirs("history/", exist_ok = True)
model.fit([X_board3d_train, X_parameter_train], Y_train, epochs = args.epochs, batch_size = 32, validation_data=([X_board3d_val, X_parameter_val], Y_val), callbacks = [CSVLogger(f"history/history_{args.name}.csv"), ReduceLROnPlateau(monitor="val_loss", patience=10), EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-4)])

model.save(f"model/model_{args.name}.h5")

# save model predictions on training an validation data
os.makedirs("prediction/", exist_ok = True)

prediction_train = model.predict([X_board3d_train, X_parameter_train])
prediction_train = np.reshape(prediction_train, (np.shape(prediction_train)[0]))

prediction_val = model.predict([X_board3d_val, X_parameter_val])
prediction_val = np.reshape(prediction_val, (np.shape(prediction_val)[0]))

table_pred_train = pd.DataFrame({"prediction": prediction_train})
table_true_train = pd.DataFrame({"true score": Y_train})

table_pred_val = pd.DataFrame({"prediction": prediction_val})
table_true_val = pd.DataFrame({"true score": Y_val})

table_pred_train = pd.concat([table_pred_train, table_true_train], axis = 1)
table_pred_val = pd.concat([table_pred_val, table_true_val], axis = 1)

# print(table_pred_train)
# print(table_pred_val)

table_pred_train.to_hdf(f"prediction/prediction_train_{args.name}.h5", key = "table")
table_pred_val.to_hdf(f"prediction/prediction_val_{args.name}.h5", key = "table")