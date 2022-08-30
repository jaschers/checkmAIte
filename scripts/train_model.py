import numpy as np
from utilities import *
from keras import models
from keras.layers import Conv2D, GlobalMaxPooling2D, Dense, Flatten, Input
import keras.utils as utils
from keras.callbacks import CSVLogger
import pandas as pd
import os
import time
from keras.applications import ResNet50

# load data
num_runs = 35
table = pd.DataFrame()
for run in range(num_runs):
    print(f"Loading data run {run}...")
    start = time.time()
    table_run = pd.read_hdf(f"data/data{run}.h5", key = "table")
    middle = time.time()
    frame = [table, table_run]
    table = pd.concat(frame)
    end = time.time()
    print(f"Data run {run} loaded within {np.round(middle-start)} sec...")

table = table.reset_index(drop = True)
print(table)

X = table["boards (int)"].values.tolist()
Y = table["score"].values.tolist()

# prepare data for neural network
X_shape = np.shape(X)
X = np.reshape(X, (X_shape[0], X_shape[1], X_shape[2], 1))
X_shape = np.shape(X)

# norm Y data between -1 and 1
Y_max = np.min(Y)
Y = Y / Y_max

X_train, X_val, X_test = np.split(X, [-int(len(X) / 5), -int(len(X) / 10)]) 
Y_train, Y_val, Y_test = np.split(Y, [-int(len(X) / 5), -int(len(Y) / 10)]) 

# print(np.shape(X_train), np.shape(X_val), np.shape(X_test))
# print(np.shape(Y_train), np.shape(Y_val), np.shape(Y_test))
print("Number of training, validation and test data:", len(X_train), len(X_val), len(X_test))

# define model
model_input = Input(shape = X_shape[1:])
model = Conv2D(16, kernel_size = (3, 3), activation = "relu", padding = "same")(model_input)
model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64], increase_dim = True)
model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [32, 64])
model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128], increase_dim = True)
model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [64, 128])
# model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256], increase_dim = True)
# model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [128, 256])
# model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512], increase_dim = True)
# model = ResBlock(model, kernelsizes = [(1, 1), (3, 3)], filters = [256, 512])

model = GlobalMaxPooling2D()(model)
model = Flatten()(model)
model = Dense(64, activation = "relu")(model)
model = Dense(1, name = "score")(model)

# # define model
# model_input = Input(shape = X_shape[1:])
# model = Conv2D(16, (3, 3), activation = "relu", padding = "same")(model_input)
# model = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model)
# model = Conv2D(64, (3, 3), activation = "relu", padding = "same")(model)
# model = Conv2D(128, (3, 3), activation = "relu", padding = "same")(model)
# model = GlobalMaxPooling2D()(model)
# model = Flatten()(model)
# model = Dense(128, activation = "relu")(model)
# model = Dense(64, activation = "relu")(model)
# # model = Dense(16, activation = "relu")(model)
# model = Dense(1, name = "score")(model)

# model_input = Input(shape = X_shape[1:])
# model = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model_input)
# model = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model)
# model = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model)
# model = Conv2D(32, (3, 3), activation = "relu", padding = "same")(model)
# model = Flatten()(model)
# model = Dense(64, activation = "relu")(model)
# model = Dense(1, name = "score")(model)

# model_input = Input(shape = X_shape[1:])
# model = Conv2D(3, (3, 3), activation = "relu", padding = "same")(model_input)
# model = ResNet50()(model)

model = models.Model(inputs = model_input, outputs = model)
model.summary()

os.makedirs("model/", exist_ok = True)
# utils.plot_model(model, to_file = "model/model.pdf", show_shapes = True, show_layer_names = False)

# compile model
model.compile(optimizer = "adam",
              loss="mse")

os.makedirs("history/", exist_ok = True)
model.fit(X_train, Y_train, epochs = 15, batch_size = 32, validation_data=(X_val, Y_val), callbacks = [CSVLogger("history/history.csv")])

model.save("model/model.h5")