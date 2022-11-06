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

######################################## argparse setup ########################################
script_descr="""
Neural network prediction of the score of the validation data 
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-r", "--runs", type = int, required = True, metavar = "-", help = "Number of runs used for training")
parser.add_argument("-nd", "--name_data", type = str, required = True, metavar = "-", help = "name of the data folder")
parser.add_argument("-sc", "--score_cut", type = float, required = True, metavar = "-", default = None, help = "score cut applied on the data (default: None)")
parser.add_argument("-ne", "--name_experiment", type = str, required = True, metavar = "-", help = "Name of this particular experiment")

args = parser.parse_args()
##########################################################################################

# load data
X_board3d, X_parameter, Y = load_data(args.runs, args.name_data, args.score_cut)

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

print("Number of training, validation and test data:", len(X_board3d_train), len(X_board3d_val), len(X_board3d_test))

model = models.load_model(f"model/model_{args.name_experiment}.h5")

# save model predictions on training an validation data
os.makedirs("prediction/", exist_ok = True)

prediction_val = model.predict([X_board3d_val, X_parameter_val])
prediction_val = np.reshape(prediction_val, (np.shape(prediction_val)[0]))

table_pred_val = pd.DataFrame({"prediction": prediction_val})
table_true_val = pd.DataFrame({"true score": Y_val})

table_pred_val = pd.concat([table_pred_val, table_true_val], axis = 1)

print(table_pred_val)

table_pred_val.to_hdf(f"prediction/prediction_val_{args.name_experiment}.h5", key = "table")

get_extreme_predictions(prediction_val, Y_val, X_board3d, X_parameter, args.name_experiment)