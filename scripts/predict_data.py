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
parser.add_argument("-nd", "--name_data", type = str, required = False, metavar = "-", default = "34_8_8_depth0_mm100_ms15000", help = "name of the data folder")
parser.add_argument("-sc", "--score_cut", type = float, required = False, nargs='+', metavar = "-", default = None, help = "score cut applied on the data (default: None)")
parser.add_argument("-rh", "--read_human", type = str, required = False, metavar = "-", default = "y", help = "Should human data be read? [y/n], default: y")
parser.add_argument("-rd", "--read_draw", type = str, required = False, metavar = "-", default = "y", help = "Should draw data be read? [y/n], default: y")
parser.add_argument("-ne", "--name_experiment", type = str, required = True, metavar = "-", help = "Name of this particular experiment")

args = parser.parse_args()
##########################################################################################

# load data
X_board3d, X_parameter, Y = load_data(args.runs, args.name_data, args.score_cut, args.read_human, args.read_draw)

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

X_board3d_train, X_board3d_val, X_board3d_test = np.split(X_board3d, [-int(len(X_board3d) / 5), -int(len(X_board3d) / 10)]) 
X_parameter_train, X_parameter_val, X_parameter_test = np.split(X_parameter, [-int(len(X_parameter) / 5), -int(len(X_parameter) / 10)]) 
Y_train, Y_val, Y_test = np.split(Y, [-int(len(Y) / 5), -int(len(Y) / 10)]) 

print("Number of training, validation and test data:", len(X_board3d_train), len(X_board3d_val), len(X_board3d_test))

model = models.load_model(f"model/model_{args.name_experiment}.h5")

# save model predictions on training an validation data
os.makedirs(f"prediction/{args.name_experiment}", exist_ok = True)

prediction_val = model.predict([X_board3d_val, X_parameter_val])

df1 = pd.DataFrame({"predicted score": prediction_val[:,0]})
df2 = pd.DataFrame({"true score": Y_val[:,0]})
df3 = pd.DataFrame({"predicted check": prediction_val[:,1]})
df4 = pd.DataFrame({"true check": Y_val[:,1]})
df5 = pd.DataFrame({"predicted checkmate": prediction_val[:,2]})
df6 = pd.DataFrame({"true checkmate": Y_val[:,2]})
df7 = pd.DataFrame({"predicted stalemate": prediction_val[:,3]})
df8 = pd.DataFrame({"true stalemate": Y_val[:,3]})
df9 = pd.DataFrame({"player move": X_parameter_val[:,0]})
df10 = pd.DataFrame({"seventyfive moves": X_parameter_val[:,4]})
df11 = pd.DataFrame({"fivefold repetition": X_parameter_val[:,5]})

table_pred_val = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11], axis = 1)

print(table_pred_val)

table_pred_val.to_hdf(f"prediction/{args.name_experiment}/prediction_val_{args.name_experiment}.h5", key = "table")

get_extreme_predictions(table_pred_val["predicted score"], table_pred_val["true score"], X_board3d_val, X_parameter_val, args.name_experiment)