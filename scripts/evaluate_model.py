import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm
import os
import argparse

######################################## argparse setup ########################################
script_descr="""
Evaluates the residual neural network
"""

# Open argument parser
parser = argparse.ArgumentParser(description=script_descr)

# Define expected arguments
parser.add_argument("-na", "--name", type = str, required = False, metavar = "-", help = "Name of this particular experiment")

args = parser.parse_args()
##########################################################################################


# load data
# print("Loading training data...")
# table_train = pd.read_hdf(f"prediction/prediction_train_{args.name}.h5", key = "table")
print("Loading validation data...")
table_val = pd.read_hdf(f"prediction/prediction_val_{args.name}.h5", key = "table")

# prediction_train = table_train["prediction"]
# true_score_train = table_train["true score"]

prediction_val = table_val["prediction"]
true_score_val = table_val["true score"]

history = pd.read_csv(f"history/history_{args.name}.csv")

os.makedirs(f"evaluation/{args.name}/", exist_ok = True)

print("Plotting history...")
plt.figure()
plt.plot(history["loss"], label="Training")
plt.plot(history["val_loss"], label = "Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.show()
plt.tight_layout()
plt.savefig(f"evaluation/{args.name}/history_{args.name}.pdf")
plt.close()

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

print("Plotting 2D scattering...")
print(np.min(true_score_val), np.max(true_score_val))
print(np.min(prediction_val), np.max(prediction_val))
plt.figure()
plt.hist2d(true_score_val, prediction_val, bins = (50, 50), cmap = newcmp, norm = LogNorm())
cbar = plt.colorbar()
cbar.set_label('Number of boards')
plt.plot(np.linspace(np.min(true_score_val), np.max(true_score_val), 100), np.linspace(np.min(true_score_val), np.max(true_score_val), 100), color = "black")
# plt.ylim(np.min(true_score_val), np.max(true_score_val))
plt.xlabel("True score")
plt.ylabel("Predicted score")
plt.tight_layout()
plt.savefig(f"evaluation/{args.name}/2Dscattering_val_{args.name}.pdf")
# plt.show()
plt.close()

# plt.figure()
# plt.hist2d(true_score_train, prediction_train, bins = (50, 50), cmap = newcmp, norm = LogNorm())
# cbar = plt.colorbar()
# cbar.set_label('Number of boards')
# plt.plot(np.linspace(np.min(true_score_train), np.max(true_score_train), 100), np.linspace(np.min(true_score_train), np.max(true_score_train), 100), color = "black")
# plt.ylim(np.min(true_score_train), np.max(true_score_train))
# plt.xlabel("True score")
# plt.ylabel("Predicted score")
# plt.tight_layout()
# plt.savefig(f"evaluation/{args.name}/2Dscattering_train_{args.name}.pdf")
# # plt.show()
# plt.close()

# test_acc = model.evaluate(X_test, Y_test, verbose = 2)
# print("Test accuracy: ", test_acc)
