import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm
import os

# load data
print("Loading training data...")
table_train = pd.read_hdf("prediction/prediction_train.h5", key = "table")
print("Loading validation data...")
table_val = pd.read_hdf("prediction/prediction_val.h5", key = "table")

prediction_train = table_train["prediction"] * 15000
true_score_train = table_train["true score"] * 15000

prediction_val = table_val["prediction"] * 15000
true_score_val = table_val["true score"] * 15000

history = pd.read_csv("history/history.csv")

os.makedirs("evaluation/", exist_ok = True)

print("Plotting history...")
plt.figure()
plt.plot(history["loss"], label="Training")
plt.plot(history["val_loss"], label = "Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.show()
plt.tight_layout()
plt.savefig("evaluation/history.pdf")
plt.close()

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

print("Plotting 2D scattering...")
plt.figure()
plt.hist2d(true_score_val, prediction_val, bins = (50, 50), cmap = newcmp, norm = LogNorm())
cbar = plt.colorbar()
cbar.set_label('Number of boards')
plt.plot(np.linspace(np.min(true_score_val), np.max(true_score_val), 100), np.linspace(np.min(true_score_val), np.max(true_score_val), 100), color = "black")
plt.ylim(np.min(true_score_val), np.max(true_score_val))
plt.xlabel("True score")
plt.ylabel("Predicted score")
plt.tight_layout()
plt.savefig("evaluation/2Dscattering_val.pdf")
# plt.show()
plt.close()

plt.figure()
plt.hist2d(true_score_train, prediction_train, bins = (50, 50), cmap = newcmp, norm = LogNorm())
cbar = plt.colorbar()
cbar.set_label('Number of boards')
plt.plot(np.linspace(np.min(true_score_train), np.max(true_score_train), 100), np.linspace(np.min(true_score_train), np.max(true_score_train), 100), color = "black")
plt.ylim(np.min(true_score_train), np.max(true_score_train))
plt.xlabel("True score")
plt.ylabel("Predicted score")
plt.tight_layout()
plt.savefig("evaluation/2Dscattering_train.pdf")
# plt.show()
plt.close()

# test_acc = model.evaluate(X_test, Y_test, verbose = 2)
# print("Test accuracy: ", test_acc)
