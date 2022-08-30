import numpy as np
from keras import models
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm
import os

# load data
num_runs = 35
table = pd.DataFrame()
for run in range(num_runs):
    print(f"Loading data run {run}...")
    table_run = pd.read_hdf(f"data/data{run}.h5", key = "table")
    frame = [table, table_run]
    table = pd.concat(frame)
table = table.dropna().reset_index()

X = table["boards (int)"].values.tolist()
Y = table["score"].values.tolist()

# prepare data for neural network
X_shape = np.shape(X)
X = np.reshape(X, (X_shape[0], X_shape[1], X_shape[2], 1))

Y_max = np.min(Y)
Y = Y / Y_max

X_train, X_val, X_test = np.split(X, [-int(len(X) / 5), -int(len(X) / 10)]) 
Y_train, Y_val, Y_test = np.split(Y, [-int(len(X) / 5), -int(len(Y) / 10)]) 

model = models.load_model("model/model.h5")

history = pd.read_csv("history/history.csv")

os.makedirs("evaluation/", exist_ok = True)

plt.figure()
plt.plot(history["loss"], label="Training")
plt.plot(history["val_loss"], label = "Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.show()
plt.tight_layout()
plt.savefig("evaluation/history.pdf")
plt.close()

prediction_train = model.predict(X_train)
prediction_train = np.reshape(prediction_train, (np.shape(prediction_train)[0]))

prediction_val = model.predict(X_val)
prediction_val = np.reshape(prediction_val, (np.shape(prediction_val)[0]))

print(prediction_train)
print(prediction_val)

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

plt.figure()
plt.hist2d(Y_val, prediction_val, bins = (50, 50), cmap = newcmp, norm = LogNorm())
cbar = plt.colorbar()
cbar.set_label('Number of boards')
plt.plot(np.linspace(np.min(Y_val), np.max(Y_val), 100), np.linspace(np.min(Y_val), np.max(Y_val), 100), color = "black")
plt.ylim(np.min(Y_val), np.max(Y_val))
plt.xlabel("True score")
plt.ylabel("Predicted score")
plt.tight_layout()
plt.savefig("evaluation/2Dscattering_val.pdf")
# plt.show()
plt.close()

plt.figure()
plt.hist2d(Y_train, prediction_train, bins = (50, 50), cmap = newcmp, norm = LogNorm())
cbar = plt.colorbar()
cbar.set_label('Number of boards')
plt.plot(np.linspace(np.min(Y_train), np.max(Y_train), 100), np.linspace(np.min(Y_train), np.max(Y_train), 100), color = "black")
plt.ylim(np.min(Y_train), np.max(Y_train))
plt.xlabel("True score")
plt.ylabel("Predicted score")
plt.tight_layout()
plt.savefig("evaluation/2Dscattering_train.pdf")
# plt.show()
plt.close()

# test_acc = model.evaluate(X_test, Y_test, verbose = 2)
# print("Test accuracy: ", test_acc)
