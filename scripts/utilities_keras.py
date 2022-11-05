from keras.layers import Conv2D, ReLU, Add, BatchNormalization
import pandas as pd
import time
import numpy as np

def ResBlock(z, kernelsizes, filters, increase_dim = False):
    # https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
    # https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
    # https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

    z_shortcut = z
    # z_shortcut = BatchNormalization()(z_shortcut)
    kernelsize_1, kernelsize_2 = kernelsizes
    filters_1, filters_2 = filters

    fz = Conv2D(filters_1, kernelsize_1)(z)
    # fz = BatchNormalization()(fz)
    fz = ReLU()(fz)

    fz = Conv2D(filters_1, kernelsize_2, padding = "same")(fz)
    # fz = BatchNormalization()(fz)
    fz = ReLU()(fz)
    
    fz = Conv2D(filters_2, kernelsize_1)(fz)
    # fz = BatchNormalization()(fz)

    if increase_dim == True:
        z_shortcut = Conv2D(filters_2, (1, 1))(z_shortcut)
        # z_shortcut = BatchNormalization()(z_shortcut)

    out = Add()([fz, z_shortcut])
    # out = BatchNormalization()(out)
    out = ReLU()(out)
    # out = MaxPooling2D(pool_size=(3, 3), strides = 1)(out)
    
    return out

def load_data(num_runs, name_data, score_cut):
    # load data
    table = pd.DataFrame()
    for run in range(num_runs):
        print(f"Loading data run {run}...")
        start = time.time()
        table_run = pd.read_hdf(f"data/3d/{name_data}/data{run}.h5", key = "table")
        middle = time.time()
        frame = [table, table_run]
        table = pd.concat(frame)
        end = time.time()
        print(f"Data run {run} loaded in {np.round(middle-start, 1)} sec...")

    if score_cut != None:
        table = table.reset_index(drop = True)
        table = table.drop(table[abs(table.score) > score_cut].index)
        table = table.reset_index(drop = True)
    print(table)

    X_board3d = table["board3d"].values.tolist()
    X_parameter = table[["player move", "halfmove clock"]].values.tolist()
    Y = table["score"].values.tolist()

    return(X_board3d, X_parameter, Y)