# CheckmAIte

<img src="https://github.com/jaschers/checkmAIte/blob/main/visuals/logo.png" width="250">

CheckmAIte is a chess engine based on a convolutional neural network and a minimax algorithm optimizied using alpha-beta pruning, a transposition table, move ordering, multiprocessing, just-in-time compilation and an engame database.

## Demo
<img src="https://github.com/jaschers/checkmAIte/blob/main/visuals/board.gif" width="350">

Here's a demo of the CheckmAIte (white) winning against a 2500 elo chess bot on chess.com, called Danny-bot (black). The game was played on my chess.com account and can be found [here](https://www.chess.com/analysis/library/3or7E8zuS6). In this game, a depth of 5 was used for the minimax algorithm of CheckmAIte. CheckmAIte is fully based on Python.

## Requirements
* Python 3.7+
* Gaviota endgame tablebases (download [here](https://archive.org/details/Gaviota) under "DOWNLOAD OPTIONS" -> "153 Files")

* Optional: [Stockfish](https://stockfishchess.org/) (tested with 15.1)

NOTE: Stockfish is only required if you want to extract the training data for the neural network by yourself or if you are using the ```game.py``` script with the ```-v 1``` option.

## Setup
* clone the repository on your local machine via ```git clone git@github.com:jaschers/checkmAIte.git```
* go into the checkmAIte directory
* create the virtual environment using the ```environment.yml``` file with ```conda env create --name checkmaite --file=environment.yml``` (or install all the dependencies manually with ```pip install -r requirements.txt```)
* activate the environment with ```conda activate checkmaite```
* Unzip the Gaviota.zip file and copy all the files within that folder into the ```endgame/gaviota/data``` directory.

#### Optional:

Stockfish
* Download the Stockfish engine from https://stockfishchess.org/download/.
* Extract the downloaded file.
* Add the path to the Stockfish engine to the .bashrc file as STOCKFISHPATH=/path/to/stockfish.

Human games
* Download a selection of human games that can be used to train the neural network [here](https://drive.google.com/file/d/1vjP3BQiILpAcB4kiO1h_eeQyfcU3sPfc/view?usp=sharing). This data was originally downloaded from https://rebel13.nl/ but does not seem to be available anymore.
* Copy the zip file into the ```data/``` directory.
* Unzip the file. This will create a ```data/human_games/``` directory with multiple ```.epd``` files.

## Usage
If you immediatly like to play against CheckmAIte proceed to Section "Play against CheckmAIte
". If you want to extract your own training data and train a neural network on your own, you can proceed as follows.

NOTE: Each script in the ```scripts/``` directory has a ```-h``` option to get furhter informations about the script and its input parameters. Simply run ```python scripts/<script_name>.py -h```.

### Extract training data
I provide multiple options to extract training data for a neural network based on different chess game scenarios. These include chess boards generated from
- Random moves
- Human games
- Draws by the seventyfive moves and fivefold repetition rules
- Draws by stalemate
- Checkmate that include a pinned piece or pawn

For each chess board, the following information is extracted and saved into an HDF5 file:
- 3D chess board with shape (30, 8, 8) with board_i representing:
    - 0: all squares covered by white pawn
    - 1: all squares covered by white knight
    - 2: all squares covered by white bishop
    - 3: all squares covered by white rook
    - 4: all squares covered by white queen
    - 5: all squares covered by white king
    - 6: all squares covered by black pawn
    - 7: all squares covered by black knight
    - 8: all squares covered by black bishop
    - 9: all squares covered by black rook
    - 10: all squares covered by black queen
    - 11: all squares covered by black king
    - 12: all squares being attacked/defended by white pawn
    - 13: all squares being attacked/defended by white knight
    - 14: all squares being attacked/defended by white bishop
    - 15: all squares being attacked/defended by white rook
    - 16: all squares being attacked/defended by white queen
    - 17: all squares being attacked/defended by white king
    - 18: all squares being attacked/defended by black pawn
    - 19: all squares being attacked/defended by black knight
    - 20: all squares being attacked/defended by black bishop
    - 21: all squares being attacked/defended by black rook
    - 22: all squares being attacked/defended by black queen
    - 23: all squares being attacked/defended by black king
    - 24: all squares being a potential move by white pawns
    - 25: all squares being a potential move by black pawns
    - 26: all squares being pinned by white with a white piece or pawn on that square
    - 27: all squares being pinned by black with a black piece or pawn on that square
    - 28: all squares being possible en passant moves for white
    - 29: all squares being possible en passant moves for black
- side to move
- halfmove clock number
- fullmove number
- check
- checkmate
- stalemate
- insufficient winning material for white
- insufficient winning material for black
- seventy-five-move rule
- fivefold repetition
- castling right king side of white
- castling right queen side of white
- castling right king side of black
- castling right queen side of black
- Stockfish evaluation

#### Random boards
Data based on random moves can be extracted using the following command

```
python scripts/create_data_random.py --number_runs 10 --number_boards 10000 --starting_run 0
```

This creates 10 HDF files with 10000 boards each for each file starting with a filename counter of 0. If you want to extract any addtional data at any point, you should specify --starting_run in order to avoid overwriting your data. The data is saved as ```data/30_8_8_depth0_mm100_ms15000/data<i>.h5```.

#### Human boards
In order to create data from human games, you need to download human games in FEN format [here](https://rebel13.nl/download/data.html). Save the files into the ```data/human_games/``` directory. Then, run the following script:

```
python scripts/create_data_human.py --file_id 2 --number_boards 10000 --starting_run 0
```

This extracts data from human games in the ```data/human_games/sf-eval-2.epd``` file. Each ```.epd``` file contains about 600 000 games. If e.g. ```--number_boards 10000``` is specified, that would lead to about 60 HDF files containing 10 000 boards each. The data is saved as ```data/30_8_8_depth0_ms15000_human/data<file_id>-<i>.h5```. 

#### Special boards
The following scenarios are some special cases in chess, which are not that common but still important to train a neural network on. For each of these scenarios, a couple few thousand boards will be generated.

To create a data generated from boards that are a draw due to the seventyfive moves or fivefold repetition rules, run:

```
scripts/create_data_draw.py
```

To create a data generated from boards that are a draw due to stalemate, run:

```
scripts/create_data_pinned_stalemate.py
```

To create a data generated from boards which include a checkmate with a pinned piece or pawn, run:

```
scripts/create_data_pinned_checkmate.py
```

### Train your convolution neural network
Once you have extracted a sufficient amount of training data (at least 1 000 000 boards are recommended), you can train your model using

```
python scripts/train_model.py 
```

The network consists of a convolutional layer, four residual layers, and two dense layers. The input of the network are the 3D chess boards and the board parameters defined above. The output layer is the (stockfish) score of the chess board.
It's required to specify the following options:

- ```--runs```: Number of runs generated from random moves that are loaded from ```data/30_8_8_depth0_mm100_ms15000/``` and used for training the network.
- ```--read_human```: Number of human data runs that are loaded from ```data/30_8_8_depth0_ms15000_human/``` and used for training the network.
- ```--name_experiment```: Name of the experiment used to specify filenames.

Additional optional options are:
- ```--score_cut```: Stockfish score cut applied on the data. 
- ```--read_draw```: Specifies if draw data should be used for training.
- ```--read_pinned```: Specifies if pinned checkmate data should be used for training.
- ```--epochs```: Number of epochs used for training
- ```--batch_size```: Batch size used for training
- ```--activation_function```: Activation function used in the main layers
- ```--loss_function```: The loss function of the network
- ```--verbose```: Verbose level during training.

E.g., the following command trains the neural network on 100 random move runs, 100 human games runs, and all draw runs available with 50 epochs, a batch size of 32, the ReLU activation function, and the mean squared error loss function:

```
python scripts/train_model.py -r 100 -rh 100 -rd y -e 50 -bs 32 -af mse -ne resnet
```

### Play against CheckmAIte
If you did not train the model on your own, you can download a pretrained model [here](https://drive.google.com/file/d/191NBt-cCkAv7UvxaaANC8jwg7qKXXpYW/view?usp=sharing) and copy it into the model/ directory. In order to start a game against CheckmAIte, run the following command:

```
python scripts/game.py -mn model_resnet.h5 -d 3 -jit 1 -mp 1 -c w -v 0 
```

This will start an GUI with a chess board. You can drag and drop the pieces to your desired position to make a move. Afterwards, CheckmAIte will make a move.

In this specific example, CheckmAIte will use the model ```model/model_resnet.h5```, a depth of 3 for the minimax algorithm, just-in-time processing and multiprocessing to speed up the calculations and let you play as the colour white. Additionally, the verbose option is turned off (make sure to properly install stockfish on your device if you want to turn it on). Depending on your setup and device, the ```--jit_compilation```, ```--multiprocessing``` and ```--sound``` option may cause the script to fail. If so, it is recommended to turn them off. A higher ```--depth``` value will make CheckmAIte stronger but also increases the computation time needed to make a move.

Good luck and have fun!

#### Acknowledgements
Thanks to my brother and Ewoud for providing so many useful tips on how to properly play chess. Your input significantly improved the performance of CheckmAIte.