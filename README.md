# CheckmAIte

<img src="https://github.com/jaschers/checkmAIte/blob/main/visuals/logo.png" width="250">

CheckmAIte is a chess engine based on a convolutional neural network.

## Demo
<img src="https://github.com/jaschers/checkmAIte/blob/main/visuals/board.gif" width="500">

Here's a demo of the CheckmAIte (white) playing against a 1300 elo chess bot (black).

## Requirements
* Python 3.7+
* [Stockfish](https://stockfishchess.org/) (tested with 15.1)

NOTE: Stockfish is only required if you want to extract the training data by yourself or if you are using the ```game.py``` script with the ```-v 1``` option.

## Setup
* clone the repository on your local machine via ```git clone git@github.com:jaschers/checkmAIte.git```
* go into the checkmAIte directory
* create the virtual environment using the ```environment.yml``` file with ```conda env create --name checkmaite --file=environment.yml``` (or install all the dependencies manually with ```pip install -r requirements.txt```)
* activate the environment with ```conda activate checkmaite```
* Download the Stockfish engine from https://stockfishchess.org/download/.
* Extract the downloaded file.
* Add the path to the Stockfish engine to the .bashrc file as STOCKFISHPATH=/path/to/stockfish.

## Usage
### Play against CheckmAIte
To play a game against the CheckmAIte, run the following command:

```python scripts/game.py -s 0 -v 0 -d 2```

The `s` option stands for save, `v` for verbose and `d` for depth. You can change these options as per your requirement. More information can be found by running `python scripts/game.py -h`.

### Create training data for the convolutional neural network

```python scripts/create_data.py```
(requires Stockfish)

### Train the convolutional neural network

```python scripts/train_model.py -h```

### Predict validation data with the convolutional neural network

```python scripts/predict_data.py -h```

### Evaluate the validation data with the convolutional neural network

```python scripts/evaluate_model.py -h```