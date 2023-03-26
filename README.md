# CheckmAIte

<img src="https://github.com/jaschers/checkmAIte/blob/main/visuals/logo.png" width="250">

CheckmAIte is a chess engine based on a convolutional neural network.

## Demo
<img src="https://github.com/jaschers/checkmAIte/blob/main/visuals/board.gif" width="500">

Here's a demo of the CheckmAIte (white) playing against a 1300 elo chess bot (black).

## Requirements
* Python 3.7+
* Stockfish engine

## Setup

* Download the Stockfish engine from https://stockfishchess.org/download/.
* Extract the downloaded file.
* Add the path to the Stockfish engine to the .bashrc file as STOCKFISHPATH=/path/to/stockfish.
* Clone this repository.
* Create a virtual environment using python -m venv env.
* Activate the virtual environment using source env/bin/activate.
* Install the required packages using pip install -r requirements.txt.
* Rename the .env.sample file to .env.
* Open the .env file and set the STOCKFISH_PATH variable to the path where you extracted the Stockfish engine.

## Usage

To play a game against the chess engine, run the following command:

```python scripts/game.py -s 0 -v 0 -d 2```

The s option stands for save, v for verbose and d for depth. You can change these options as per your requirement.
