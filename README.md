# Chessai

<img src="https://github.com/jaschers/daily-paper/blob/main/logo/logo.png" width="500">

This is a chess engine built using a convolutional neural network as the evaluation function and a minimax algorithm to find the best move. It uses the popular Stockfish engine as the opponent.
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

## Demo

<img src="https://github.com/jaschers/daily-paper/blob/main/logo/logo.png" width="500">

Here's a demo of the chess engine playing against another player.