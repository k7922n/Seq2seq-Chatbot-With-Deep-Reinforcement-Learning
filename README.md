# Seq2seq Chatbot With Deep Reinforcement Learning

Train the conventeional seq2seq model using deep reinforcement learning.
This project is aimed to make chatbot responses more positive.

- Reward Function:
	- Sentiment Analysis Score: Trying make chatbot's response positive.
	- Coherence Score: To make response suitable for the users' input.

## Prerequisites
1. Python packages:
	- Python 3.4 or higher
	- Tensorflow r1.0.1
	- Numpy

2. Clone this repository:
```shell=
git clone https://github.com/Chung-I/Variational-Recurrent-Autoencoder-Tensorflow.git
```

## Usage

Before training the seq2seq model with reinforcement learning, you need to pre-train the seq2seq model and sentiment analysis model.

### Sentiment Analysis Model:

First go to `./sentiment_analysis`

Download sentiment analysis dataset here(XXX) and put it in `sentiment_analyis/corpus` and name is as `SAD.csv`

Run:
`python run.py`

### Pre-train seq2seq model

Go back to `./`

Pre-train the seq2seq model as the coherence reward function and also as the initialization for the reinforcement learning.

Put the dataset in `/corpus` as input sentence dataset named `source` and output sentence named `target` (These two files should contain the same amount of dialogues)

Run:
`python run.py --mode MLE`

Start training seq2seq with reinfocement learning

### Reinforcement Learning

After finish training sentiment analysis model and pre-training seq2seq model.

Run:
`python run.py --mode RL`

Program will automatically read the pre-trained models and start training seq2seq model using reinforcement learning.

### Test Model

Run:
`python run.py --mode TEST`

### Hyperparameters of the run.py
`--vocab_size`: the vocabulary size of the input.
`--hidden_size`: number of units of hidden layer.
`--num_layers`: numbers of the layer.
`--batch_size`: batch size
`mode`: mode of the seq2seq model (MLE, RL, TEST)
`source_data_dir`: the path of the source file
`target_data_dir`: the path of the target file
`model_dir`: directory of the pre-trained seq2seq model
`model_rl_dir`: direcory of the reinforcement learning seq2seq model
`check_step`: step interval of saving model


## File in this project

Folders:
`corpus/`: store the training data.
`model/`: store the pre-trained seq2seq model.
`model_RL/`: store the reinforcement learning seq2seq model.
`sentiment_analysis/`: the code of sentiement analysis.

Files:
`data_utils.py`: Data preprocessing (Tokenizer, load data ...etc).
`seq2seq_model.py`: the core function of the reinforcment learning model.
`seq2seq.py`: some functions modified from tensorflow source code in order to fit the reinforcement learning algorithm. (only this function is from open source)
`run.py`: the load, train, and test function of the whole project.
