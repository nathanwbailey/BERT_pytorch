# PyTorch Implementation of BERT

Implements BERT in PyTorch, adapted from: https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891

The model is trained using the Cornell Movie-Dialogs Corpus: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

### Code:
The main code is located in the following files:
* main.py - Main entry file for training the network
* model.py - Implements the BERT model
* model_building_blocks.py - Embedding block and Encoder Transformer block for BERT
* dataset.py - Creates the BERT dataset using the Corpus
* train.py - Trains the PyTorch Model
* lint.sh - runs linters on the code

