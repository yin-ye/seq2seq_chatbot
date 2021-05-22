# Sequence to Sequence Chatbot
Implements to sequence to sequence chatbot learning using Encoder-Decoder and Long Short Term Memory (LSTM) architectures.

## Requirements
- Numpy
- Tensorflow
- Scikit-learn

## Module documentation
Following is a list of functions and classes exported by modules.

### Data Preprocessing
Cleans and tokenizes traini, test, and validation datasets.
- unicode_to_ascii(word): converts unicode text files to Ascii standard.
- preprocess_sentence(sentence): removing punctuation (except fullstop) and adding <start> and <end> tokens to the sentence.
- create_dataset(path): compiles sentence pairs using the path to the dataset.
- tokenize(lang): converts sentences to fixed length vectors and returns tokenized sentences and vocabulary.
- load_dataset(path): compiles the above functions to return vocabulary and sentence vecotrs using the source and target dataset.

 ### Encoder Decoder Model
 - Encoder(self, vocab_size, embedding_dim, enc_units, batch_sz): returns encoder output and states based on initial hidden states and input source dataset.
 - Decoder(self, vocab_size, embedding_dim, dec_units, batch_sz): returns decoder states, outputs and attention weights using encoder outputs.
 - BahdanauAttention(self, units): attention function used with Decoder model.
 - train_model(EPOCHS): trains the model.
 - train_step(inp, targ, enc_hidden): calculates the training batch loss based on input data, target data and encoder hidden states.
 - loss_function(real, pred): defines the loss function to train the model.
  
 ### LSTM Model

