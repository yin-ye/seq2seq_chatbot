import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.test.is_gpu_available()
import keras, tensorflow
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import concurrent.futures
import collections
import unicodedata
import re
import numpy as np
import os
import io
import sys
import math
import time

path_to_data_train = "dataset/dstc8-train.txt"
path_to_data_val = "dataset/dstc8-val-candidates.txt"
path_to_data_test = "dataset/dstc8-test-candidates.txt"
predicted_references = './predicted_references'

def preprocess_sentence(path):
  with open(path, 'r', encoding='utf-8') as f:
    sentence = f.read().split('\n')
    num_samples = len(sentence)
    return sentence, num_samples
  
sentence, num_samples = preprocess_sentence(path_to_data_train)

# initialize character set
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

def character_sequence(sentence):
  for s in sentence[: min(num_samples, len(sentence) - 1)]:
    input_text, target_text = s.split('<start>')
    target_text = '<start>' + target_text + '<end>'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
      if char not in input_characters:
        input_characters.add(char)
    for char in target_text:
      if char not in target_characters:
        target_characters.add(char)
        
character_sequence(sentence)

input_characters = sorted(list(input_characters)) # individual source characters
target_characters = sorted(list(target_characters)) # individual target characters
num_encoder_tokens = len(input_characters)  # number of unique encoder tokens
num_decoder_tokens = len(target_characters) # number of unique decoder tokens
max_encoder_seq_length = max([len(txt) for txt in input_texts]) # max length of encoder sequence
max_decoder_seq_length = max([len(txt) for txt in target_texts]) # max length of decoder sequence

# create source and target vocabulary
input_token_index = dict(
  [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
  [(char, i) for i, char in enumerate(target_characters)])

# initialize encoder, decoder and decoder_target shape
encoder_input_data = np.zeros(
  (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
  dtype='float32')
decoder_input_data = np.zeros(
  (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
  dtype='float32')
decoder_target_data = np.zeros(
  (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
  dtype='float32')

#tokenize (one-hot encode) individual source and target characters
def tokenize():
  for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
      encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
      # decoder_target_data is ahead of decoder_input_data by one timestep
      decoder_input_data[i, t, target_token_index[char]] = 1.
      if t > 0:
        # decoder_target_data will be ahead by one timestep
        # and will not include the start character.
        decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    return encoder_input_data, decoder_input_data, decoder_target_data
tokenize()

###LSTM MODEL###
def lstm_model():
  encoder_inputs = Input(shape=(None, num_encoder_tokens))
  encoder = LSTM(latent_dim, return_state=True)
  encoder_outputs, state_h, state_c = encoder(encoder_inputs)
  encoder_states = [state_h, state_c]

  decoder_inputs = Input(shape=(None, num_decoder_tokens))
  decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
  decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                      initial_state=encoder_states)
  decoder_dense = Dense(num_decoder_tokens, activation='softmax')
  decoder_outputs = decoder_dense(decoder_outputs)

  model = Model(inputs=[encoder_inputs, decoder_inputs], 
              outputs=decoder_outputs)
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  
  
### MAIN PROGRAM####
batch_size = 128  # batch size for training
epochs = 600  # number of epochs to train for
latent_dim = 400 # latent dimensionality of the encoding space

# create checkpoint path according to tensorflow standard
checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)

def train_model():
  model.save_weights(checkpoint_path.format(epoch=0))
  model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          callbacks=[cp_callback],
          epochs=epochs)
train_model()

## inference code
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
  decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
  [decoder_inputs] + decoder_states_inputs,
  [decoder_outputs] + decoder_states)

# reverse-lookup token index to turn sequences back to characters
reverse_input_char_index = dict(
  (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
  (i, char) for char, i in target_token_index.items())

def translate(sentence):
  input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='int')
  for t, char in enumerate(sentence):
    try:
      input_seq[0, t, input_token_index[char]] = 1
    except:
      input_seq[0, t, input_token_index[' ']] = 1

  
  states_value = encoder_model.predict(input_seq)
  
  # generate empty target sequence of length 1 with only the start character
  target_seq = np.zeros((1, 1, num_decoder_tokens))
  target_seq[0, 0, target_token_index['\t']] = 1.
  
  # output sequence loop
  stop_condition = False
  decoded_sentence = ''
  while not stop_condition:
    output_tokens, h, c = decoder_model.predict(
      [target_seq] + states_value)
    
    # sample a token and add the corresponding character to the 
    # decoded sequence
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_char = reverse_target_char_index[sampled_token_index]
    decoded_sentence += sampled_char
    
    # check for the exit condition: either hitting max length
    # or predicting the 'stop' character
    if (sampled_char == '\n' or 
        len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True
      
    # update the target sequence (length 1).
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1.
    
    # update states
    states_value = [h, c]
    
  return decoded_sentence

def loadTestData(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    lines = text.strip().split('\n')

    allCandidates = []
    candidates = []
    contexts = []

    for i in range(0, len(lines)):
        if lines[i].startswith("CONTEXT:"): 
            candidate = lines[i][8:]
            contexts.append(candidate)
            continue
            
        elif len(lines[i].strip()) == 0:
            if i>0: allCandidates.append(candidates)
            candidates = []
            
        else:
            candidate = lines[i][12:]
            candidates.append(candidate)
    
    allCandidates.append(candidates)
    return allCandidates, contexts

def evaluate_model(filename_testdata, checkpoint_dir):
    f_predicted = open(predicted_reference+"/dstc8-sgd-predicted.txt", 'w')
    f_reference = open(predicted_reference+"/dstc8-sgd-reference.txt", 'w')
    
    candidates, contexts = loadTestData(filename_testdata)
    
    for i in range(0, len(contexts)):
        context = contexts[i]
        reference = candidates[i][0]
        
        response = translate(context)
    
        f_predicted.write(response+"\n")
        f_reference.write(reference+"\n")
        f_reference.write("\n")
      
    f_predicted.close()
    f_reference.close()

    print("BLUE Scores=go to https://www.letsmt.eu/Bleu.aspx and provide your *.txt files under "+str(checkpoint_dir))

# evaluate the model
evaluate_model(path_to_data_val, predicted_references)
