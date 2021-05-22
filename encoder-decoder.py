## reference made to https://www.tensorflow.org/tutorials/text/nmt_with_attention

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
tf.test.is_gpu_available()
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
predicted_references = './predicted_references' #where the predicted and reference texts go
checkpoint_dir = './training_checkpoints'  # for training weights

def unicode_to_ascii(word):
  return ''.join(c for c in unicodedata.normalize('NFD', word)
      if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sentence):
  sentence = unicode_to_ascii(sentence.lower().strip())
  sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]', " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  sentence = '<start> ' + sentence + ' <end>'
  return sentence


def create_dataset(path):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  sentence_pairs = [[preprocess_sentence(sentence) for sentence in l.split('\t')]  for l in lines]
  return zip(*sentence_pairs)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)
  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, lang_tokenizer

def load_dataset(path):
  inp_lang, targ_lang = create_dataset(path)
  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

input_tensor_train, target_tensor_train, inp_lang, targ_lang = load_dataset(path_to_data_train)
max_length_targ, max_length_inp = target_tensor_train.shape[1], input_tensor_train.shape[1]

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True)

  def call(self, x):
    x = self.embedding(x)
    output, hidden_state, cell_state = self.lstm(x)
    state = [hidden_state, cell_state]
    # hidden_state, cell_state are the encoder states
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

  class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
  
  class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

  ###TRAINING#####
  optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def train_model(EPOCHS):
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            
def evaluate(sentence, candidate=None, id=None):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    inputs = []
    for word in sentence.split(' '):
        if word in inp_lang.word_index:
            inputs.append(inp_lang.word_index[word])

    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    value = 0

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    candidate_words = []
    if candidate != None:
        candidate = candidate + " <end>"
        for word in candidate.split(' '):
            if word in targ_lang.word_index:
                candidate_words.append(targ_lang.word_index[word])

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        if candidate == None:
            predicted_id = tf.argmax(predictions[0]).numpy()
        else:
            predicted_id = candidate_words[t]

        value += predictions[0][predicted_id].numpy()

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, value/t, id, sentence, attention_plot

        result += targ_lang.index_word[predicted_id] + ' '

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, value/t, id, sentence, attention_plot


        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
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
  
 def getRankValue(target_value, unsorted_distribution):
    sorted_distribution = sorted(unsorted_distribution, reverse=True)
    print("sorted_distribution="+str(sorted_distribution))
    for i in range (0, len(sorted_distribution)):
        value = sorted_distribution[i]
        print("value(rank"+str((i+1))+")="+str(value)+" <==> target="+str(target_value))
        if value == target_value: 
            return 1/(i+1)
    return None
  
 def evaluate_model(filename_testdata, checkpoint_dir):
    f_predicted = open(predicted_references+"/dstc8-sgd-predicted.txt", 'w')
    f_reference = open(predicted_references+"/dstc8-sgd-reference.txt", 'w')
    
    candidates, contexts = loadTestData(filename_testdata)
    correct_predictions = 0
    total_predictions = 0
    cumulative_mrr = 0
    recall_at_1 = None
    mrr = None

    for i in range(0, len(contexts)):
        total_predictions += 1
        #best_value = 0
        #best_index = 0
        response = ""
        response_value = 0
        response_attention_plot = []
        target_value = 0
        context = contexts[i]
        reference = candidates[i][0]
        distribution = []
        jobs = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs.append(executor.submit(evaluate, preprocess_sentence(context), None, 0))
            for j in range (0,len(candidates[i])):
                jobs.append(executor.submit(evaluate, preprocess_sentence(context), candidates[i][j], (j+1)))
        
        for future in concurrent.futures.as_completed(jobs):
            candidate_sentence, value_candidate, id, inp_sentence, attention_plot = future.result()
            if id == 0:
                response = candidate_sentence
                response_value = value_candidate
                response_attention_plot = attention_plot
                print(str(i+1)+' '+str(id)+' predicted_sentence:', candidate_sentence, 'value:', value_candidate) 
            else:
                distribution.append(value_candidate)
                print(str(i+1)+' '+str(id)+' candidate_sentence:', candidate_sentence, 'value:', value_candidate) 

            if id == 1: target_value = value_candidate

            #if value_candidate > best_value:
            #    best_value = value_candidate
            #    best_index = id

        rank = getRankValue(target_value, distribution)
        cumulative_mrr += rank
        correct_predictions += 1 if rank == 1 else 0
        
        recall_at_1 = correct_predictions/total_predictions
        mrr = cumulative_mrr/total_predictions
        print(str(i)+' INPUT='+str(context)+' PREDICTED='+str(response)+' REFERENCE='+str(reference)+' CumulativeR@1='+str(recall_at_1)+' ('+str(correct_predictions)+' out of '+str(total_predictions)+') CumulativeMRR='+str(mrr)+' ('+str(cumulative_mrr)+' out of '+str(total_predictions)+')')
        print("")
        #plot_attention(response_attention_plot, context.split(' '), response.split(' '))

        f_predicted.write(response+"\n")
        f_reference.write(reference+"\n")
    
    f_predicted.close()
    f_reference.close()

    print("BLUE Scores=go to https://www.letsmt.eu/Bleu.aspx and provide your *.txt files under "+str(checkpoint_dir))
    print("RECALL@1="+str(recall_at_1))
    print("Mean Reciprocal Rank="+str(mrr))
    
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()
  
  ####MAIN PROGRAM####
  BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 128
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 128
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
num_epochs = 1000
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

## TRAIN MODEL
for filename in os.listdir(checkpoint_dir):
        print("Erasing file "+str(filename))
        os.remove(checkpoint_dir+"/"+filename)
train_model(num_epochs)

## RETRAIN MODEL
'''
def retrain_model(num_epochs):
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  train_model(num_epochs)
retrain_model(retrain_model)
'''

## EVALUATE MODEL
def val_model():
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  evaluate_model(path_to_data_val, checkpoint_dir) ## change path_to_data_val --> path_to_data_test to use test dataset
