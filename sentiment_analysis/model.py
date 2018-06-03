import dataset

import numpy as np
import random
import tensorflow as tf

class discriminator():

  def __init__(self, vocab_size, unit_size, batch_size, max_length, mode):
    self.vocab_size = vocab_size
    self.unit_size = unit_size
    self.batch_size = batch_size
    self.max_length = max_length
    self.mode = mode

    self.build_model()
    self.saver = tf.train.Saver(max_to_keep = 2)

  def build_model(self):
    cell = tf.contrib.rnn.GRUCell(self.unit_size)
    params = tf.get_variable('embedding', [self.vocab_size, self.unit_size])
    self.encoder_input = tf.placeholder(tf.int32, [None, self.max_length])
    embedding = tf.nn.embedding_lookup(params, self.encoder_input)
    self.seq_length = tf.placeholder(tf.int32, [None])
    
    _, hidden_state = tf.nn.dynamic_rnn(cell, embedding, sequence_length = self.seq_length, dtype = tf.float32) 
    
    w = tf.get_variable('w', [self.unit_size, 1])
    b = tf.get_variable('b', [1])
    output = tf.matmul(hidden_state, w) + b

    self.logit = tf.nn.sigmoid(output)

    if self.mode != 'test':
      self.target = tf.placeholder(tf.float32, [None, 1])
      self.loss = tf.reduce_mean(tf.square(self.target - self.logit))

      self.opt = tf.train.AdamOptimizer().minimize(self.loss)
    else:
      self.vocab_map, _ = dataset.read_map('sentiment_analysis/corpus/mapping')


  def step(self, session, encoder_inputs, seq_length, target = None):
    input_feed = {}
    input_feed[self.encoder_input] = encoder_inputs
    input_feed[self.seq_length] = seq_length

    output_feed = []

    if self.mode == 'train':
      input_feed[self.target] = target
      output_feed.append(self.loss)
      output_feed.append(self.opt)
      #output_feed.append(self.encoder_input)
      #output_feed.append(self.target)
      outputs = session.run(output_feed, input_feed)
      #return outputs[0], outputs[2], outputs[3]
      return outputs[0]
    elif self.mode == 'valid':
      input_feed[self.target] = target
      output_feed.append(self.loss)
      outputs = session.run(output_feed, input_feed)
      return outputs[0]
    elif self.mode == 'test':
      output_feed.append(self.logit)
      outputs = session.run(output_feed, input_feed)
      return outputs[0]

  def get_batch(self, data):
    encoder_inputs = []
    encoder_length = []
    target = []

    for i in range(self.batch_size):
      pair = random.choice(data)
      #pair = data[i]
      length = len(pair[1])
      target.append([pair[0]])
      if length > self.max_length:
        encoder_inputs.append(pair[1][:self.max_length])
        encoder_length.append(self.max_length)
      else:
        encoder_pad = [dataset.PAD_ID] * (self.max_length - length)
        encoder_inputs.append(pair[1] + encoder_pad)
        encoder_length.append(length)

    batch_input = np.array(encoder_inputs, dtype = np.int32)
    batch_length = np.array(encoder_length, dtype = np.int32)
    batch_target = np.array(target, dtype = np.float32)

    return batch_input, batch_length, batch_target

if __name__ == '__main__':
  test = discriminator(1000, 100, 32, 1, 50)
