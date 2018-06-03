import tensorflow as tf
from tensorflow.python.platform import gfile
import random
import os
import sys
import numpy as np
sys.path.append('../sentiment_analysis/')
import dataset
from . import model

VOCAB_SIZE = 10000
BATCH_SIZE = 32
UNIT_SIZE = 256
MAX_LENGTH = 40
CHECK_STEP = 1000.

def create_model(session, mode):
  m = model.discriminator(VOCAB_SIZE,
                          UNIT_SIZE,
                          BATCH_SIZE,
                          MAX_LENGTH,
                          mode)
  ckpt = tf.train.get_checkpoint_state('sentiment_analysis/saved_model/')

  if ckpt:
    print("Reading model from %s" % ckpt.model_checkpoint_path)
    m.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters")
    session.run(tf.global_variables_initializer())

  return m

def train():
   if gfile.Exists('corpus/mapping') and gfile.Exists('corpus/SAD.csv.token'):
     print('Files have already been formed!')
   else:
     dataset.form_vocab_mapping(50000)
     vocab_map, _ = dataset.read_map('corpus/mapping')
     dataset.file_to_token('corpus/SAD.csv', vocab_map)

   d = dataset.read_data('corpus/SAD.csv.token')
   random.shuffle(d)    
   
   train_set = d[:int(0.9 * len(d))]
   valid_set = d[int(-0.1 * len(d)):]

   sess = tf.Session()

   Model = create_model(sess, 'train')
   #Model = create_model(sess, 'valid')
   step = 0
   loss = 0

   while(True):
     step += 1
     encoder_input, encoder_length, target = Model.get_batch(train_set)
     '''
     print(encoder_input)
     print(encoder_length)
     print(target)
     exit()
     '''
     loss_train = Model.step(sess, encoder_input, encoder_length, target)
     loss += loss_train/CHECK_STEP
     if step % CHECK_STEP == 0:
       Model.mode = 'valid'
       temp_loss = 0
       for _ in range(100):
         encoder_input, encoder_length, target = Model.get_batch(valid_set)
         loss_valid = Model.step(sess, encoder_input, encoder_length, target)
         temp_loss += loss_valid/100.
       Model.mode = 'train'
       print("Train Loss: %s" % loss)
       print("Valid Loss: %s" % temp_loss)
       checkpoint_path = os.path.join('saved_model/', 'dis.ckpt')
       Model.saver.save(sess, checkpoint_path, global_step = step)
       print("Model Saved!")
       loss = 0

def evaluate():
  vocab_map, _ = dataset.read_map('corpus/mapping')
  sess = tf.Session()
  Model = create_model(sess, 'test')
  Model.batch_size = 1
  
  sys.stdout.write('>')
  sys.stdout.flush()
  sentence = sys.stdin.readline()

  while(sentence):
    token_ids = dataset.convert_to_token(sentence, vocab_map)
    encoder_input, encoder_length, _ = Model.get_batch([(0, token_ids)]) 
    score = Model.step(sess, encoder_input, encoder_length)
    print('Score: ' + str(score[0][0]))
    print('>', end = '')
    sys.stdout.flush()
    sentence = sys.stdin.readline()
if __name__ == '__main__':
  train()
  #evaluate()
