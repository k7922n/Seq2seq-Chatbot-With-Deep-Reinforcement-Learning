from __future__ import absolute_import
from __future__ import division

import re
import sys
#import nltk

import tensorflow as tf
from tensorflow.python.platform import gfile

WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
DIGIT_RE = re.compile(br"\d")
DU_RE = re.compile(b"\!")

_PAD = b"PAD"
_GO = b"GO"
_EOS = b"EOS"
_UNK = b"UNK"

_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Tokenize a sentence into a word list
def tokenizer(sentence):
  sentence = DU_RE.sub(b'', sentence)
  words = []
  for split_sen in sentence.lower().strip().split():
    words.extend(WORD_SPLIT.split(split_sen))
  return [word for word in words if word]

# Form vocab map (vocab to index) according to maxsize
# Temporary combine source and target vocabulary map together
def form_vocab_mapping(filename_1, filename_2, max_size, nltk_tokenizer):
  
  output_path = filename_1 + '.' + str(max_size) + '.mapping'
  
  if gfile.Exists(output_path):
    print('Map file has already been formed!')
  else:
    print('Forming mapping file according to %s and %s' % (filename_1, filename_2))  
    print('Max vocabulary size : %s' % max_size)

    vocab = {}
    with gfile.GFile(filename_1, mode = 'rb') as f_1:
      with gfile.GFile(filename_2, mode = 'rb') as f_2:
        f = [f_1, f_2]
        counter = 0
        for i, fil in enumerate(f):
          print('Processing file %s' % i)
          for line in fil:
            counter += 1
            if counter % 100000 == 0:
              print("  Processing to line %s" % counter)

            line = tf.compat.as_bytes(line) 
            tokens = nltk.word_tokenize(line) if nltk_tokenizer else tokenizer(line)
            for w in tokens:
              word = DIGIT_RE.sub(b"0", w)
              if word in vocab:
                vocab[word] += 1
              else:
                vocab[word] = 1
      
        vocab_list = _START_VOCAB + sorted(vocab, key = vocab.get, reverse = True)
        if len(vocab_list) > max_size:
          vocab_list = vocab_list[:max_size]

        with gfile.GFile(output_path, 'wb') as vocab_file:
          for w in vocab_list:
            vocab_file.write(w + b'\n')

# Read mapping file from map_path
# Return mapping dictionary
def read_map(map_path):

  if gfile.Exists(map_path):
    vocab_list = []
    with gfile.GFile(map_path, mode = 'rb') as f:
      vocab_list.extend(f.readlines())
    
    vocab_list = [tf.compat.as_bytes(line).strip() for line in vocab_list]
    vocab_dict = dict([(x, y) for (y, x) in enumerate(vocab_list)])
    
    return vocab_dict, vocab_list

  else:
    raise ValueError("Vocabulary file %s not found!", map_path)

def convert_to_token(sentence, vocab_map, nltk_tokenizer):
 
  if nltk_tokenizer:
    words = nltk.word_tokenize(sentence)
  else:
    words = tokenizer(sentence)  
  
  return [vocab_map.get(DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def file_to_token(file_path, vocab_map, nltk_tokenizer):
  output_path = file_path + ".token"
  if gfile.Exists(output_path):
    print("Token file %s has already existed!" % output_path)
  else:
    print("Tokenizing data according to %s" % file_path)

    with gfile.GFile(file_path, 'rb') as input_file:
      with gfile.GFile(output_path, 'w') as output_file:
        counter = 0
        for line in input_file:
          counter += 1
          if counter % 100000 == 0:
            print("  Tokenizing line %s" % counter)
          token_ids = convert_to_token(tf.compat.as_bytes(line), vocab_map, nltk_tokenizer)

          output_file.write(" ".join([str(tok) for tok in token_ids]) + '\n')

def prepare_whole_data(input_path_1, input_path_2, max_size, nltk_tokenizer = False):
  form_vocab_mapping(input_path_1, input_path_2, max_size, nltk_tokenizer)
  map_path = input_path_1 + '.' + str(max_size) + '.mapping'  
  vocab_map, _ = read_map(map_path)
  file_to_token(input_path_1, vocab_map, nltk_tokenizer)
  file_to_token(input_path_2, vocab_map, nltk_tokenizer)


def read_data(source_path, target_path, bucket):

  data_set = [[] for _ in range(len(bucket))]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      
      source, target = source_file.readline(), target_file.readline()
      counter = 0     
 
      while source and target:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)

        for bucket_id, (source_size, target_size) in enumerate(bucket):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append((source_ids, target_ids))
            break

        source, target = source_file.readline(), target_file.readline()

  return data_set

# Read token data from tokenized data
def read_token_data(file_path):
  token_path = file_path + '.token'
  if gfile.Exists(token_path):
    data_set = []
    print(" Reading from file %s" % file_path)
    with gfile.GFile(token_path, mode = 'r') as t_file:
      counter = 0
      token_file = t_file.readline()
      while token_file:
        counter += 1
        if counter % 100000 == 0:
          print("  Reading data line %s" % counter)
          sys.stdout.flush()
        token_ids = [int(x) for x in token_file.split()]
        data_set.append(token_ids)
        token_file = t_file.readline()

    return data_set

  else:
    raise ValueError("Can not find token file %s" % token_path)

if __name__ == "__main__":
  prepare_whole_data('corpus/source', 'corpus/target', 60000)
  #data_set_1 = read_token_data('corpus/valid.source')
  #data_set_2 = read_token_data('corpus/valid.target')

