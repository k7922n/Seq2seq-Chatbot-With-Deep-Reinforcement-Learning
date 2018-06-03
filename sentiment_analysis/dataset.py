import csv
import re

WORD_SPLIT = re.compile("([.,!?\"':;)(])")
DIGIT_RE = re.compile(r"\d")
DUMMY = re.compile('\.|\,|\@')

_PAD = "PAD"
_GO = "GO"
_EOS = "EOS"
_UNK = "UNK"

_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def tokenizer(sentence):
  words = []
  sentence = DUMMY.sub('', sentence)
  for split_sen in sentence.lower().strip().split():
    words.extend(WORD_SPLIT.split(split_sen))
  return [word for word in words if word]

def form_vocab_mapping(max_size):
  vocab = {}
  data = []
  counter = 0
  f = open('corpus/SAD.csv', 'r')
  for i, line in enumerate(csv.reader(f)):
    counter += 1
    if counter % 100000 == 0:
      print("  Processing to line %s" % counter)

    tokens = tokenizer(line[3])
    for w in tokens:
      word = DIGIT_RE.sub("0", w)
      if word in vocab:
        vocab[word] += 1
      else:
        vocab[word] = 1

  vocab_list = _START_VOCAB + sorted(vocab, key = vocab.get, reverse = True)
  if len(vocab_list) > max_size:
    vocab_list = vocab_list[:max_size]

  with open('corpus/mapping', 'w') as o:
    for w in vocab_list:
      o.write(w + '\n')

def read_map(map_path):
  vocab_list = []
  with open(map_path, 'r') as f:
    vocab_list.extend(f.readlines())

  vocab_list = [line.strip() for line in vocab_list]
  vocab_dict = dict([(x,y) for (y,x) in enumerate(vocab_list)])

  return vocab_dict, vocab_list

def convert_to_token(sentence, vocab_map):
  words = tokenizer(sentence)
  return [vocab_map.get(w, UNK_ID) for w in words]

def file_to_token(file_path, vocab_map):
  output_path = file_path + '.token'
  with open(file_path, 'r') as input_file:
    with open(output_path, 'w') as output_file:
      counter = 0
      for line in csv.reader(input_file):
        counter += 1
        if counter % 100000 == 0:
          print('  Tokenizing line %s' % counter)
        token_ids = convert_to_token(line[3], vocab_map)
        output_file.write(line[1] + ',' + " ".join([str(tok) for tok in token_ids]) + '\n')

def read_data(path):
  data = []
  with open(path, 'r') as i:
    counter = 0
    for line in i.readlines():
      counter += 1
      if counter % 100000 == 0:
        print("  Reading data line %s" % counter)
      data_ids = [int(x) for x in line.split(',')[1].split()]
      data.append((int(line.split(',')[0]), data_ids))

  return data

if __name__ == '__main__':
  #form_vocab_mapping(50000)
  #vocab_map, _ = read_map('corpus/mapping')
  #file_to_token('corpus/SAD.csv', vocab_map)
  #d = read_data('corpus/SAD.csv.token')
  #print(d[0])]
  pass
