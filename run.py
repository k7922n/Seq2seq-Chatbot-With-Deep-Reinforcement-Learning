import tensorflow as tf
import numpy as np
import os
import sys 
sys.path.append('sentiment_analysis/')
import math

import data_utils
import seq2seq_model
from sentiment_analysis import run
from sentiment_analysis import dataset

tf.app.flags.DEFINE_integer('vocab_size', 60000, 'vocabulary size of the input')
tf.app.flags.DEFINE_integer('hidden_size', 256, 'number of units of hidden layer')
tf.app.flags.DEFINE_integer('num_layers', 3, 'number of layers')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_string('mode', 'MLE', 'mode of the seq2seq model')
tf.app.flags.DEFINE_string('source_data_dir', 'corpus/source', 'directory of source')
tf.app.flags.DEFINE_string('target_data_dir', 'corpus/target', 'directory of target')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'directory of model')
tf.app.flags.DEFINE_string('model_rl_dir', 'model_RL/', 'directory of RL model')
tf.app.flags.DEFINE_integer('check_step', '300', 'step interval of saving model')

FLAGS = tf.app.flags.FLAGS

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

# mode variable has three different mode:
# 1. MLE
# 2. RL
# 3. TEST
def create_seq2seq(session, mode):

  model = seq2seq_model.Seq2seq(vocab_size = FLAGS.vocab_size,
                                buckets = buckets,
                                size = FLAGS.hidden_size,
                                num_layers = FLAGS.num_layers,
                                batch_size = FLAGS.batch_size,
                                mode = mode)
  
  #if mode != 'TEST':
  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  #else:
  #  ckpt = tf.train.get_checkpoint_state(FLAGS.model_rl_dir)
  
  if ckpt:
    print("Reading model from %s, mode: %s" % (ckpt.model_checkpoint_path, mode))
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Create model with fresh parameters, mode: %s" % mode)
    session.run(tf.global_variables_initializer())
  
  return model

def train_MLE(): 
  data_utils.prepare_whole_data(FLAGS.source_data_dir, FLAGS.target_data_dir, FLAGS.vocab_size)

  # read dataset and split to training set and validation set
  d = data_utils.read_data(FLAGS.source_data_dir + '.token', FLAGS.target_data_dir + '.token', buckets)
  print('Total document size: %s' % sum(len(l) for l in d))

  d_train = [[] for _ in range(len(d))]
  d_valid = [[] for _ in range(len(d))]
  for i in range(len(d)):
    d_train[i] = d[i][:int(0.9 * len(d[i]))]
    d_valid[i] = d[i][int(-0.1 * len(d[i])):]

  train_bucket_sizes = [len(d[b]) for b in range(len(d))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]

  sess = tf.Session()

  model = create_seq2seq(sess, 'MLE')
  step = 0
  loss = 0
  loss_list = []
 
  while(True):
    step += 1

    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number])
    encoder_input, decoder_input, weight = model.get_batch(d_train, bucket_id)
    loss_train, _ = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
    loss += loss_train / FLAGS.check_step
    #print(model.token2word(sen)[0])
    if step % FLAGS.check_step == 0:
      print('Step %s, Training perplexity: %s, Learning rate: %s' % (step, math.exp(loss),
                                sess.run(model.learning_rate))) 
      for i in range(len(d)):
        encoder_input, decoder_input, weight = model.get_batch(d_valid, i)
        loss_valid, _ = model.run(sess, encoder_input, decoder_input, weight, i, forward_only = True)
        print('  Validation perplexity in bucket %s: %s' % (i, math.exp(loss_valid)))
      if len(loss_list) > 2 and loss > max(loss_list[-3:]):
        sess.run(model.learning_rate_decay)
      loss_list.append(loss)  
      loss = 0

      checkpoint_path = os.path.join(FLAGS.model_dir, "MLE.ckpt")
      model.saver.save(sess, checkpoint_path, global_step = step)
      print('Saving model at step %s' % step)

def train_RL():
  g1 = tf.Graph()
  g2 = tf.Graph()
  g3 = tf.Graph()
  sess1 = tf.Session(graph = g1)
  sess2 = tf.Session(graph = g2)
  sess3 = tf.Session(graph = g3)
  # model is for training seq2seq with Reinforcement Learning
  with g1.as_default():
    model = create_seq2seq(sess1, 'RL')
    # we set sample size = ?
    model.batch_size = 5
  # model_LM is for a reward function (language model)
  with g2.as_default():
    model_LM = create_seq2seq(sess2, 'MLE')
    # calculate probibility of only one sentence
    model_LM.batch_size = 1

  def LM(encoder_input, decoder_input, weight, bucket_id):
    return model_LM.run(sess2, encoder_input, decoder_input, weight, bucket_id, forward_only = True)[1]
  # new reward function: sentiment score
  with g3.as_default():
    model_SA = run.create_model(sess3, 'test') 
    model_SA.batch_size = 1
 
  def SA(sentence, encoder_length):
    sentence = ' '.join(sentence)
    token_ids = dataset.convert_to_token(sentence, model_SA.vocab_map)
    encoder_input, encoder_length, _ = model_SA.get_batch([(0, token_ids)])
    return model_SA.step(sess3, encoder_input, encoder_length)[0][0]

  data_utils.prepare_whole_data(FLAGS.source_data_dir, FLAGS.target_data_dir, FLAGS.vocab_size)
  d = data_utils.read_data(FLAGS.source_data_dir + '.token', FLAGS.target_data_dir + '.token', buckets)

  train_bucket_sizes = [len(d[b]) for b in range(len(d))]
  train_total_size = float(sum(train_bucket_sizes))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]

  # make RL object read vocab mapping dict, list  
  model.RL_readmap(FLAGS.source_data_dir + '.' + str(FLAGS.vocab_size) + '.mapping')
  step = 0
  while(True):
    step += 1

    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number])
    
    # the same encoder_input for sampling batch_size times
    #encoder_input, decoder_input, weight = model.get_batch(d, bucket_id, rand = False)    
    encoder_input, decoder_input, weight = model.get_batch(d, bucket_id, rand = False)    
    loss = model.run(sess1, encoder_input, decoder_input, weight, bucket_id, X = LM, Y = SA)
   
    # debug 
    #encoder_input = np.reshape(np.transpose(encoder_input, (1, 0, 2)), (-1, FLAGS.vocab_size))
    #encoder_input = np.split(encoder_input, FLAGS.max_length)

    #print(model.token2word(encoder_input)[0])
    #print(model.token2word(sen)[0])
    
    if step % FLAGS.check_step == 0:
      print('Loss at step %s: %s' % (step, loss))
      checkpoint_path = os.path.join('model_RL', "RL.ckpt")
      model.saver.save(sess1, checkpoint_path, global_step = step)
      print('Saving model at step %s' % step)


def test():
  sess = tf.Session()
  vocab_dict, vocab_list = data_utils.read_map(FLAGS.source_data_dir + '.' + str(FLAGS.vocab_size) + '.mapping')
  model = create_seq2seq(sess, 'TEST')
  model.batch_size = 1
  
  sys.stdout.write("Input sentence: ")
  sys.stdout.flush()
  sentence = sys.stdin.readline()

  while(sentence):
    token_ids = data_utils.convert_to_token(tf.compat.as_bytes(sentence), vocab_dict, False)
    bucket_id = len(buckets) - 1
    for i, bucket in enumerate(buckets):
      if bucket[0] >= len(token_ids):
        bucket_id = i
        break
    # Get a 1-element batch to feed the sentence to the model.
    encoder_input, decoder_input, weight = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    output = model.run(sess, encoder_input, decoder_input, weight, bucket_id)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output]
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
      outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    # Print out French sentence corresponding to outputs.
    print("Syetem reply: " + " ".join([tf.compat.as_str(vocab_list[output]) for output in outputs]))
    print("User input  : ", end="")
    sys.stdout.flush()
    sentence = sys.stdin.readline()

if __name__ == '__main__':
  if FLAGS.mode == 'MLE':
    train_MLE()
  elif FLAGS.mode == 'RL':
    train_RL()
  else:
    test()

