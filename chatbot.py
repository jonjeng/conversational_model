
# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# This script initially translated English utterances to French. It has been modified for our purposes

# The other imports from the __future__ module is not necessary for our present purposes
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange
# the module six contains constants that may differ between python versions. In this case, we wish to import xrange from six because such may provide a noticeable performance (e.g., memory) enhancement, as we are working with large amounts of data

import tensorflow as tf

import seq2seq_model

'''
Instead of using flags, we define a dict in which to store values so that these are easier to read and may be modified more easily    
The stored values are largely hyperparameters, which we [may] wish to modify
'''

params = {
  'mode'               : 'train',
  'train_enc'          : '/home/jjeng/Downloads/chatbot2/train.enc',
  'train_dec'          : '/home/jjeng/Downloads/chatbot2/train.dec',
  'test_enc'           : '/home/jjeng/Downloads/chatbot2/test.enc',
  'test_dec'           : '/home/jjeng/Downloads/chatbot2/test.dec',

  'working_directory'  : '/home/jjeng/Downloads/chatbot2/',

  'enc_vocab_size'     : 77500,
  'dec_vocab_size'     : 77500,


  'num_layers'          : 3,
  'layer_size'          : 256,    # We would like to look at performance with more units in layer, but machine runs out of mem during training (restrictive quota imposed at the Design Center)

  'max_train_data_size' : 0,
  'batch_size'          : 64,

  'steps_per_checkpt'   : 100,

  'learning_rate'       : 1e-3,
  'learning_rate_decay_factor'  : 0.9,
  'max_gradient_norm'   : 5.0,
  'optimizer'           : 'RMSProp'
}

# Define default vocabulary entities (padding, the marker to begin decoding, the end of sentence marker, and the marker for vocabulary in excess of the defined limit)
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

# Define the token IDs of the default vocabulary
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  
  # Read data from the source and target files and put them in the appropriate buckets (of _buckets)
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1

        # Periodically print to the screen the current progress
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

def initialize_vocabulary(vocabulary_path):

  # Obtain the vocabulary and its reverse (such has been found to improve performance, perhaps because temporal correlations are thus more apparent)
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):

  words = tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def create_model(session, forward_only):

  # This function creates a model and initializes or loads parameters from the dict params
  model = seq2seq_model.Seq2SeqModel( params['enc_vocab_size'], params['dec_vocab_size'], _buckets, params['layer_size'], params['num_layers'], params['max_gradient_norm'], params['batch_size'], params['learning_rate'], params['learning_rate_decay_factor'], params['optimizer'], forward_only=forward_only)

  # If a model has been trained, its weights will be stored in an entry in params so load from there using saver.restore to model
  ckpt = tf.train.get_checkpoint_state(params['working_directory'])

  # If a checkpoint file exists, then load the weights from there (again using saver.restore) to model
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)

  else:
  # If no saved weights exist, create a model with 'fresh' parameters (instead of loading preexisting weights)
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():

  # Setup to use GPU for training, using the BFC allocator
  config = tf.ConfigProto()  
  config.gpu_options.allocator_type = 'BFC'

  # Prepare for training
  with tf.Session(config=config) as sess:
    # Create model using create_model() (the previously defined function)
    print("Creating %d layers of %d units." % (params['num_layers'], params['layer_size']))
    model = create_model(sess, False)

    # Read data into corresponding buckets and compute the resulting bucket sizes.
    print ("Reading development and training data (limit: %d)."
           % params['max_train_data_size'])
    enc_dev = params['test_enc'] + (".ids%d" % params['enc_vocab_size'])
    enc_train = params['train_enc'] + (".ids%d" % params['enc_vocab_size'])
    dec_train = params['train_dec'] + (".ids%d" % params['dec_vocab_size'])
    dec_dev = params['test_dec'] + (".ids%d" % params['dec_vocab_size'])
    dev_set = read_data(enc_dev, dec_dev)
    train_set = read_data(enc_train, dec_train, params['max_train_data_size'])
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # Prepare a writer to log results
    writer = open(params['working_directory'] + 'result_log_%d_%d_%d_%f_%f_%s.txt' % (params['enc_vocab_size'], params['layer_size'], params['batch_size'], params['learning_rate'], params['learning_rate_decay_factor'], params['optimizer']), 'a')
    writer.write('enc_vocab_size: %s \nlayer_size: %s\n batch_size: %s \nlearning rate: %s \nlearning rate decay factor: %s \noptimizer: %s \n' % (params['enc_vocab_size'], params['layer_size'], params['batch_size'], params['learning_rate'], params['learning_rate_decay_factor'], params['optimizer']))
    writer.close()

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

    while current_step < 40000: # Define how many iterations to train for
      # Choose a bucket according to data distribution. We pick a random number (using numpy's random)
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / params['steps_per_checkpt']
      loss += step_loss / params['steps_per_checkpt']
      current_step += 1

      # Print the iteration number and write to the log file so results can be recorded and visualized
      if current_step % 1000 == 0:
        print("step %d" % current_step)
        writer = open(params['working_directory'] + 'current_res_log_%d_%d_%d_%f_%f_%s.txt' % (params['enc_vocab_size'], params['layer_size'], params['batch_size'], params['learning_rate'], params['learning_rate_decay_factor'], params['optimizer']), 'a')
        writer.write("\nstep %d\n" % current_step)
        writer.close()

      # Once in a while (defined in 'steps_per_checkpt in params), save a checkpoint, print statistics, and run evals.
      if current_step % params['steps_per_checkpt'] == 0:

        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.8f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        ctime = time.time()
        writer = open(params['working_directory'] + 'result_log_%d_%d_%d_%f_%f_%s.txt' % (params['enc_vocab_size'], params['layer_size'], params['batch_size'], params['learning_rate'], params['learning_rate_decay_factor'], params['optimizer']), 'a')
        writer.write('\ntime: %s' % str(ctime))
        writer.write("\nglobal step %d learning rate %.8f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)

        # Save checkpoint and zero timer and loss.
        if current_step % 40000 == 0 or current_step % 100000 == 0:
          checkpoint_path = os.path.join(params['working_directory'], "seq2seq.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          step_time, loss = 0.0, 0.0

        # Run evals on development set and print their perplexites.
        for bucket_id_ in xrange(len(_buckets)):
          #b_loss = 0.0
          if len(dev_set[bucket_id_]) == 0:
            print("  eval: empty bucket %d" % (bucket_id_))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id_)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id_, True)
          #b_loss += eval_loss / params['steps_per_checkpt']  # Can also change next line to contain math.exp(b_loss) This makes it identical to calculating global perplexity, but doesn't change the fact that individual bucket perplexity increases overall
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id_, eval_ppx))
          writer.write("\n  eval: bucket %d perplexity %.2f" % (bucket_id_, eval_ppx))
        sys.stdout.flush()
      writer.close()
        


def decode():
# This assumes forward propagation only, in 'decoding' a thought vector to a response. The function is named correspondingly
  with tf.Session() as sess:
    
    # Create a model using the create_model() function defined previously
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time (each turn involves a single reply utterance so the batch size (number of sentences to decode) is 1)

    # Load vocabularies
    enc_vocab_path = os.path.join(params['working_directory'],"vocab%d.enc" % params['enc_vocab_size'])
    dec_vocab_path = os.path.join(params['working_directory'],"vocab%d.dec" % params['dec_vocab_size'])
    enc_vocab, _ = initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = initialize_vocabulary(dec_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
      # Set a variable bucket_id to record which bucket the input sentence corresponds to
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if EOS_ID in outputs:
        outputs = outputs[:outputs.index(EOS_ID)]
      # Print out response sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


if __name__ == '__main__':
    if (len(sys.argv) == 2) and (str(sys.argv[1]) == 'train' or str(sys.argv[1]) == 'test'):
        params['mode'] = str(sys.argv[1])
        print('\n>> Mode : %s\n' %(params['mode']))

        if params['mode'] == 'train':
          train()
        elif params['mode'] == 'test':
          decode()
        else:
          print('Error: could not complete')
    else:
        print('Error: must input 1 argument: train or test')
