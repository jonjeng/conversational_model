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

# This script (adapted from tensorflow's dataflow script) initially translated English utterances to French. It has been modified for our purposes


# The other importing from the __future__ module is not necessary for our present purposes

# We would like to use print as a function (e.g., print()) while on this older version of python  so we import print_function
from __future__ import print_function

import os
import re
import nltk

from tensorflow.python.platform import gfile

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

# Regular expressions used to tokenize.
_DIGIT_RE = re.compile(br"\d")

def nltk_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(nltk.word_tokenize(sentence))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, normalize_digits=True):
  # Create the vocabulary stores in the given path and according to the given parameters
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = nltk_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      print('>> Full Vocabulary Size :',len(vocab_list))
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")

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

  words = nltk_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path, normalize_digits=True):

  print("Tokenizing data in %s" % data_path)
  vocab, _ = initialize_vocabulary(vocabulary_path)
  with gfile.GFile(data_path, mode="rb") as data_file:
    with gfile.GFile(target_path, mode="w") as tokens_file:
      counter = 0
      for line in data_file:
        counter += 1
        if counter % 100000 == 0:
          print("  tokenizing line %d" % counter)
        token_ids = sentence_to_token_ids(line, vocab, normalize_digits)
        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def main(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size):
  # This function was adapted from https://github.com/suriyadeepan/easy_seq2seq/blob/master/data_utils.py
  # We use a different dataset (in our case, a cleaned version of text scraped from IMSDB).

  # Create vocabularies of the appropriate sizes.
  enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
  dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
  if not gfile.Exists(enc_vocab_path):
      create_vocabulary(dec_vocab_path, train_enc, enc_vocabulary_size, tokenizer)
  if not gfile.Exists(enc_vocab_path):
      create_vocabulary(dec_vocab_path, train_dec, dec_vocabulary_size, tokenizer)

  # Create token ids for the training data.
  enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
  dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
  if not gfile.Exists(enc_train_ids_path):
    data_to_token_ids(train_enc, enc_train_ids_path, enc_vocab_path)
  if not gfile.Exists(dec_train_ids_path):
    data_to_token_ids(train_dec, dec_train_ids_path, dec_vocab_path)

  # Create token ids for the development data.
  enc_dev_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
  dec_dev_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
  if not gfile.Exists(enc_dev_ids_path):
    data_to_token_ids(test_enc, enc_dev_ids_path, enc_vocab_path)
  if not gfile.Exists(dec_dev_ids_path):
    data_to_token_ids(test_dec, dec_dev_ids_path, dec_vocab_path)

  return (enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path)

if __name__ == '__main__':
  working_directory = '/Users/jonathanjeng/Documents/COEN281/Project/vocab/'
  train_enc = working_directory + 'train.enc'
  train_dec = working_directory + 'train.dec'
  test_enc = working_directory + 'test.enc'
  test_dec = working_directory + 'test.dec'
  enc_vocabulary_size = 77500
  dec_vocabulary_size = 77500
  main(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size)