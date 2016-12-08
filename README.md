# This set of files is to create a conversational model using tensorflow.
# They are to be used in this order:
#   1. Obtain a dialogue dataset. We used text scraped from IMSDB (much like Film Corpus 2.0).
#   2. Run extract_text.py on this dataset. This script selectively extracts usable utterances then partitions the extracted text into four files: train.enc, train.dec, test.enc, test.dec (training and testing datasets for the encoder and decoder portions of the seq2seq model used).
#   3. Run data_utils.py on the four resulting files from step 2. This script extracts the enc_vocab_size and dec_vocab_size most common words in each category of datasets and writes these to a separate files, then assigns token ID's to each word and writes this output to separate files, all of which are correspondingly named). Sample outputs for steps 2 and 3 are provided.
#   4. Train a model on the data by entering the following on the command line: >> python chatbot.py train
#       Note: parameters and the correct directory must be specified in the script beforehand. The script chatbot.py refers to seq2seq_model.py (the former uses a seq2seq model, which is defined in the latter) so the files must be in the same directory.
#       The script additionally presumes that one will be training using a GPU with CUDA. As such, it specifies tf.ConfigProto() in line 156 and specifies a memory allocator in line 157.
#       Also, one may optionally choose to visualize perplexity results, which are also printed during training. This can be done using vis_ppx.py and specifying in it a log file which is written to during training.
#       During the training process, one may choose to stop at any point. An alternate time at which to save a checkpoint of model values may be specified.
#    5. When one has a trained model, one may test and/or use it by entering the following on the command line: >> python chatbot.py test
#
# Note: these files contain specifications of the parameters and paths used in developing this model. They can be changed.
#   Also, we must give credit where credit is due. The scripts chatbot.py, data_utils.py, and seq2seq_model.py were adapted from tensorflow (link: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn/translate) for the present particular purposes, as can be surmised from the copyright definitions left unaltered in the scripts.
