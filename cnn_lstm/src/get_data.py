import tensorflow as tf 
from vocab import *
import re
import os
import functools
import random
from collections import namedtuple
import utils
from pathlib import Path 
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir',"",'the input dir ')
flags.DEFINE_integer('least_freq',2,'the least frequence of a word in vocabulary')
flags.DEFINE_integer('max_vocab_size',100000,'the max size of the vocabulary ')

Document = namedtuple('Document','sentence label is_test is_dev')

def split_by_punct(segment):
  """Splits str segment by punctuation, filters our empties and spaces."""
  return [s for s in re.split(r'\W+', segment) if s and not s.isspace()]

def get_vocab():
    vocab_freq = {}
    for doc in get_example_from_document('train'):
        for token in split_by_punct(doc.sentence):
            token = token.strip()
            if token not in vocab_freq:
                vocab_freq[token] = 1
            else:
                vocab_freq[token] += 1 
    vocab_file_path = Path(FLAGS.input_dir)/'vocab.txt'         
    return makeVocabFormDict(word2count=vocab_freq,max_size=10*10000,least_freq=FLAGS.least_freq,\
            vocab_file=vocab_file_path)            

def get_example_from_document(data_mode = None):
    file_list = [str(filename) for filename in Path(FLAGS.input_dir).glob('*.*')]
    if data_mode =='train':
        data_files = []
        for inp_fname in file_list:
            if 'train' in inp_fname:
                data_files.append(inp_fname)
        for filename in data_files:
            with open(filename,'r') as rt_f:
                for content in rt_f:
                    class_label = int(content[0])
                    sentence = content[2:]
                    yield Document(sentence=sentence,label=class_label,is_test=False,is_dev=False)
    elif data_mode == 'test':
        data_files = []
        for inp_fname in file_list:
            # if inp_fname.endswith('test'):
            if 'test' in inp_fname:
                data_files.append(inp_fname)
        for filename in data_files:
            with open(filename,'r') as sst_f:
                for content in sst_f:
                    class_label = int(content[0])
                    sentence = content[2:]
                    yield Document(sentence=sentence,label=class_label,is_test=False,is_dev=False)
    elif data_mode == 'dev':
        data_files = []
        for inp_fname in file_list:
            # if inp_fname.endswith('dev'):
            if 'dev' in inp_fname:
                data_files.append(inp_fname)
        for filename in data_files:
            with open(filename,'r') as sst_f:
                for content in sst_f:
                    class_label = int(content[0])
                    sentence = content[2:]
                    yield Document(sentence=sentence, label=class_label, is_test=False, is_dev=True)



def generator_fn(data_mode = None, vocab=None):
    for doc in get_example_from_document(data_mode):
        sent_list_one = [vocab.word2id(word.lower()) for word in split_by_punct(doc.sentence)]
        sent_list_one = sent_list_one[:FLAGS.num_timesteps-1] + [EOS_IDX]
        yield sent_list_one,len(sent_list_one),doc.label

def get_dataset(data_mode = None, vocab=None):
    sst_shapes = ([None],(),()) # sentence sentence_len , sentencc_label
    sst_types = (tf.int32,tf.int32, tf.int32)
    vocab_file_path = os.path.join(FLAGS.input_dir,'vocab.txt')
    if vocab:
        pass
    elif os.path.exists(vocab_file_path):
        print('will load vocabulary from %s'%vocab_file_path)
        vocab = Vocab(vocab_file_path)
    else:
        vocab = get_vocab()
    return tf.data.Dataset.from_generator(functools.partial(generator_fn, data_mode, vocab),\
                                    output_shapes=sst_shapes, output_types=sst_types), vocab



