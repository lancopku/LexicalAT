# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import layer as layer_lib
import numpy as np
import random 
from gene_new_sentence import single_sentence_generator

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float('generator_learning_rate',1e-2,'the initial learning rate for adam')
flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

## my bert
tf.flags.DEFINE_string(
    "train_dir", None,
    "the directory to save checkpoint during train")
tf.flags.DEFINE_integer(
    'every',2,'for each training sample generate how many adversarial sample'
)

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


# class ColaProcessor(DataProcessor):
#   """Processor for the CoLA data set (GLUE version)."""

#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

#   def get_labels(self):
#     """See base class."""
#     return ["0", "1"]

#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       # Only the test set has a header
#       if set_type == "test" and i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       if set_type == "test":
#         text_a = tokenization.convert_to_unicode(line[1])
#         label = "0"
#       else:
#         text_a = tokenization.convert_to_unicode(line[3])
#         label = tokenization.convert_to_unicode(line[1])
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
#     return examples
class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "stsa.binary.train")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "stsa.binary.test")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "stsa.binary.test")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      label = line[0]
      text_a = tokenization.convert_to_unicode(line[1:])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  def read_data(self,data_path):
    line_list = []
    with open(data_path,'r') as f:
      for line in f:
        line_list.append(line)
    return line_list

class Sst5Processor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "stsa.fine.train")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "stsa.fine.test")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "stsa.fine.test")), "test")

  def get_labels(self):
    """See base class."""
    return ["0","1",'2','3','4']

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      label = line[0]
      text_a = tokenization.convert_to_unicode(line[1:])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  def read_data(self,data_path):
    line_list = []
    with open(data_path,'r') as f:
      for line in f:
        line_list.append(line)
    return line_list

class RtProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "rt-polarity.all.train")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "rt-polarity.all.test")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "rt-polarity.all.test")), "test")

  def get_labels(self):
    """See base class."""
    return ["0","1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      label = line[0]
      text_a = tokenization.convert_to_unicode(line[1:])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  def read_data(self,data_path):
    line_list = []
    with open(data_path,'r') as f:
      for line in f:
        line_list.append(line)
    return line_list

class YelpProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "yelp.train")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "yelp.test")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_data(os.path.join(data_dir, "yelp.test")), "test")

  def get_labels(self):
    """See base class."""
    return ["0","1",'2','3','4']

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      label = line[0]
      text_a = tokenization.convert_to_unicode(line[1:])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  def read_data(self,data_path):
    line_list = []
    with open(data_path,'r') as f:
      for line in f:
        line_list.append(line)
    return line_list

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")  
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy( 
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features

# def optimize(loss, global_step=None):
#     return layers_lib.optimize(
#         loss, global_step, FLAGS.max_grad_norm, FLAGS.learning_rate,
#         FLAGS.learning_rate_decay_factor)


def create_generator_model(input_sentence=None,reward_score=None,sentence_len=None,action_idx=None):
  tf.logging.set_verbosity(tf.logging.INFO)
  embedding_layer = layer_lib.Embedding()
  lstm_layer = layer_lib.LSTM()
  action_select_layer = layer_lib.Actionselect()

  embedded = embedding_layer(input_sentence)
  lstm_out,_=lstm_layer(embedded,None,sentence_len)
  action_logits = action_select_layer(lstm_out)
  ori_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action_idx,logits=action_logits)
  reward_score = tf.nn.relu(reward_score)
  gene_loss = tf.reduce_mean(reward_score * tf.reduce_mean(ori_loss,-1))
  prob = tf.nn.softmax(action_logits,axis=-1)
  return prob,gene_loss
  
def get_sentence_len(sentence_mask=None):
  sentence_len=[]
  for sentence in sentence_mask:
    idx = 0
    for word in sentence:
      if word==1:
        idx +=1
      else :
        break
    sentence_len.append(idx)
  return np.array(sentence_len)

# def mrpc_get_reward_score(probabilities,probabilities_original):
def mrpc_get_reward_score(probabilities,probabilities_original,train_label):
  reward_score = []
  for idx,label in enumerate(train_label):
    reward_score.append(probabilities_original[idx][label]-probabilities[idx][label])
  return np.array(reward_score)
  # return probabilities_original - probabilities


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      'sst5': Sst5Processor,
      'rt': RtProcessor,
      'yelp':YelpProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
      raise ValueError(
          "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
      raise ValueError(
          "Cannot use sequence length %d because the BERT model "
          "was only trained up to sequence length %d" %
          (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
      raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
      train_examples = processor.get_train_examples(FLAGS.data_dir)
      num_train_steps = int(
          len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
      num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  # tf.logging.info("*** Features ***")
  # for name in sorted(features.keys()):
  #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

  # input_ids = features["input_ids"]
  # input_mask = features["input_mask"]
  # segment_ids = features["segment_ids"]
  # label_ids = features["label_ids"]
  # is_real_example = None
  # if "is_real_example" in features:
  #     is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
  # else:
  #     is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

  name_to_features = {
      "input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
      """Decodes a record to a TensorFlow example."""
      example = tf.parse_single_example(record, name_to_features)
      # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
      # So cast all int64 to int32.
      for name in list(example.keys()):
          t = example[name]
          if t.dtype == tf.int64:
              t = tf.to_int32(t)
          example[name] = t
      return example

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features( 
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    dataset_train = tf.data.TFRecordDataset(train_file)
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.shuffle(buffer_size=100)

    dataset_train = dataset_train.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=FLAGS.train_batch_size,
            drop_remainder=True))
    train_iterator = dataset_train.make_one_shot_iterator()
    # sentence_train_op,sentence_mask_op,segment_ids_op,label_op = train_iterator.get_next()
    get_train_data_op = train_iterator.get_next()
    global_step = tf.train.get_or_create_global_step()
    # model 
    train_sentence_placeholder = tf.placeholder(dtype=tf.int32,shape=[FLAGS.train_batch_size,FLAGS.max_seq_length],name='train_sentence')
    train_mask_placeholder = tf.placeholder(dtype=tf.int32,shape=[FLAGS.train_batch_size,FLAGS.max_seq_length],name='train_mask')
    train_segment_placeholder = tf.placeholder(dtype=tf.int32,shape=[FLAGS.train_batch_size,FLAGS.max_seq_length],name='train_segment')
    train_label_placeholder = tf.placeholder(dtype=tf.int32,shape=[FLAGS.train_batch_size],name='train_label')
    train_sentence_new_placeholder = tf.placeholder(dtype=tf.int32,shape=[FLAGS.train_batch_size,FLAGS.max_seq_length],name='train_sentence_new')
    reward_score_placeholder = tf.placeholder(dtype=tf.float32,shape=[FLAGS.train_batch_size],name='reward_score')
    train_sentence_len_placeholder = tf.placeholder(dtype=tf.int32,shape=[FLAGS.train_batch_size],name='train_sentence_len')
    action_idx_placeholder = tf.placeholder(dtype=tf.int32,shape=[FLAGS.train_batch_size,FLAGS.max_seq_length],name='action_idx')
    # (total_loss, per_example_loss, logits, probabilities) = create_model(
    # bert_config, True, input_ids, input_mask, segment_ids, label_ids,2, False)
    action_prob_op,gene_loss_op = create_generator_model(train_sentence_placeholder,reward_score_placeholder,train_sentence_len_placeholder,action_idx_placeholder)
    generator_train_op = optimization.create_generator_optimizer(gene_loss_op, FLAGS.generator_learning_rate,num_train_steps,num_warmup_steps)
    this_dataset_class_num =5 if FLAGS.task_name.lower() == 'sst5' or FLAGS.task_name.lower() == 'yelp' else 2
    (total_loss_op, per_example_loss_op, logits_op, probabilities_op) = create_model(
    bert_config, True, train_sentence_new_placeholder, train_mask_placeholder, train_segment_placeholder, train_label_placeholder,this_dataset_class_num, False)
    train_op = optimization.create_optimizer(
        total_loss_op, FLAGS.learning_rate, num_train_steps, num_warmup_steps, FLAGS.use_tpu)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if FLAGS.init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
      if FLAGS.use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    saver = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      this_step = 0
      random.seed(302)
      class Vocab(object):
        def __init__(self,path=None):
          self._id2word={}
          self._word2id={}
          with open(path,'r',encoding='utf-8') as f:
            for idx,word in enumerate(f.readlines()):
              word  = word.strip()
              self._id2word[idx] = word
              self._word2id[word] = idx
        def id2word(self,idx=None):
          return self._id2word[idx]
        def word2id(self,word=None):
          return self._word2id[word]
      vocab = Vocab(FLAGS.vocab_file)

      while this_step < num_train_steps:
        train_values = sess.run(get_train_data_op)
        this_step += 1
        rand_num = random.choice([1]*FLAGS.every+[0])
        if rand_num !=0:  # train generated data
          sentence_len = get_sentence_len(train_values['input_mask'])
          action_prob = sess.run(action_prob_op,feed_dict={'train_sentence:0':train_values['input_ids'],
                                                            'train_sentence_len:0':sentence_len})
          probabilities_original = sess.run(probabilities_op,feed_dict={'train_sentence_new:0':train_values['input_ids'], # NOTE for getting probability, pass value to train_sentence_new
                                                      'train_mask:0':train_values['input_mask'],  
                                                      'train_segment:0':train_values['segment_ids'],
                                                      'train_label:0':train_values['label_ids']})
          original_sentence = train_values['input_ids'].copy()
          # sentence_new = mrpc_gene_new_sentence(vocab,  # TODO change to single sentence replace
          #                                       action_prob,
          #                                       train_values['input_ids'],
          #                                       sentence_len,
          #                                       mode=rand_num)
          sentence_new,action_idx = single_sentence_generator(vocab,
                                                  action_prob,
                                                  train_values['input_ids'],
                                                  sentence_len)
          probabilities,_,dis_loss = sess.run([probabilities_op,train_op,per_example_loss_op],
                                            feed_dict={'train_sentence_new:0':sentence_new,
                                                      'train_mask:0':train_values['input_mask'],
                                                      'train_segment:0':train_values['segment_ids'],
                                                      'train_label:0':train_values['label_ids']})

          reward_score = mrpc_get_reward_score(probabilities,probabilities_original,train_values['label_ids']) #TODO correcting reward function
          gene_loss,_ = sess.run([gene_loss_op,generator_train_op],
                                          feed_dict={'reward_score:0':reward_score,
                                                    'train_sentence:0':original_sentence,
                                                    'train_sentence_len:0':sentence_len,
                                                    'action_idx:0':action_idx})
        else:  # train true data
          _,dis_loss = sess.run([train_op,per_example_loss_op],
                                            feed_dict={'train_sentence_new:0':train_values['input_ids'],
                                                      'train_mask:0':train_values['input_mask'],
                                                      'train_segment:0':train_values['segment_ids'],
                                                      'train_label:0':train_values['label_ids']})    
          gene_loss = 0
        if this_step >30 and this_step%30==0:
          print('At step {}, gene loss is {}, dis loss is {}'.format(this_step,gene_loss,np.mean(dis_loss)))    

      save_path = os.path.join(FLAGS.train_dir,'train_ckpt','model_'+str(this_step)+'.ckpt')
      saver.save(sess,save_path)

  if FLAGS.do_eval:
    tf.reset_default_graph()

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)

    while len(eval_examples) % FLAGS.eval_batch_size != 0:
      eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    # file_based_convert_examples_to_features(
    #     eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features( 
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    dataset_eval = tf.data.TFRecordDataset(eval_file)

    dataset_eval = dataset_eval.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=FLAGS.eval_batch_size,
            drop_remainder=eval_drop_remainder))
    eval_iterator = dataset_eval.make_one_shot_iterator()
    get_eval_data_op = eval_iterator.get_next()
    (_, _, eval_logits_op, _) = create_model(bert_config, False, get_eval_data_op['input_ids'],
                    get_eval_data_op['input_mask'], get_eval_data_op['segment_ids'], get_eval_data_op['label_ids'],this_dataset_class_num, False)

    eval_predictions_op = tf.argmax(eval_logits_op, axis=-1, output_type=tf.int32)
    eval_accuracy_op,eval_accuracy_update_op = tf.metrics.accuracy( 
          labels=get_eval_data_op['label_ids'], predictions=eval_predictions_op, weights=get_eval_data_op['is_real_example'],name='my_metric')

    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    saver_for_restore = tf.train.Saver()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(running_vars_initializer)
      train_ckpt_dir = os.path.join(FLAGS.train_dir,'train_ckpt')
      saver_for_restore.restore(sess,tf.train.latest_checkpoint(train_ckpt_dir))
      while True:
        try:
          sess.run(eval_accuracy_update_op)
        except tf.errors.OutOfRangeError:
          break      
      print('eval acc:',sess.run(eval_accuracy_op))
        

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
