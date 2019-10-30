
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

# Dependency imports

import tensorflow as tf
import layers as layers_lib
from get_data import get_dataset
import adversarial_attacking
flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags governing adversarial training are defined in adversarial_losses.py.

# Classifier
flags.DEFINE_integer('num_classes', 2, 'Number of classes for classification')

# Data path
flags.DEFINE_string('data_dir','','Directory path to preprocessed text dataset.')
flags.DEFINE_string('vocab_freq_path', '',
                    'Path to pre-calculated vocab frequency data. If '
                    'None, use FLAGS.data_dir/vocab_freq.txt.')
flags.DEFINE_integer('batch_size', 64, 'Size of the batch.')
flags.DEFINE_integer('num_timesteps', 120, 'Number of timesteps for BPTT')

# Model architechture
flags.DEFINE_bool('bidir_lstm', False, 'Whether to build a bidirectional LSTM.')
flags.DEFINE_bool('single_label', True, 'Whether the sequence has a single '
                                        'label, for optimization.')
flags.DEFINE_integer('rnn_num_layers', 1, 'Number of LSTM layers.')
flags.DEFINE_integer('rnn_cell_size', 2,
                     'Number of hidden units in the LSTM.')

# Vocabulary and embeddings
flags.DEFINE_integer('embedding_dims', 256, 'Dimensions of embedded vector.')
flags.DEFINE_bool('normalize_embeddings', False,
                  'Normalize word embeddings by vocab frequency')

# flags.DEFINE_integer('action_type', 4,  """ zero denotes not change , one denotes change"""  # ANCHOR  before is five  we test when use four action what will change
#                      'action type 0/no action 1/synets 2/upper 3/blew 4/upperdown')

# Optimization
# flags.DEFINE_float('learning_rate_generator', 0, 'lr for generator')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate while fine-tuning.')
flags.DEFINE_float('learning_rate_decay_factor', 1.0,
                   'Learning rate decay factor')
# flags.DEFINE_boolean('sync_replicas', False, 'sync_replica or not')
# flags.DEFINE_integer('replicas_to_aggregate', 1,
#                      'The number of replicas to aggregate')    
flags.DEFINE_float('max_grad_norm', 1.0,
                   'Clip the global gradient norm to this value.')
flags.DEFINE_float('keep_prob_emb', 0.6, 'keep probability on embedding layer.')
flags.DEFINE_float('keep_prob_lstm_out', 0.9,
                   'keep probability on lstm output.')
flags.DEFINE_float('keep_prob_cl_hidden', 0.9,
                   'keep probability on classification hidden layer')
flags.DEFINE_float('keep_prob_dense',0.9,'keep probability on dense layers')
flags.DEFINE_float('generator_learning_rate',0.001,'the learning rate of generator')


class cnnModel(object):
    def __init__(self,cl_logits_input_dim=None):
        self.layers = {}
        self.initialize_vocab()

        self.layers['embedding'] = layers_lib.Embedding(  
            self.vocab_size, FLAGS.embedding_dims, FLAGS.normalize_embeddings,
            self.vocab_freqs, FLAGS.keep_prob_emb, name='embedding') 
        self.layers['embedding_1'] = layers_lib.Embedding( 
            self.vocab_size, FLAGS.embedding_dims, FLAGS.normalize_embeddings,
            self.vocab_freqs, FLAGS.keep_prob_emb,name='embedding_1')

        self.layers['cnn'] = layers_lib.CNN(  
            FLAGS.embedding_dims, FLAGS.keep_prob_emb)
        self.layers['lstm_1'] = layers_lib.BiLSTM(
            FLAGS.rnn_cell_size, FLAGS.rnn_num_layers,name="Bilstm")
        action_type = 5 if FLAGS.action == 'all' else 4
        self.layers['action_select'] = layers_lib.Actionselect(action_type,FLAGS.keep_prob_dense,name='action_output')
        self.layers['cl_logits'] = layers_lib.Project_layer(FLAGS.num_classes,FLAGS.keep_prob_dense,name='project_layer')
        
    def build_train_graph(self,global_step):
        self.initialize_train_dataset()
        self.global_step = global_step
        embedded_one = self.layers['embedding'](self.train_sentence)
        one_next_state = self.layers['cnn'](embedded_one)
        logits = tf.squeeze(self.layers['cl_logits'](one_next_state))
        loss  = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.train_label,logits=logits)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss',loss)
        prediction = tf.equal(tf.cast(tf.argmax(logits,-1), tf.int32),self.train_label)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        train_op = adam_optimize(loss, self.global_step)
        tf.summary.scalar('train_accuracy', accuracy)
        return loss, train_op, accuracy 

    def initialize_vocab(self):
        _,self.vocab  = get_dataset('train')
        self.vocab_freqs = self.vocab.get_freq()
        self.vocab_size = self.vocab.size

    def initialize_train_dataset(self):
        train_dataset,_  = get_dataset('train',self.vocab)
        train_dataset = train_dataset.shuffle(buffer_size = 1000 , seed=321)
        train_dataset = train_dataset.padded_batch(FLAGS.batch_size, 
                        padded_shapes=([FLAGS.num_timesteps],(),()),drop_remainder=True).repeat().prefetch(3*FLAGS.batch_size)
        train_dataset = train_dataset.make_one_shot_iterator()
        self.train_sentence,self.train_sentence_len,self.train_label = train_dataset.get_next()
        
    def initialize_test_dataset(self):
        test_dataset,_ = get_dataset('test',self.vocab)
        test_dataset = test_dataset.padded_batch(FLAGS.batch_size,   # because default padding_value and pad_idx are 0
                        padded_shapes=([FLAGS.num_timesteps],(),())).prefetch(FLAGS.batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()
        self.test_init_op = test_iterator.make_initializer(test_dataset,name='test_init')
        self.test_sentence,self.test_sentence_len,self.test_label = test_iterator.get_next()

    def initialize_dev_dataset(self):
        dev_dataset,_ = get_dataset('dev',self.vocab)
        dev_dataset = dev_dataset.padded_batch(FLAGS.batch_size,   # because default padding_value and pad_idx are 0
                        padded_shapes=([FLAGS.num_timesteps],(),())).prefetch(FLAGS.batch_size*3)
        dev_iterator = dev_dataset.make_one_shot_iterator()
        self.dev_init_op = dev_iterator.make_initializer(dev_dataset,name='dev_init')
        self.dev_sentence,self.dev_sentence_len,self.dev_label = dev_iterator.get_next()

    def build_dev_graph(self):
        self.initialize_dev_dataset()
        embedded_one = self.layers['embedding'](self.dev_sentence,False)
        batch_number = tf.shape(embedded_one)[0]
        # _, one_next_state = self.layers['lstm'](embedded_one, None,self.dev_sentence_len,False)
        one_next_state = self.layers['cnn'](embedded_one,False)
        # logits = tf.squeeze(self.layers['cl_logits'](one_next_state[0].h))
        logits = tf.squeeze(self.layers['cl_logits'](one_next_state,False))
        prediction = tf.equal(tf.cast(tf.argmax(logits,-1), tf.int32),self.dev_label)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        tf.summary.scalar('dev_accuracy', accuracy)
        return accuracy,batch_number,self.dev_init_op      

    def build_test_graph(self):
        self.initialize_test_dataset()
        embedded_one = self.layers['embedding'](self.test_sentence,False)
        batch_number = tf.shape(embedded_one)[0]
        # _, one_next_state = self.layers['lstm'](embedded_one, None,self.test_sentence_len,False)
        one_next_state = self.layers['cnn'](embedded_one,False)
        # logits = tf.squeeze(self.layers['cl_logits'](one_next_state[0].h))

        logits = tf.squeeze(self.layers['cl_logits'](one_next_state,False))
        prediction = tf.equal(tf.cast(tf.argmax(logits,-1), tf.int32),self.test_label)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        tf.summary.scalar('test_accuracy', accuracy)
        return accuracy,batch_number,self.test_init_op

    @property
    def pretrained_variables(self):
        return self.layers['embedding'].trainable_weights

    
    @property
    def dis_pretrained_variables(self):
        return self.layers['embedding'].trainable_weights + self.layers['cnn'].trainable_weights + self.layers['cl_logits'].trainable_weights
   
    def train_generator(self,global_step):
        with tf.name_scope(name='generator'):
            self.initialize_train_dataset()
            self.global_step = global_step
            train_sentence = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size,FLAGS.num_timesteps],name='train_sentence')
            train_sentence_len = tf.placeholder(dtype=tf.int32,shape=[None],name='train_sentence_len')
            reward_score = tf.placeholder(dtype=tf.float32, shape=[None],name="reward_score")   
            # action_idx = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.num_timesteps,3],name='action_idx') # if the choiced action is [[1,2,4,1],[2,4,3,1]] then action idx is [[[0,0,1][0,1,2],[0,2,4][0,3,1]],[[1,0,2],[1,1,4],[1,2,3],[1,3,1]]]
            action_idx = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.num_timesteps],name='action_idx')  #the action of each position in sentence
            embedded = self.layers['embedding_1'](train_sentence)
            lstm_out_all, _, = self.layers['lstm_1'](embedded,None,train_sentence_len)
            lstm_out = tf.concat([lstm_out_all[0],lstm_out_all[1]],-1)  # concat forward and backward lstm_out doing predict
            action_logits = self.layers['action_select'](lstm_out)
            prob = tf.nn.softmax(action_logits,axis=-1)
            # ori_loss = tf.gather_nd(tf.log(prob),action_idx)
            # ori_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(self.action_logits,2),logits=self.action_logits)
            ori_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= action_idx,logits=action_logits)
            reward_score = tf.nn.relu(reward_score) # we don't use negative reward for training
            gene_loss = tf.reduce_mean(reward_score * tf.reduce_sum(ori_loss,-1)) 
            # train_gene_op = adam_optimize(gene_loss,self.global_step)
            train_gene_op = gene_adam_optimize(gene_loss,self.global_step)
            return prob,gene_loss,train_gene_op

    def get_generator_data(self):
        return self.train_sentence,self.train_sentence_len,self.train_label
        
    def get_original_prob(self):
        sentence = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.num_timesteps), name="sentence_original")
        sentence_len = tf.placeholder(dtype=tf.int32,shape=[None],name='sentence_len_original')
        sentence_label = tf.placeholder(dtype=tf.int32,shape=[None],name='sentence_label_original')
        embedded_one = self.layers['embedding'](sentence)
        # _, one_next_state = self.layers['lstm'](embedded_one, None,sentence_len)
        one_next_state = self.layers['cnn'](embedded_one)
        # logits = tf.squeeze(self.layers['cl_logits'](one_next_state[0].h))
        logits = tf.squeeze(self.layers['cl_logits'](one_next_state))
        prob = tf.nn.softmax(logits)
        return tf.gather_nd(prob, tf.stack((tf.range(tf.shape(prob)[0],dtype=tf.int32),sentence_label),axis=1))
    

    def train_discriminator(self):
        with tf.name_scope(name='discriminator') as scope:
            sentence = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.num_timesteps), name="sentence")
            sentence_len = tf.placeholder(dtype=tf.int32,shape=[None],name='sentence_len')
            train_label = tf.placeholder(dtype=tf.int32,shape=[None],name='train_label')
            original_prob = tf.placeholder(dtype=tf.float32,shape=[None],name='original_prob')
            embedded_one = self.layers['embedding'](sentence)
            # _, one_next_state = self.layers['lstm'](embedded_one, None,sentence_len)
            one_next_state = self.layers['cnn'](embedded_one)
            # logits = tf.squeeze(self.layers['cl_logits'](one_next_state[0].h))
            logits = tf.squeeze(self.layers['cl_logits'](one_next_state))
            loss  = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label,logits=logits)
            dis_loss = tf.reduce_mean(loss)
            train_dis_op = adam_optimize(dis_loss, self.global_step)
            tf.summary.scalar('dis_loss', dis_loss)
            softmax_prob = tf.nn.softmax(logits)
            prob = tf.gather_nd(softmax_prob, tf.stack((tf.range(tf.shape(softmax_prob)[0],dtype=tf.int32),train_label),axis=1))
            # train_label=tf.cast(train_label,tf.float32) 
            # reward = prob*train_label+(1-prob)*(1-train_label)
            # reward = tf.abs(original_prob-prob)
            reward = original_prob - prob
            return dis_loss,train_dis_op,reward

    # def train_discriminator(self):
    #     with tf.name_scope(name='discriminator') as scope:
    #         sentence = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.num_timesteps), name="sentence")
    #         sentence_len = tf.placeholder(dtype=tf.int32,shape=[None],name='sentence_len')
    #         train_label = tf.placeholder(dtype=tf.int32,shape=[None],name='train_label')
    #         embedded_one = self.layers['embedding'](sentence)
    #         _, one_next_state = self.layers['lstm'](embedded_one, None,sentence_len)
    #         logits = tf.squeeze(self.layers['cl_logits_1'](one_next_state[0].h))
    #         loss  = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label,logits=logits)
    #         dis_loss = tf.reduce_mean(loss)
    #         train_dis_op = adam_optimize(dis_loss, self.global_step)
    #         tf.summary.scalar('dis_loss', dis_loss)
    #         this_logits = tf.nn.softmax(logits)  
    #         prob = this_logits[:,1]+this_logits[:,3]  # the probability of 1 label
    #         train_label=tf.cast(train_label,tf.float32) 
    #         reward = prob*train_label+(1-prob)*(1-train_label)
    #         return dis_loss,train_dis_op,reward


class rnnModel(object):
    def __init__(self,cl_logits_input_dim=None):
        self.layers = {}
        self.initialize_vocab()
        self.layers['embedding'] = layers_lib.Embedding(  
            self.vocab_size, FLAGS.embedding_dims, FLAGS.normalize_embeddings,
            self.vocab_freqs, FLAGS.keep_prob_emb)
        self.layers['embedding_1'] = layers_lib.Embedding( 
            self.vocab_size, FLAGS.embedding_dims, FLAGS.normalize_embeddings,
            self.vocab_freqs, FLAGS.keep_prob_emb,name='embedding_1')

        self.layers['lstm'] = layers_lib.LSTM(  
            FLAGS.rnn_cell_size, FLAGS.rnn_num_layers)
        self.layers['lstm_1'] = layers_lib.BiLSTM(
            FLAGS.rnn_cell_size, FLAGS.rnn_num_layers,name="Bilstm")
        action_type=5 if FLAGS.action=='all' else 4
        self.layers['action_select'] = layers_lib.Actionselect(action_type,FLAGS.keep_prob_dense,name='action_output')
        self.layers['cl_logits'] = layers_lib.Project_layer(FLAGS.num_classes,FLAGS.keep_prob_dense,name='project_layer')
        
    def build_train_graph(self,global_step):
        self.initialize_train_dataset()
        self.global_step = global_step
        embedded_one = self.layers['embedding'](self.train_sentence)
        _, one_next_state = self.layers['lstm'](embedded_one, None,self.train_sentence_len)
        logits = tf.squeeze(self.layers['cl_logits'](one_next_state[0].h))
        loss  = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.train_label,logits=logits)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss',loss)
        prediction = tf.equal(tf.cast(tf.argmax(logits,-1), tf.int32),self.train_label)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        train_op = adam_optimize(loss, self.global_step)
        tf.summary.scalar('train_accuracy', accuracy)
        return loss, train_op, accuracy 

    def initialize_vocab(self):
        _,self.vocab  = get_dataset('train')
        self.vocab_freqs = self.vocab.get_freq()
        self.vocab_size = self.vocab.size

    def initialize_train_dataset(self):
        train_dataset,_  = get_dataset('train',self.vocab)
        train_dataset = train_dataset.shuffle(buffer_size = 1000 , seed=321)
        train_dataset = train_dataset.padded_batch(FLAGS.batch_size, 
                        padded_shapes=([FLAGS.num_timesteps],(),()),drop_remainder=True).repeat().prefetch(3*FLAGS.batch_size)
        train_dataset = train_dataset.make_one_shot_iterator()
        self.train_sentence,self.train_sentence_len,self.train_label = train_dataset.get_next()
        
    def initialize_test_dataset(self):
        test_dataset,_ = get_dataset('test',self.vocab)
        test_dataset = test_dataset.padded_batch(FLAGS.batch_size,   # because default padding_value and pad_idx are 0
                        padded_shapes=([FLAGS.num_timesteps],(),())).prefetch(FLAGS.batch_size)
        test_iterator = test_dataset.make_one_shot_iterator()
        self.test_init_op = test_iterator.make_initializer(test_dataset,name='test_init')
        self.test_sentence,self.test_sentence_len,self.test_label = test_iterator.get_next()

    def initialize_dev_dataset(self):
        dev_dataset,_ = get_dataset('dev',self.vocab)
        dev_dataset = dev_dataset.padded_batch(FLAGS.batch_size,   # because default padding_value and pad_idx are 0
                        padded_shapes=([FLAGS.num_timesteps],(),())).prefetch(FLAGS.batch_size*3)
        dev_iterator = dev_dataset.make_one_shot_iterator()
        self.dev_init_op = dev_iterator.make_initializer(dev_dataset,name='dev_init')
        self.dev_sentence,self.dev_sentence_len,self.dev_label = dev_iterator.get_next()

    def build_dev_graph(self):
        self.initialize_dev_dataset()
        embedded_one = self.layers['embedding'](self.dev_sentence,False)
        batch_number = tf.shape(embedded_one)[0]
        _, one_next_state = self.layers['lstm'](embedded_one, None,self.dev_sentence_len,False)
        logits = tf.squeeze(self.layers['cl_logits'](one_next_state[0].h,False))
        prediction = tf.equal(tf.cast(tf.argmax(logits,-1), tf.int32),self.dev_label)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        tf.summary.scalar('dev_accuracy', accuracy)
        return accuracy,batch_number,self.dev_init_op      

    def build_test_graph(self):
        self.initialize_test_dataset()
        embedded_one = self.layers['embedding'](self.test_sentence,False)
        batch_number = tf.shape(embedded_one)[0]
        _, one_next_state = self.layers['lstm'](embedded_one, None,self.test_sentence_len,False)
        logits = tf.squeeze(self.layers['cl_logits'](one_next_state[0].h,False))
        prediction = tf.equal(tf.cast(tf.argmax(logits,-1), tf.int32),self.test_label)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        tf.summary.scalar('test_accuracy', accuracy)
        return accuracy,batch_number,self.test_init_op

    @property
    def pretrained_variables(self):
        return self.layers['embedding'].trainable_weights

    @property
    def dis_pretrained_variables(self):
        return self.layers['embedding'].trainable_weights + self.layers['lstm'].trainable_weights + self.layers['cl_logits'].trainable_weights

    def train_generator(self,global_step):
        with tf.name_scope(name='generator'):
            self.initialize_train_dataset()
            self.global_step = global_step
            train_sentence = tf.placeholder(dtype=tf.int32,shape=[FLAGS.batch_size,FLAGS.num_timesteps],name='train_sentence')
            train_sentence_len = tf.placeholder(dtype=tf.int32,shape=[None],name='train_sentence_len')
            reward_score = tf.placeholder(dtype=tf.float32, shape=[None],name="reward_score")   
            action_idx = tf.placeholder(dtype=tf.int32,shape=[None,FLAGS.num_timesteps],name='action_idx')  #the action of each position in sentence
            embedded = self.layers['embedding_1'](train_sentence)
            lstm_out_all, _, = self.layers['lstm_1'](embedded,None,train_sentence_len)
            lstm_out = tf.concat([lstm_out_all[0],lstm_out_all[1]],-1)
            action_logits= self.layers['action_select'](lstm_out)
            prob = tf.nn.softmax(action_logits,axis=-1)
            ori_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels= action_idx,logits=action_logits)
            reward_score = tf.nn.relu(reward_score) # 
            gene_loss = tf.reduce_mean(reward_score * tf.reduce_sum(ori_loss,-1)) 
            train_gene_op = gene_adam_optimize(gene_loss,self.global_step)
            return prob,gene_loss,train_gene_op

    def get_generator_data(self):
        return self.train_sentence,self.train_sentence_len,self.train_label
        
    def get_original_prob(self):
        sentence = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.num_timesteps), name="sentence_original")
        sentence_len = tf.placeholder(dtype=tf.int32,shape=[None],name='sentence_len_original')
        sentence_label = tf.placeholder(dtype=tf.int32,shape=[None],name='sentence_label_original')
        embedded_one = self.layers['embedding'](sentence)
        _, one_next_state = self.layers['lstm'](embedded_one, None,sentence_len)
        logits = tf.squeeze(self.layers['cl_logits'](one_next_state[0].h))
        prob = tf.nn.softmax(logits)
        return tf.gather_nd(prob, tf.stack((tf.range(tf.shape(prob)[0],dtype=tf.int32),sentence_label),axis=1))

    def train_discriminator(self):
        with tf.name_scope(name='discriminator') as scope:
            sentence = tf.placeholder(dtype=tf.int32, shape=(None, FLAGS.num_timesteps), name="sentence")
            sentence_len = tf.placeholder(dtype=tf.int32,shape=[None],name='sentence_len')
            train_label = tf.placeholder(dtype=tf.int32,shape=[None],name='train_label')
            original_prob = tf.placeholder(dtype=tf.float32,shape=[None],name='original_prob')
            embedded_one = self.layers['embedding'](sentence)
            _, one_next_state = self.layers['lstm'](embedded_one, None,sentence_len)
            logits = tf.squeeze(self.layers['cl_logits'](one_next_state[0].h))
            loss  = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_label,logits=logits)
            dis_loss = tf.reduce_mean(loss)
            train_dis_op = adam_optimize(dis_loss, self.global_step)
            tf.summary.scalar('dis_loss', dis_loss)
            softmax_prob = tf.nn.softmax(logits)
            prob = tf.gather_nd(softmax_prob, tf.stack((tf.range(tf.shape(softmax_prob)[0],dtype=tf.int32),train_label),axis=1))
            reward = original_prob-prob
            return dis_loss,train_dis_op,reward

def make_restore_average_vars_dict():
    """Returns dict mapping moving average names to variables."""
    var_restore_dict = {}
    variable_averages = tf.train.ExponentialMovingAverage(0.999)
    for v in tf.global_variables():
        if v in tf.trainable_variables():
            name = variable_averages.average_name(v)
        else:
            name = v.op.name
        var_restore_dict[name] = v
    return var_restore_dict

def adam_optimize(loss,global_step=None):
    return layers_lib.adam_optimize(loss,global_step,FLAGS.learning_rate,FLAGS.max_grad_norm)

def gene_adam_optimize(loss,global_step=None):
    return layers_lib.adam_optimize(loss,global_step,FLAGS.generator_learning_rate,FLAGS.max_grad_norm)
    
def optimize(loss, global_step=None):
    return layers_lib.optimize(
        loss, global_step, FLAGS.max_grad_norm, FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor)
