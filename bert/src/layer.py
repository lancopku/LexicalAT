import tensorflow as tf
K = tf.keras

class Embedding(object):
  def __init__(self,
               vocab_size=30522,
               embedding_dim=128,
               keep_prob=1.,
               name='embedding',
               **kwargs):
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.keep_prob = keep_prob
    self.trainable_weights = None
    self.name = name
    self.reuse = None

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
        self.var = tf.get_variable(shape=(self.vocab_size, self.embedding_dim),
                                  dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(-1., 1.), #change -1.0 to -1
                                  name=self.name)

        self.trainable_weights = vs.global_variables()

  def __call__(self,x,is_training=True):
    
    embedded = tf.nn.embedding_lookup(self.var, x)
    if self.keep_prob < 1. and is_training:
      shape = embedded.get_shape().as_list()
      embedded = tf.nn.dropout(
          embedded, self.keep_prob, noise_shape=(shape[0], 1, shape[2]))
    return embedded



class LSTM(object):
  """LSTM layer using dynamic_rnn.

  Exposes variables in `trainable_weights` property.
  """
  def __init__(self, cell_size=128, num_layers=1, keep_prob=1., name='LSTM'):
    self.cell_size = cell_size
    self.num_layers = num_layers
    self.keep_prob = keep_prob
    self.reuse = None
    self.trainable_weights = None
    self.name = name

  def __call__(self, x, initial_state, seq_length,is_training = True ):
    # initial_state = (tf.contrib.rnn.BasicLSTMCell(self.cell_size).zero_state(128,dtype=tf.float32),)
    with tf.variable_scope(self.name, reuse=self.reuse) as vs:
      cell = tf.contrib.rnn.MultiRNNCell([
          tf.contrib.rnn.BasicLSTMCell(
              self.cell_size,
              forget_bias=0.0,
              reuse=tf.get_variable_scope().reuse)
          for _ in range(self.num_layers)
      ])
      lstm_out, next_state = tf.nn.dynamic_rnn(
          cell, x, initial_state=initial_state, sequence_length=seq_length,dtype=tf.float32)

      # shape(lstm_out) = (batch_size, timesteps, cell_size)

      if self.keep_prob < 1. and is_training:
        lstm_out = tf.nn.dropout(lstm_out, self.keep_prob)

      if self.reuse is None:
        self.trainable_weights = vs.global_variables()

    self.reuse = True

    return lstm_out, next_state

class Actionselect(object):

  def __init__(self,
               action_class=5,
               **kwargs):
    self.multiclass_dense_layer = K.layers.Dense(action_class)  
    
  def __call__(self,input_data):
    return self.multiclass_dense_layer(input_data)


# def optimize(loss, 
#              global_step,
#              max_grad_norm,
#              lr,
#              lr_decay,
#              sync_replicas=False,
#              replicas_to_aggregate=1,
#              task_id=0):
#   """Builds optimization graph.
#   * Creates an optimizer, and optionally wraps with SyncReplicasOptimizer
#   * Computes, clips, and applies gradients
#   * Maintains moving averages for all trainable variables
#   * Summarizes variables and gradients

#   Args:
#     loss: scalar loss to minimize.
#     global_step: integer scalar Variable.
#     max_grad_norm: float scalar. Grads will be clipped to this value.
#     lr: float scalar, learning rate.
#     lr_decay: float scalar, learning rate decay rate.
#     sync_replicas: bool, whether to use SyncReplicasOptimizer.
#     replicas_to_aggregate: int, number of replicas to aggregate when using
#       SyncReplicasOptimizer.
#     task_id: int, id of the current task; used to ensure proper initialization
#       of SyncReplicasOptimizer.

#   Returns:
#     train_op
#   """
#   with tf.name_scope('optimization'):
#     # Compute gradients.
#     tvars = tf.trainable_variables()
#     grads = tf.gradients(
#         loss,
#         tvars,
#         aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

#     # Clip non-embedding grads
#     non_embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
#                                     if 'embedding' not in v.op.name]
#     embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
#                                 if 'embedding' in v.op.name]

#     ne_grads, ne_vars = zip(*non_embedding_grads_and_vars)
#     ne_grads, _ = tf.clip_by_global_norm(ne_grads, max_grad_norm)
#     non_embedding_grads_and_vars = list(zip(ne_grads, ne_vars))

#     grads_and_vars = embedding_grads_and_vars + non_embedding_grads_and_vars
#     if not global_step:
#       opt = tf.train.AdamOptimizer(lr)
#       apply_gradient_op = opt.apply_gradients(grads_and_vars)
#       return apply_gradient_op
#     # Summarize
#     _summarize_vars_and_grads(grads_and_vars)

#     # Decaying learning rate
#     lr = tf.train.exponential_decay(
#         lr, global_step, 1, lr_decay, staircase=True)
#     tf.summary.scalar('learning_rate', lr)
#     opt = tf.train.AdamOptimizer(lr)
#     # Track the moving averages of all trainable variables.
#     variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
    
#     # Apply gradients
#     if sync_replicas:
#       opt = tf.train.SyncReplicasOptimizer(
#           opt,
#           replicas_to_aggregate,
#           variable_averages=variable_averages,
#           variables_to_average=tvars,
#           total_num_replicas=replicas_to_aggregate)
#       apply_gradient_op = opt.apply_gradients(
#           grads_and_vars, global_step=global_step)
#       with tf.control_dependencies([apply_gradient_op]):
#         train_op = tf.no_op(name='train_op')

#       # Initialization ops
#       tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS,
#                            opt.get_chief_queue_runner())
#       if task_id == 0:  # Chief task
#         local_init_op = opt.chief_init_op
#         tf.add_to_collection('chief_init_op', opt.get_init_tokens_op())
#       else:
#         local_init_op = opt.local_step_init_op
#       tf.add_to_collection('local_init_op', local_init_op)
#       tf.add_to_collection('ready_for_local_init_op',
#                            opt.ready_for_local_init_op)
#     else:
#       # Non-sync optimizer
#       apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step)
#       with tf.control_dependencies([apply_gradient_op]):
#         train_op = variable_averages.apply(tvars)

#     return train_op