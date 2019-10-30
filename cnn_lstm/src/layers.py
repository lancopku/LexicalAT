import tensorflow as tf
K = tf.keras

class Project_layer(object):
    def __init__(self, num_classes, keep_prob, name='project_layer'):
        self.name = name 
        self.keep_prob = keep_prob
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs: 
            self.linear_model = tf.layers.Dense(units=num_classes,activation='relu')
            self.trainable_weights = vs.global_variables()

    def __call__(self, x, is_training=True):
        if self.keep_prob < 1 and is_training:
            x = tf.nn.dropout(x,self.keep_prob)
        return self.linear_model(x)

def cl_logits_subgraph(layer_sizes, input_size, num_classes, keep_prob=1.,name = None):
    """Construct multiple ReLU layers with dropout and a linear layer."""
    subgraph = K.models.Sequential(name=name)
    for i, layer_size in enumerate(layer_sizes):
        if i == 0:
            subgraph.add(K.layers.Dense(layer_size, activation='relu', input_shape=(input_size,)))
        else:
            subgraph.add(K.layers.Dense(layer_size, activation='relu'))

        if keep_prob < 1.:
            subgraph.add(K.layers.Dropout(1. - keep_prob))
    subgraph.add(K.layers.Dense(1 if num_classes == 2 else num_classes))
    return subgraph

class Embedding(object):
    def __init__(self, vocab_size, embedding_dim, normalize=False, vocab_freqs=None, keep_prob=1.,
                name='embedding', **kwargs):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.normalized = normalize
        self.keep_prob = keep_prob
        self.trainable_weights = None
        self.name = name
        self.reuse = None

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            self.var = tf.get_variable(shape=(self.vocab_size, self.embedding_dim), dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-0.25, 0.25), name=self.name)
            if self.normalized:
                assert vocab_freqs is not None
                self.vocab_freqs = tf.constant(vocab_freqs, dtype=tf.float32, shape=(vocab_size, 1))
                self.var = self._normalize(self.var)
            self.trainable_weights = vs.global_variables()


    def __call__(self,x,is_training=True):
        embedded = tf.nn.embedding_lookup(self.var, x)
        if self.keep_prob < 1. and is_training:
            embedded = tf.nn.dropout( embedded, self.keep_prob, noise_shape=(tf.shape(embedded)[0], 1, tf.shape(embedded)[-1]))
        return embedded

    def _normalize(self, emb):
        weights = self.vocab_freqs / tf.reduce_sum(self.vocab_freqs)
        mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev

class CNN(object):
    def __init__(self,embedding_size,keep_prob,name='cnn'):
        self.name = name 
        self.embedding_size = embedding_size
        self.keep_prob = keep_prob
        self.out_unit = embedding_size // 3
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as vs: 
            out_unit = embedding_size//3
            self.cnn_w2 = tf.layers.Conv2D(out_unit,kernel_size=[2,embedding_size],strides=1,padding='valid',use_bias=False)
            self.cnn_w3 = tf.layers.Conv2D(out_unit,kernel_size=[3,embedding_size],strides=1,padding='valid',use_bias=False)
            self.cnn_w4 = tf.layers.Conv2D(out_unit,kernel_size=[4,embedding_size],strides=1,padding='valid',use_bias=False)
            self.cnn_w5 = tf.layers.Conv2D(out_unit,kernel_size=[5,embedding_size],strides=1,padding='valid',use_bias=False)
            self.mlp = tf.layers.Dense(units=embedding_size,activation='relu')
            self.trainable_weights = vs.global_variables()
    
    def __call__(self,x,is_training=True):

        x = tf.expand_dims(x,-1) 
        h_w2 = tf.squeeze(tf.reduce_max(self.cnn_w2(x),axis=1),axis=1)
        h_w3 = tf.squeeze(tf.reduce_max(self.cnn_w3(x),axis=1),axis=1)
        h_w4 = tf.squeeze(tf.reduce_max(self.cnn_w4(x),axis=1),axis=1)
        h_w5 = tf.squeeze(tf.reduce_max(self.cnn_w5(x),axis=1),axis=1)
        h = tf.concat([h_w2,h_w3,h_w4,h_w5],1)
        h = tf.nn.relu(h)
        if self.keep_prob < 1 and is_training:
            h = tf.nn.dropout(h,self.keep_prob)
        h = self.mlp(tf.reshape(h,[-1,self.embedding_size//3 * 4]))
        return h


class LSTM(object):
    """LSTM layer using dynamic_rnn.

    Exposes variables in `trainable_weights` property.
    """

    def __init__(self, cell_size, num_layers=1, name='LSTM'):
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x, initial_state, seq_length, is_training=True):
        
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            cell = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.BasicLSTMCell(
                    self.cell_size,
                    forget_bias=0.0,
                    reuse=tf.get_variable_scope().reuse)
                for _ in range(self.num_layers)
            ])

            lstm_out, next_state = tf.nn.dynamic_rnn(
                cell, x, initial_state=initial_state, sequence_length=seq_length, dtype=tf.float32)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True

        return lstm_out, next_state
    
class BiLSTM(object):
    """LSTM layer using dynamic_rnn.

    Exposes variables in `trainable_weights` property.
    """

    def __init__(self, cell_size, num_layers=1, name='BiLSTM'):
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.reuse = None
        self.trainable_weights = None
        self.name = name

    def __call__(self, x, initial_state, seq_length,is_training=True):    
        with tf.variable_scope(self.name, reuse=self.reuse) as vs:
            cell_fw = tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(
                        self.cell_size,
                        forget_bias=0.0,
                        reuse=tf.get_variable_scope().reuse)
                    for _ in range(self.num_layers)
            ])
            cell_bw = tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(
                        self.cell_size,
                        forget_bias=0.0,
                        reuse=tf.get_variable_scope().reuse)
                    for _ in range(self.num_layers)
            ])
            lstm_out, next_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,cell_bw, x, initial_state_fw=initial_state, initial_state_bw=initial_state, 
                sequence_length=seq_length, dtype=tf.float32)

            if self.reuse is None:
                self.trainable_weights = vs.global_variables()

        self.reuse = True
        return lstm_out, next_state

class SoftmaxLoss(K.layers.Layer):
    """Softmax xentropy loss with candidate sampling."""
    def __init__(self, vocab_size, num_candidate_samples=-1, vocab_freqs=None):
        self.vocab_size = vocab_size
        self.num_candidate_samples = num_candidate_samples
        self.vocab_freqs = vocab_freqs
        super(SoftmaxLoss, self).__init__()
        self.multiclass_dense_layer = K.layers.Dense(self.vocab_size)  

    def build(self, input_shape):
        input_shape = input_shape[0]
        with tf.device('/cpu:0'):
            self.lin_w = self.add_weight(
                shape=(input_shape[-1], self.vocab_size),
                name='lm_lin_w',
                initializer=K.initializers.glorot_uniform())
            self.lin_b = self.add_weight(
                shape=(self.vocab_size,),
                name='lm_lin_b',
                initializer=K.initializers.glorot_uniform())
            self.multiclass_dense_layer.build(input_shape)    
        super(SoftmaxLoss, self).build(input_shape)

    def call(self, inputs):
        x, labels, weights = inputs
        if self.num_candidate_samples > -1:
            assert self.vocab_freqs is not None
            labels_reshaped = tf.reshape(labels, [-1])
            labels_reshaped = tf.expand_dims(labels_reshaped, -1)
            sampled = tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_reshaped,
                num_true=1,
                num_sampled=self.num_candidate_samples,
                unique=True,
                range_max=self.vocab_size,
                unigrams=self.vocab_freqs)
            inputs_reshaped = tf.reshape(x, [-1, int(x.get_shape()[2])])

            lm_loss = tf.nn.sampled_softmax_loss(
                weights=tf.transpose(self.lin_w),
                biases=self.lin_b,
                labels=labels_reshaped,
                inputs=inputs_reshaped,
                num_sampled=self.num_candidate_samples,
                num_classes=self.vocab_size,
                sampled_values=sampled)
            lm_loss = tf.reshape(
                lm_loss,
                [int(x.get_shape()[0]), int(x.get_shape()[1])])
        else:
            logits = self.multiclass_dense_layer(x)
            lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        lm_loss = tf.identity( tf.reduce_sum(lm_loss * weights) / _num_labels(weights), name='lm_xentropy_loss')
        return lm_loss


class Actionselect(object):
    def __init__(self,action_class,keep_prob,name='action_select'):
        self.name = name
        self.keep_prob = keep_prob
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            self.multiclass_dense_layer = K.layers.Dense(action_class,activation='relu')  
            self.trainable_weights = vs.global_variables()
    
    def __call__(self,input_data,is_training=True):
        if self.keep_prob<1 and is_training:
            input_data = tf.nn.dropout(input_data,self.keep_prob)
        return self.multiclass_dense_layer(input_data)

def classification_loss(logits, labels, weights):

    inner_dim = logits.get_shape().as_list()[-1]
    with tf.name_scope('classifier_loss'):
        if inner_dim == 1:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.squeeze(logits, -1), labels=tf.cast(labels, tf.float32))
            # Softmax loss
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
        num_lab = _num_labels(weights)
        tf.summary.scalar('num_labels', num_lab)
        return tf.identity(tf.reduce_sum(weights * loss) / num_lab, name='classification_xentropy')


def accuracy(logits, targets, weights):
    with tf.name_scope('accuracy'):
        eq = tf.cast(tf.equal(predictions(logits), targets), tf.float32)
        return tf.identity(
            tf.reduce_sum(weights * eq) / _num_labels(weights), name='accuracy')


def predictions(logits):
    inner_dim = logits.get_shape().as_list()[-1]
    with tf.name_scope('predictions'):
        # For binary classification
        if inner_dim == 1:
            pred = tf.cast(tf.greater(tf.squeeze(logits, -1), 0.), tf.int64)
        # For multi-class classification
        else:
            pred = tf.argmax(logits, 2)
        return pred


def _num_labels(weights):
    """Number of 1's in weights. Returns 1. if 0."""
    num_labels = tf.reduce_sum(weights)
    num_labels = tf.where(tf.equal(num_labels, 0.), 1., num_labels)
    return num_labels

def adam_optimize(loss, global_step,lr, max_grad_norm):
    optimizer = tf.train.AdamOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -max_grad_norm, max_grad_norm), var) 
                for grad, var in grads_and_vars if grad!=None]
    training_op = optimizer.apply_gradients(capped_gvs)
    return training_op

def optimize(loss, global_step, max_grad_norm, lr, lr_decay, sync_replicas=False, 
            replicas_to_aggregate=1, task_id=0):
    """Builds optimization graph.

    * Creates an optimizer, and optionally wraps with SyncReplicasOptimizer
    * Computes, clips, and applies gradients
    * Maintains moving averages for all trainable variables
    * Summarizes variables and gradients

    Args:
        loss: scalar loss to minimize.
        global_step: integer scalar Variable.
        max_grad_norm: float scalar. Grads will be clipped to this value.
        lr: float scalar, learning rate.
        lr_decay: float scalar, learning rate decay rate.
        sync_replicas: bool, whether to use SyncReplicasOptimizer.
        replicas_to_aggregate: int, number of replicas to aggregate when using
        SyncReplicasOptimizer.
        task_id: int, id of the current task; used to ensure proper initialization
        of SyncReplicasOptimizer.

    Returns:
        train_op
    """
    with tf.name_scope('optimization'):
        # Compute gradients.
        tvars = tf.trainable_variables()
        grads = tf.gradients(
            loss,
            tvars,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        # Clip non-embedding grads
        non_embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
                                        if 'embedding' not in v.op.name]
        embedding_grads_and_vars = [(g, v) for (g, v) in zip(grads, tvars)
                                    if 'embedding' in v.op.name]

        ne_grads, ne_vars = zip(*non_embedding_grads_and_vars)
        ne_grads, _ = tf.clip_by_global_norm(ne_grads, max_grad_norm)
        non_embedding_grads_and_vars = list(zip(ne_grads, ne_vars))

        grads_and_vars = embedding_grads_and_vars + non_embedding_grads_and_vars
        if not global_step:
            opt = tf.train.AdamOptimizer(lr)
            apply_gradient_op = opt.apply_gradients(grads_and_vars)
            return apply_gradient_op
        # Summarize
        _summarize_vars_and_grads(grads_and_vars)

        # Decaying learning rate
        lr = tf.train.exponential_decay(lr, global_step, 1, lr_decay, staircase=True)
        tf.summary.scalar('learning_rate', lr)
        opt = tf.train.AdamOptimizer(lr)
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
        global_step = tf.assign_sub(global_step,1)
        
        # Apply gradients
    if sync_replicas:
        opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate, variable_averages=variable_averages,
                variables_to_average=tvars, total_num_replicas=replicas_to_aggregate)
        apply_gradient_op = opt.apply_gradients(grads_and_vars)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train_op')

        # Initialization ops
        tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, opt.get_chief_queue_runner())
        if task_id == 0:  
            local_init_op = opt.chief_init_op
            tf.add_to_collection('chief_init_op', opt.get_init_tokens_op())
        else:
            local_init_op = opt.local_step_init_op
        tf.add_to_collection('local_init_op', local_init_op)
        tf.add_to_collection('ready_for_local_init_op', opt.ready_for_local_init_op)
    else:
        # Non-sync optimizer
        apply_gradient_op = opt.apply_gradients(grads_and_vars)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = variable_averages.apply(tvars)
    return train_op


def _summarize_vars_and_grads(grads_and_vars):
    tf.logging.info('Trainable variables:')
    tf.logging.info('-' * 60)
    for grad, var in grads_and_vars:
        tf.logging.info(var)
        def tag(name, v=var):
            return v.op.name + '_' + name

        # Variable summary
        mean = tf.reduce_mean(var)
        tf.summary.scalar(tag('mean'), mean)
        with tf.name_scope(tag('stddev')):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(tag('stddev'), stddev)
        tf.summary.scalar(tag('max'), tf.reduce_max(var))
        tf.summary.scalar(tag('min'), tf.reduce_min(var))
        tf.summary.histogram(tag('histogram'), var)

        # Gradient summary
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                grad_values = grad.values
            else:
                grad_values = grad

            tf.summary.histogram(tag('gradient'), grad_values)
            tf.summary.scalar(tag('gradient_norm'), tf.global_norm([grad_values]))
        else:
            tf.logging.info('Var %s has no gradient', var.op.name)
