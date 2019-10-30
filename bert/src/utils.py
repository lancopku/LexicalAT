import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
def print_log(filename):
    def print_writer(s):
        print(s)
        with open(filename,'a') as f:
            f.write(str(s).strip()+'\n')
    return print_writer

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
  config.gpu_options.allow_growth=True
  return config

def maybe_restore_pretrained_model(sess, saver_for_restore, model_dir,print_log=None):
    
    """Restores pretrained model if there is no ckpt model."""
    print_log = print if not print_log else print_log
    print_log('maybe restore pretrained model')
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    checkpoint_exists = ckpt and ckpt.model_checkpoint_path
    if checkpoint_exists:
        print_log('Checkpoint exists in FLAGS.train_dir; skipping '
                        'pretraining restore')
        return
    if model_dir:
        pretrain_ckpt = tf.train.get_checkpoint_state(model_dir)
        if not (pretrain_ckpt and pretrain_ckpt.model_checkpoint_path):
            raise ValueError(                
                'Asked to restore model from %s but no checkpoint found.' % model_dir)
        saver_for_restore.restore(sess, pretrain_ckpt.model_checkpoint_path)
        print_log('!!load pretrain model successfully!')

def restore_from_checkpoint(sess=None, saver=None, ckpt_dir=None):
  """Restore model from checkpoint.

  Args:
    sess: Session.
    saver: Saver for restoring the checkpoint.

  Returns:
    bool: Whether the checkpoint was found and restored
  """
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if not ckpt or not ckpt.model_checkpoint_path:
    tf.logging.info('No checkpoint found at %s', ckpt_dir)
    return False

  saver.restore(sess, ckpt.model_checkpoint_path)
  return True