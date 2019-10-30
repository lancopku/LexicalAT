import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    return config

def restore_from_checkpoint(sess=None, saver=None, ckpt_dir=None):

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if not ckpt or not ckpt.model_checkpoint_path:
        tf.logging.info('No checkpoint found at %s', ckpt_dir)
        return False
    saver.restore(sess, ckpt.model_checkpoint_path)
    return True