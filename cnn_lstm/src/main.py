from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import graphs
import itertools
import utils
from hook import EvaluateHook, LoggerHook
import random
import os
from adversarial_attacking import generate_new_sentence_with_action

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir','','Directory for logs and checkpoints.')
flags.DEFINE_string('pretrain_dir','','Directory of the pretrain model.')
flags.DEFINE_string('mode','','train or train_adv or test only')
flags.DEFINE_integer('max_steps', 10000, 'Number of batches to run.')
flags.DEFINE_integer('test_steps', 300, 'test at every step')
flags.DEFINE_integer("every", 1, "one sample generate 2 negative sample")  # number of batches negtive samples
flags.DEFINE_integer("random_seed", 321, "random seed")  
flags.DEFINE_integer('dis_warm_up_step', 500, 'discriminator warm up step')
flags.DEFINE_integer('gene_warm_up_step', 500, 'generator warm up step')
flags.DEFINE_string('model_type','rnn',"")
flags.DEFINE_string('dis_pretrain','','')

"""
checkpoint_dir = FLAGS.train_dir,
save_checkpoint_step = FLAGS.save_step,
hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_step),
LoggerHook(dis_loss,dis_acc,FLAGS.display_step,FLAGS.batch_size,print)
EvaluateHook(eval_acc,FLAGS.eval_step,FLAGS.eval_start_step,print)
"""

def train(model = None):
    assert FLAGS.train_dir, 'train_dir must be given'
    global_step = tf.Variable(0, trainable=False)
    add_global = tf.assign_add(global_step, 1)

    loss_op , train_op, acc_op = model.build_train_graph(global_step)
    dev_acc_op ,dev_num_op ,dev_init_op = model.build_dev_graph()
    test_acc_op, test_num_op, test_init_op = model.build_test_graph()

    train_ckpt_dir = FLAGS.train_dir + '/train_ckpt'
    os.makedirs(train_ckpt_dir, exist_ok=True)
    sum_writer = tf.summary.FileWriter(str(train_ckpt_dir), graph=tf.get_default_graph())
    best_dev_acc = 0.0
    final_acc = 0.0
    saver = tf.train.Saver(max_to_keep=1)
    init = tf.global_variables_initializer()
    with tf.Session(config=utils.get_config()) as sess:
        tf.set_random_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        sess.run(init)
        for _ in itertools.count(1):
            this_global_step = sess.run(add_global)
            if this_global_step >= FLAGS.max_steps + 1:
                break
            _, loss, acc = sess.run([train_op, loss_op, acc_op])
            if this_global_step != 0 and this_global_step % FLAGS.test_steps == 0:
                number = 0
                accuracy = 0.0
                while True:
                    try:
                        acc, num = sess.run([dev_acc_op, dev_num_op])
                        number += num
                        accuracy += acc * num
                    except tf.errors.OutOfRangeError:
                        break
                accuracy /= number 
                print('At step %d. dev num=%d acc=%f.'%(this_global_step, number, accuracy))
                if accuracy > best_dev_acc:
                    best_dev_acc = accuracy
                    print("best acc=%f At step %d."%(best_dev_acc, this_global_step))
                    test_accuracy = 0.
                    test_number = 0
                    while True:
                        try:
                            test_acc, test_num= sess.run([test_acc_op, test_num_op])
                            test_number += test_num
                            test_accuracy += test_acc*test_num
                        except tf.errors.OutOfRangeError:
                            break
                    test_accuracy /= test_number 
                    print('test num=%d acc=%f.'%(test_number,test_accuracy))
                    final_acc = test_accuracy
                    sess.run(test_init_op)
                    save_checkpoint(saver, sess, FLAGS.train_dir, this_global_step)
                summary = tf.Summary()
                summary.value.add(tag='test_acc', simple_value=accuracy)
                summary.value.add(tag='best_dev_acc', simple_value=best_dev_acc)
                sum_writer.add_summary(summary, this_global_step)
                sess.run(dev_init_op)
    sum_writer.close()
    print('Accuracy of test set is %f .'%final_acc)

def train_adv(model=None):
    assert FLAGS.train_dir, 'train_dir must be given'
    print('train dir is %s'%FLAGS.train_dir)
    # global_step = tf.train.get_or_create_global_step()
    global_step = tf.Variable(0, trainable=False)
    add_global = tf.assign_add(global_step, 1)

    action_prob_op, gene_loss_op, train_gene_op = model.train_generator(global_step)
    dis_loss_op, train_dis_op, reward_op = model.train_discriminator()
    train_sentence_op, train_sentence_len_op, train_label_op = model.get_generator_data()
    original_prob_op = model.get_original_prob()
    dev_acc_op, dev_num_op, dev_init_op = model.build_dev_graph()
    test_acc_op, test_num_op, test_init_op = model.build_test_graph()

    train_ckpt_dir = FLAGS.train_dir+'/train_ckpt'
    os.makedirs(train_ckpt_dir, exist_ok=True)
    sum_writer = tf.summary.FileWriter(str(train_ckpt_dir), graph=tf.get_default_graph())
    best_dev_acc = 0.0
    final_acc = 0.0

    average_reward = 0
    all_reward = 0
    all_sent_num = 0
    
    saver = tf.train.Saver(max_to_keep=1)
    init = tf.global_variables_initializer()
    with tf.Session(config=utils.get_config()) as sess:
        tf.set_random_seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        sess.run(init)
        for _ in itertools.count(1):
            this_global_step = sess.run(add_global)
            if this_global_step >= FLAGS.max_steps + 1:
                break
            sentence, sentence_len, label=sess.run([train_sentence_op,
                                                train_sentence_len_op,
                                                train_label_op])
            raw_sentence = sentence.copy()
            if this_global_step < FLAGS.dis_warm_up_step:  # discriminator warm up
                dis_loss, _,= sess.run([dis_loss_op, train_dis_op],feed_dict={
                    'discriminator/sentence:0': sentence,
                    'discriminator/sentence_len:0': sentence_len,
                    'discriminator/train_label:0': label,
                })
                gene_loss = 0.0  
            elif this_global_step < FLAGS.gene_warm_up_step+FLAGS.dis_warm_up_step:  # generator warm up
                original_prob = sess.run(original_prob_op, feed_dict={
                                            'sentence_original:0': sentence,
                                            'sentence_len_original:0': sentence_len,
                                            'sentence_label_original:0': label
                })
                action = sess.run(action_prob_op, feed_dict={
                            'generator/train_sentence:0': sentence,
                            'generator/train_sentence_len:0': sentence_len
                            }
                )
                sentence_new, action_idx = generate_new_sentence_with_action(model.vocab,
                                                    action,
                                                    sentence,
                                                    sentence_len)
                reward = sess.run(reward_op, feed_dict={
                    'discriminator/sentence:0':sentence_new,
                    'discriminator/sentence_len:0':sentence_len,
                    'discriminator/train_label:0':label,
                    'discriminator/original_prob:0':original_prob
                })
                all_sent_num += len(reward)
                all_reward += np.sum(reward)
                average_reward = all_reward/all_sent_num
                reward -= average_reward
                gene_loss, _=sess.run([gene_loss_op, train_gene_op], feed_dict={
                                    'generator/train_sentence:0': raw_sentence,
                                    'generator/train_sentence_len:0': sentence_len,
                                    'generator/reward_score:0': reward,
                                    'generator/action_idx:0': action_idx})
                dis_loss = 0
            else:  # adversarial train
                rand_num = random.choice([1]*FLAGS.every+[0])
                if rand_num!=0:  # train with generated sentences
                    original_prob = sess.run(original_prob_op, feed_dict={
                                            'sentence_original:0': sentence,
                                            'sentence_len_original:0': sentence_len,
                                            'sentence_label_original:0': label
                    })
                    action = sess.run(action_prob_op, feed_dict={
                                'generator/train_sentence:0': sentence,
                                'generator/train_sentence_len:0': sentence_len
                                }
                    )
                    sentence_new, action_idx = generate_new_sentence_with_action(model.vocab,
                                                        action,
                                                        sentence,
                                                        sentence_len
                                                        )

                    dis_loss, _, reward = sess.run([dis_loss_op, train_dis_op, reward_op],feed_dict={
                        'discriminator/sentence:0': sentence_new,
                        'discriminator/sentence_len:0': sentence_len,
                        'discriminator/train_label:0': label,
                        'discriminator/original_prob:0': original_prob
                    })
                    all_sent_num += len(reward)
                    all_reward += np.sum(reward)
                    average_reward = all_reward/all_sent_num
                    reward -= average_reward
                    gene_loss, _=sess.run([gene_loss_op, train_gene_op],feed_dict={
                                        'generator/train_sentence:0': raw_sentence,
                                        'generator/train_sentence_len:0': sentence_len,
                                        'generator/reward_score:0': reward,
                                        'generator/action_idx:0': action_idx})
                else:  # train with original sentence 
                    dis_loss,_,= sess.run([dis_loss_op, train_dis_op],feed_dict={
                        'discriminator/sentence:0': sentence,
                        'discriminator/sentence_len:0': sentence_len,
                        'discriminator/train_label:0': label,
                    })
                    gene_loss = 0.0  

            if this_global_step!=0 and this_global_step % FLAGS.test_steps==0 and this_global_step > FLAGS.dis_warm_up_step + FLAGS.gene_warm_up_step :
                number = 0
                accuracy = 0.0
                while True:
                    try:
                        acc, num = sess.run([dev_acc_op, dev_num_op])
                        number += num
                        accuracy += acc * num
                    except tf.errors.OutOfRangeError:
                        break
                accuracy /= number 
                print('At step %d. dev num=%d acc=%f.'%(this_global_step, number, accuracy))
                if accuracy > best_dev_acc:
                    best_dev_acc = accuracy
                    print("best acc=%f At step %d."%(best_dev_acc, this_global_step))
                    test_accuracy = 0.
                    test_number = 0
                    while True:
                        try:
                            test_acc, test_num= sess.run([test_acc_op, test_num_op])
                            test_number += test_num
                            test_accuracy += test_acc*test_num
                        except tf.errors.OutOfRangeError:
                            break
                    test_accuracy /= test_number 
                    print('test num=%d acc=%f.'%(test_number,test_accuracy))
                    final_acc = test_accuracy
                    sess.run(test_init_op)
                    save_checkpoint(saver, sess, FLAGS.train_dir, this_global_step)
                summary = tf.Summary()
                summary.value.add(tag='test_acc', simple_value=accuracy)
                summary.value.add(tag='best_dev_acc', simple_value=best_dev_acc)
                sum_writer.add_summary(summary, this_global_step)
                sess.run(dev_init_op)
    sum_writer.close()
    print('Accuracy of test set is %f .'%final_acc)

def save_checkpoint(saver=None, sess=None, dir=None, step=None):
    path = os.path.join(dir,'train_ckpt','model_'+str(step)+'.ckpt')
    saver.save(sess, path)

if __name__=='__main__':

    if FLAGS.model_type=='cnn':
        print('loading cnn model')
        if FLAGS.mode == 'train':
            print('train baseline ...')
            train(graphs.cnnModel())
        if FLAGS.mode == 'train_adv': 
            print('train adv...')
            train_adv(graphs.cnnModel())
    elif FLAGS.model_type=='rnn':
        print('loading rnn model')
        if FLAGS.mode == 'train':
            print('train baseline ...')
            train(graphs.rnnModel())
        if FLAGS.mode == 'train_adv': 
            print('train adv...')
            train_adv(graphs.rnnModel())
    else:
        print('model type must be cnn or rnn')

