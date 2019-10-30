import numpy as np
import random
from nltk.corpus import wordnet as wn
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('action', '', '')

def get_word(vocab=None, action = None, this_word=None):
    if FLAGS.action == 'no_up' and action == 2:
        return vocab.word2id(this_word)
    if FLAGS.action == 'no_down' and action == 1:
        return vocab.word2id(this_word)
    if FLAGS.action == 'no_same' and action == 3:
        return vocab.word2id(this_word)
    if FLAGS.action == 'no_updownsame' and action == 4:
        return vocab.word2id(this_word)

    if not wn.synsets(this_word) or action==0 or len(this_word)<3:
        return vocab.word2id(this_word)
    elif action == 1:
        word_candidate=[]
        for _,sys in enumerate(wn.synsets(this_word)):
            for hyp in sys.hyponyms():
                for word in hyp.lemma_names():
                    if word in vocab._word2id and word not in this_word and this_word not in word:
                        word_candidate.append(word)
    elif action ==2:
        word_candidate=[]
        for _, sys in enumerate(wn.synsets(this_word)):  # for its every hyponyms()
            for hyp in sys.hypernyms():
                for word in hyp.lemma_names():
                    if word in vocab._word2id and word not in this_word and this_word not in word:
                        word_candidate.append(word)        
    elif action ==3:
        word_candidate=[]
        for _, sys in enumerate(wn.synsets(this_word)):  # for its every hyponyms()
            for word in sys.lemma_names():
                if word in vocab._word2id and word not in this_word and this_word not in word:
                    word_candidate.append(word)
    elif action ==4:
        word_candidate=[]
        for _, sys in enumerate(wn.synsets(this_word)):  
            for hyp in sys.hypernyms():
                for down_hyp in hyp.hyponyms():
                    for word in down_hyp.lemma_names():
                        if word in vocab._word2id and word not in this_word and this_word not in word:
                            word_candidate.append(word)   
    if not word_candidate:
        return vocab.word2id(this_word)
    else:
        return vocab.word2id(sorted(word_candidate)[0])  

def mrpc_gene_new_sentence(vocab,
                          action_logit,
                          sentence,
                          sentence_len,
                          mode=None):

    action_idx = np.argmax(action_logit,axis=2)
    for sent_idx in range(sentence.shape[0]):
        new_sent = sentence[sent_idx,:]
        sent_action = action_idx[sent_idx,:]
        first_sent_start_idx = 1
        second_sent_end_idx = sentence_len[sent_idx]-2
        for idx,word_id in enumerate(new_sent):
            if word_id==102:
                first_sent_end_idx = idx-1
                second_sent_start_idx = idx+1
                break
        if mode==1:
            for idx,this_action in enumerate(sent_action[first_sent_start_idx:first_sent_end_idx+1],start=first_sent_start_idx):
                before_word = new_sent[idx]
                candidate_word = get_word(vocab,this_action,vocab.id2word(new_sent[idx]))
                if before_word != candidate_word:
                    for second_sent_idx,second_sent_word in enumerate(new_sent[second_sent_start_idx:second_sent_end_idx+1],start=second_sent_start_idx):
                        if second_sent_word == before_word:
                            new_sent[second_sent_idx] = candidate_word

        elif mode==2:
            for idx,this_action in enumerate(sent_action[second_sent_start_idx:second_sent_end_idx+1],start=second_sent_start_idx):
                before_word = new_sent[idx]
                candidate_word = get_word(vocab,this_action,vocab.id2word(new_sent[idx]))
                if before_word != candidate_word:
                    for first_sent_idx,first_sent_word in enumerate(new_sent[first_sent_start_idx:first_sent_end_idx+1],start=first_sent_start_idx):
                        if first_sent_word == before_word:
                            new_sent[first_sent_idx] = candidate_word
    return sentence


def single_sentence_generator(vocab=None,
                        action=None,
                        sentence=None,
                        sentence_len=None,
                        print_log=None):
    # action_idx = np.argmax(action, axis=2)
    action_list = []
    for sent in action: # sample action from every position's probability
        word_action=[]
        for word in sent:
            word_action.append(np.random.choice(list(range(len(word))), 1, replace=True, p=word)[0])
        action_list.append(word_action)
    action_idx=np.array(action_list)
    for sent_idx in range(sentence.shape[0]):# for each sentence
        new_sent = sentence[sent_idx,:]
        sent_action = action_idx[sent_idx,:sentence_len[sent_idx]-1]  #
        # flags = 0
        for idx,this_action in enumerate(sent_action): # for each action 
            candidate_word = get_word(vocab,this_action,vocab.id2word(new_sent[idx]))  # return word idx 
            new_sent[idx] = candidate_word
    return sentence,action_idx
