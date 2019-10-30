import numpy as np
import random
from vocab import Vocab
from nltk.corpus import wordnet as wn
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('action', 'all', '')

def get_word(vocab=None, action = None, this_word=None):
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
        word2count = dict((word,vocab._word2count[word]) for word in word_candidate) 
        word2count = sorted(word2count.items(),key=lambda x:x[1], reverse=True)
        return vocab.word2id(word2count[0][0])  #take out the word with the hightest frequency
        
def get_word_no_up(vocab=None, action = None, this_word=None):
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
            for word in sys.lemma_names():
                if word in vocab._word2id and word not in this_word and this_word not in word:
                    word_candidate.append(word)
    elif action ==3:
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
        word2count = dict((word,vocab._word2count[word]) for word in word_candidate) 
        word2count = sorted(word2count.items(),key=lambda x:x[1], reverse=True)
        return vocab.word2id(word2count[0][0])  #take out the word with the hightest frequency

def get_word_no_down(vocab=None, action = None, this_word=None):
    if not wn.synsets(this_word) or action==0 or len(this_word)<3:
        return vocab.word2id(this_word)
    elif action ==1:
        word_candidate=[]
        for _, sys in enumerate(wn.synsets(this_word)):  
            for hyp in sys.hypernyms():
                for word in hyp.lemma_names():
                    if word in vocab._word2id and word not in this_word and this_word not in word:
                        word_candidate.append(word)        
    elif action ==2:
        word_candidate=[]
        for _, sys in enumerate(wn.synsets(this_word)):  
            for word in sys.lemma_names():
                if word in vocab._word2id and word not in this_word and this_word not in word:
                    word_candidate.append(word)
    elif action ==3:
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
        word2count = dict((word,vocab._word2count[word]) for word in word_candidate) 
        word2count = sorted(word2count.items(),key=lambda x:x[1], reverse=True)
        return vocab.word2id(word2count[0][0])  #take out the word with the hightest frequency

def get_word_no_same(vocab=None, action = None, this_word=None):
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
        for _, sys in enumerate(wn.synsets(this_word)):  
            for hyp in sys.hypernyms():
                for word in hyp.lemma_names():
                    if word in vocab._word2id and word not in this_word and this_word not in word:
                        word_candidate.append(word)        
    elif action ==3:
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
        word2count = dict((word,vocab._word2count[word]) for word in word_candidate) 
        word2count = sorted(word2count.items(),key=lambda x:x[1], reverse=True)
        return vocab.word2id(word2count[0][0])  #take out the word with the hightest frequency

def get_word_no_updownsame(vocab=None, action = None, this_word=None):
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
        for _, sys in enumerate(wn.synsets(this_word)):  
            for hyp in sys.hypernyms():
                for word in hyp.lemma_names():
                    if word in vocab._word2id and word not in this_word and this_word not in word:
                        word_candidate.append(word)        
    elif action ==3:
        word_candidate=[]
        for _, sys in enumerate(wn.synsets(this_word)):  
            for word in sys.lemma_names():
                if word in vocab._word2id and word not in this_word and this_word not in word:
                    word_candidate.append(word)
    if not word_candidate:
        return vocab.word2id(this_word)
    else:
        word2count = dict((word,vocab._word2count[word]) for word in word_candidate) 
        word2count = sorted(word2count.items(),key=lambda x:x[1], reverse=True)
        return vocab.word2id(word2count[0][0])  #take out the word with the hightest frequency


def generate_new_sentence_with_action(vocab=None,
                        action=None,
                        sentence=None,
                        sentence_len=None):
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
        for idx,this_action in enumerate(sent_action): # for each action 
            if FLAGS.action=='no_up':
                candidate_word = get_word_no_up(vocab,this_action,vocab.id2word(new_sent[idx]))  # return word idx 
            elif FLAGS.action=='no_down':
                candidate_word = get_word_no_down(vocab,this_action,vocab.id2word(new_sent[idx]))  # return word idx 
            elif FLAGS.action=='no_same':
                candidate_word = get_word_no_same(vocab,this_action,vocab.id2word(new_sent[idx]))  # return word idx 
            elif FLAGS.action=='no_updownsame':
                candidate_word = get_word_no_updownsame(vocab,this_action,vocab.id2word(new_sent[idx]))  # return word idx 
            else:
                candidate_word = get_word(vocab,this_action,vocab.id2word(new_sent[idx]))  # return word idx 
            new_sent[idx] = candidate_word
    return sentence,action_idx
