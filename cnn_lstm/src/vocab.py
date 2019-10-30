from pathlib import Path
from collections import OrderedDict

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

PAD_WORD = '<pad>'
UNK_WORD = '<unk> '
BOS_WORD = '<s>'
EOS_WORD = '</s>'

def makeVocabFromText(
            filelist=None,
            max_size=10*10000,
            least_freq=2,
            trunc_len=100,
            filter_len=0,
            print_log=None,
            vocab_file=None,
            encoding_format='utf-8',
            lowercase = True): 

    """ the core of this function is getting a word2count dict and writing it to a .txt file,then use Vocab to read it """
    if print_log:
        print_log("%s: the max vocab size  = %d, least_freq is %d  truncate length = %d" \
                                % ( filelist[0], max_size, least_freq , trunc_len ))
    else:
        print("%s: the max vocab size  = %d, least_freq is %d  truncate length = %d" \
                                % ( filelist[0], max_size, least_freq , trunc_len ))

    """tokenizing sentence and add word to word2count dict"""
    word2count={}
    for filename in filelist:
        with open(filename, 'r', encoding = encoding_format) as f:
            for sent in f:
                tokens = sent.strip().split()
                if 0 < filter_len < len(sent.strip().split()):
                    continue
                if trunc_len > 0:
                    tokens = tokens[:trunc_len]
                for word in tokens:
                    word = word if not lowercase else word.lower()
                    if word not in word2count:
                        word2count[word] = 1 
                    else:
                        word2count[word] += 1

    return makeVocabFormDict(word2count=word2count,max_size=max_size,least_freq=least_freq,\
                    vocab_file=vocab_file,encoding_format=encoding_format,filename=filelist[0])

def makeVocabFormDict(word2count=None,
                    max_size=10*10000,
                    least_freq=2,
                    vocab_file=None,
                    encoding_format='utf-8',
                    filename=None):
    """ generate a vocab from a raw word2coun dictionary"""
    word2count = dict((term, freq) for term,freq in word2count.items() if freq >= least_freq) 
    word2count = sorted(word2count.items(),key=lambda x:x[1], reverse=True)
    word2count = word2count[:max_size]
    
    vocab_path = vocab_file if vocab_file else Path(filename).with_name('vocab.txt')
    
    with open(vocab_path,'w',encoding=encoding_format) as f:
        for word,freq in word2count:
            f.write(word+' '+str(freq)+'\n')

    return Vocab(vocab_path,encoding_format)
    
class Vocab(object):
    
    def __init__(self, vocab_file=None,encoding_format='utf-8'):
        """data is a vocabfile that in it's every line is 'word freq \n' """
        self._id2word = {}
        self._word2id = {}
        self._word2count = {}
        self._count = 0
        self.encoding_format = encoding_format
        # [PAD],[UNK],[START] and [STOP] get the ids 0,1,2,3.
        for w in [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]:
            self._word2id[w] = self._count
            self._id2word[self._count] = w
            self._word2count[w] = 1
            self._count += 1
        assert vocab_file,' vocab_file must not be None'
        self.loadFile(vocab_file)
    

    @property
    def size(self):
        return self._count

    # Load entries from a file.
    def loadFile(self, filename=None):
        # load vocabulary from text , one line in this file looks like 'word 1231\n', the number is 
            # Read the vocab file and add words up to max_size
        with open(filename, 'r',encoding=self.encoding_format) as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD]:
                    raise Exception('<s>, </s>, [UNK], [PAD] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word2id[w] = self._count
                self._id2word[self._count] = w
                self._word2count[w] = int(pieces[1])
                self._count += 1

        print ("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id2word[self._count-1]))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word2id:
            return self._word2id[UNK_WORD]
        return self._word2id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id2word[word_id]
    def get_freq(self):
        return [freq for word,freq in self._word2count.items()]

class rtVocab():
    def __init__(self, vocab_file=None,encoding_format='latin-1'):
        """data is a vocabfile that in it's every line is 'word freq \n' """
        self._id2word = {}
        self._word2id = {}
        self._word2count = {}
        self._count = 0
        self.encoding_format = encoding_format
        assert vocab_file,' vocab_file must not be None'
        self.loadFile(vocab_file)

    @property
    def size(self):
        return self._count
    @property
    def eosid(self):
        return len(self._id2word)-1
    # Load entries from a file.
    def loadFile(self, filename=None):
        with open(filename, 'r',encoding=self.encoding_format) as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in self._word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word2id[w] = self._count
                self._id2word[self._count] = w
                self._word2count[w] = int(pieces[1])
                self._count += 1
        print ("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id2word[self._count-1]))

    def word2id(self, word):
        return self._word2id[word]

    def id2word(self, word_id):
        return self._id2word[word_id]
    def get_freq(self):
        return [freq for word, freq in self._word2count.items()]
