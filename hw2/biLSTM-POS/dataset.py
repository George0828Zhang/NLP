import sys
import csv
import numpy as np
import wordninja
from nltk import word_tokenize
from nltk import pos_tag
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool
import torch
from torch.utils import data
from copy import deepcopy

# This module reads input data and formulate it into data that can be used by the lstm models. It includes preprocessing.
class Dataset:
    def __init__(self, filename, task=0, n_workers=4, testlabel=None):
        lines = self._read_tsv(filename)
        self.tweets = []
        self.labels = []
        self.id2index = {}
        self.lbspace = {0:{'NOT':0, 'OFF':1}, 1:{'TIN':0, 'UNT':1}, 2:{'IND':0, 'GRP':1, "OTH":2}}[task]
        self.n_workers = n_workers
        testset = testlabel!=None
        
        for i, fields in enumerate(lines[1:]):
            #['id', 'tweet', 'subtask_a', 'subtask_b', 'subtask_c']
            tid, tweet = fields[:2]            
            if testset:
                tag = list(self.lbspace.keys())[0] # temperary tag
            else:
                tag = fields[2+task]
            if tag != "NULL":                
                self.tweets.append(tweet)
                self.labels.append([self.lbspace[tag]])
                self.id2index[tid] = i
        
        if testset:
            with open(testlabel, "r", encoding="utf-8") as f:
                for line in f:
                    ids, tag = line.strip().split(',')                    
                    index = self.id2index[ids]
                    self.labels[index] = [self.lbspace[tag]]
        
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.size = len(self.tweets)
        
        print("[info] {} data.".format(self.size))
            
    # preprocessing: cut the sentence into words, including those in hashtags, which are not seperated by whitespace also, perform POS tagging on the segmented sentence.
    def _subtask(self, s):
        s = s.lower()
        tokens = wordninja.split(s)
        #tokens = word_tokenize(s)
        pos = pos_tag(tokens)
        return pos
    
    # Main preprocessing function. Tokenization, POS tagging, vocabulary construction, token to id conversion, padding and truncating. 
    def prepare(self, load_vocabs=None, max_len=512, vcutoff=99999):
        with Pool(self.n_workers) as p:
            chunksize = 100
            pos_tokens = list(p.imap(self._subtask, tqdm(self.tweets, total=self.size), chunksize=chunksize))

        if load_vocabs != None:
            self.vocab = load_vocabs[0]
            self.pos_vocab = load_vocabs[1]
        else:            
            # create vocab
            vocab = {"[PAD]":9999, "[UNK]":9999}
            self.pos_vocab = {}
            
            for pos in pos_tokens:
                for w, t in pos:
                    if w not in vocab:
                        vocab[w] = 0
                    else:
                        vocab[w] += 1
                    if t not in self.pos_vocab:
                        self.pos_vocab[t] = len(self.pos_vocab)

            vocab = sorted(vocab.items(), key=lambda x: -x[1])[:vcutoff]
            self.vocab = {w:i for i,(w,f) in enumerate(vocab)}
        
        # change to index
        self.sequence = []
        self.pos_seqs = []
        
        ######################################
        # lendist = [len(x) for x in pos_tokens]
        # import matplotlib.pyplot as plt
        # plt.hist(lendist)
        # plt.show()
        # plt.hist(self.labels)
        # plt.show()
        ######################################
        
        for pos in pos_tokens:            
            seq = []
            posseq = []
            for w, t in pos[:max_len]:
                if w in self.vocab:
                    seq.append(self.vocab[w])
                else:
                    seq.append(self.vocab['[UNK]'])
                posseq.append(self.pos_vocab[t])
            
            curlen = len(pos)
            seq += [self.vocab['[PAD]']]*(max_len-curlen)
            posseq += [self.pos_vocab['CD']]*(max_len-curlen)
            
            self.sequence.append(seq)
            self.pos_seqs.append(posseq)            
                
        self.sequence = np.asarray(self.sequence, dtype=np.int64)
        self.pos_seqs = np.asarray(self.pos_seqs, dtype=np.int64)
        print("[info] vocab size =", len(self.vocab))
        
            
                
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        return torch.from_numpy(self.sequence[index]),\
                torch.from_numpy(self.pos_seqs[index]),\
                torch.from_numpy(self.labels[index])
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

# splits the given dataset into train/validation sets, according to the ratio given. Data are split according to labels as well, so that we have similar distribution between training and validation set.
def validation_split(data_set, split=0.1):
    sequence = data_set.sequence
    pos_seqs = data_set.pos_seqs
    lbls = data_set.labels
    sz = data_set.size
    space = data_set.lbspace.values()
        
    # find all labels
    indices = [[] for _ in space]
    for i, l in enumerate(lbls):
        indices[int(l)].append(i)
    
    # cut split
    cutoffs = [int(len(x)*split) for x in indices]
    
    # make validation 
    train_inds = []
    validation_inds = []
    for tp in space:
        cut = cutoffs[tp]
        train_inds += indices[tp][cut:]
        validation_inds += indices[tp][:cut]
    
    validation_set = deepcopy(data_set)
    validation_set.sequence = np.asarray([sequence[i] for i in validation_inds], dtype=np.int64)
    validation_set.pos_seqs = np.asarray([pos_seqs[i] for i in validation_inds], dtype=np.int64)
    validation_set.labels = np.asarray([lbls[i] for i in validation_inds], dtype=np.int64)
    validation_set.size = len(validation_inds)
    
    # trimdown
    data_set.sequence = np.asarray([sequence[i] for i in train_inds], dtype=np.int64)
    data_set.pos_seqs = np.asarray([pos_seqs[i] for i in train_inds], dtype=np.int64)
    data_set.labels = np.asarray([lbls[i] for i in train_inds], dtype=np.int64)
    data_set.size = len(train_inds)
    print("[info] {} train. {} valid.".format(data_set.size, validation_set.size))
    return data_set, validation_set

# function to create a generator for batches to be fed into LSTMs. Optionally calls validation_split to create train/validation split.
def make_data_generator(filename, task, batch_size, val_split=0, n_workers=4, testlabel=None, load_vocabs=None, max_len=512, vcutoff=99999, shuffle=True):
    data_set = Dataset(filename, task, n_workers, testlabel)
    data_set.prepare(load_vocabs, max_len, vcutoff)
    
    if val_split > 0:
        data_set, val_set = validation_split(data_set, split=val_split)        
    
    params = {'batch_size':batch_size,
         'shuffle': shuffle,
         'num_workers': n_workers}
    
    train_generator = data.DataLoader(data_set, **params)
    
    if val_split > 0:
        val_generator = data.DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=n_workers)
        return (data_set, train_generator), (val_set, val_generator)
    
    return data_set, train_generator