import sys
import csv
import numpy as np
import wordninja
from nltk import pos_tag
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool
import torch
from torch.utils import data

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
        
        #self.ids = np.asarray(self.ids, dtype=np.int64)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.size = len(self.tweets)
        
        print("[info] {} data.".format(self.size))
            
            
    def _subtask(self, s):
        s = s.lower()
        tokens = wordninja.split(s)
        pos = pos_tag(tokens)
        return pos
    
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
        
def make_data_generator(filename, task, batch_size, n_workers=4, testlabel=None, load_vocabs=None, max_len=512, vcutoff=99999, shuffle=True):    
    data_set = Dataset(filename, task, n_workers, testlabel)
    data_set.prepare(load_vocabs, max_len, vcutoff)
    
    params = {'batch_size':batch_size,
         'shuffle': shuffle,
         'num_workers': n_workers}
    generator = data.DataLoader(data_set, **params)
    return data_set, generator