import random
import torch
from torch import nn
import numpy as np
from AMRGraph import AMRGraph
from extract import read_file

PAD, UNK, DUM, NIL, END, CLS = '<PAD>', '<UNK>', '<DUMMY>', '<NULL>', '<END>', '<CLS>'

def load_pretrained_word_embed(fname):
    vs= dict()
    start_line = True
    with open('glove_vocab', 'w') as fo:
        for line in open(fname).readlines():
            if start_line:
                start_line = False
                continue
            d = line.strip().split()
            w, v = d[0], torch.from_numpy(np.array(d[1:], dtype=np.float32))
            vs[w] = v
            fo.write(w+'\t1\n')
    vocab = Vocab('glove_vocab', 0)
    weights = v.data.new(vocab.size, v.size(0)).zero_()
    for w in vocab._idx2token:
        if w in vs:
            weights[vocab.token2idx(w)] = vs[w]
    return vocab, nn.Embedding.from_pretrained(weights, freeze=True)

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._priority = dict()
        for line in open(filename).readlines():
            token, cnt = line.strip().split()
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
            self._priority[token] = int(cnt)
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]
    
    def priority(self, x):
        return self._priority.get(x, 0)
    
    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    pad = vocab.padding_idx if vocab else 0
    
    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        ys.append(y)
    data = np.transpose(np.array(ys))
    return data

def ListsofStringToTensor(xs, vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD]*(max_len -len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([CLS]+z+[END]) + [vocab.padding_idx]*(max_string_len - len(z)))
        ys.append(zs)

    data = np.transpose(np.array(ys), (1, 0, 2))
    return data

def batchify(data, vocabs, unk_rate=0.):
    _tok = ListsToTensor([ [CLS]+x['tok'] for x in data], vocabs.get('glove', vocabs['tok']), unk_rate=unk_rate)
    _lem = ListsToTensor([ [CLS]+x['lem'] for x in data], vocabs['lem'], unk_rate=unk_rate)
    _pos = ListsToTensor([ [CLS]+x['pos'] for x in data], vocabs['pos'], unk_rate=unk_rate)
    _ner = ListsToTensor([ [CLS]+x['ner'] for x in data], vocabs['ner'], unk_rate=unk_rate)
    _word_char = ListsofStringToTensor([ [CLS]+x['tok'] for x in data], vocabs['word_char'])

    local_token2idx = [x['token2idx'] for x in data]
    local_idx2token = [x['idx2token'] for x in data]
    _cp_seq = ListsToTensor([ x['cp_seq'] for x in data], vocabs['predictable_concept'], local_token2idx)
    _mp_seq = ListsToTensor([ x['mp_seq'] for x in data], vocabs['predictable_concept'], local_token2idx)

    concept, edge = [], []
    for x in data:
        amr = x['amr']
        concept_i, edge_i, _ = amr.root_centered_sort(vocabs['rel'].priority)
        concept.append(concept_i)
        edge.append(edge_i)

    augmented_concept = [[DUM]+x+[END] for x in concept]

    _concept_in = ListsToTensor(augmented_concept, vocabs['concept'], unk_rate=unk_rate)[:-1]
    _concept_char_in = ListsofStringToTensor(augmented_concept, vocabs['concept_char'])[:-1]
    _concept_out = ListsToTensor(augmented_concept, vocabs['predictable_concept'], local_token2idx)[1:]

    out_conc_len, bsz = _concept_out.shape
    _rel = np.full((out_conc_len, bsz, out_conc_len), vocabs['rel'].token2idx(PAD))

    for bidx, (x, y) in enumerate(zip(edge, concept)):
        for l in range(1, len(y)):
            _rel[l, bidx, 1:l+1] = vocabs['rel'].token2idx(NIL)
        for v, u, r in x:
            r = vocabs['rel'].token2idx(r)
            _rel[v, bidx, u+1] = r   # v: [concept_1, ..., concept_n, <end>] u: [<dummy>, concept_1, ..., concept_n}]
 
    ret = {'lem':_lem, 'tok':_tok, 'pos':_pos, 'ner':_ner, 'word_char':_word_char, \
           'copy_seq': np.stack([_cp_seq, _mp_seq], -1), \
           'local_token2idx':local_token2idx, 'local_idx2token': local_idx2token, \
           'concept_in':_concept_in, 'concept_char_in':_concept_char_in, \
           'concept_out':_concept_out, 'rel':_rel}
    return ret
    
class DataLoader(object):
    def __init__(self, vocabs, lex_map, filename, batch_size, for_train):
        self.data = []
        for amr, token, lemma, pos, ner in zip(*read_file(filename)):
            if for_train:
                _, _, not_ok = amr.root_centered_sort()
                if not_ok:
                    continue
                if ' '.join(token) == "https://www.com.html https://www.com.html </a>":
                    continue
            cp_seq, mp_seq, token2idx, idx2token = lex_map.get_concepts(lemma, token, vocabs['predictable_concept']) 
            datum = {'amr':amr, 'tok':token, 'lem':lemma, 'pos':pos, 'ner':ner, \
                     'cp_seq':cp_seq, 'mp_seq':mp_seq,\
                     'token2idx':token2idx, 'idx2token':idx2token}
            self.data.append(datum)
        print ("Get %d AMRs from %s"%(len(self.data), filename))
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train
        self.unk_rate = 0.

    def set_unk_rate(self, x):
        self.unk_rate = x

    def __iter__(self):
        idx = list(range(len(self.data)))
        
        if self.train:
            random.shuffle(idx)
            idx.sort(key = lambda x: len(self.data[x]['tok']) + len(self.data[x]['amr']))

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(self.data[i]['tok']) + len(self.data[i]['amr'])
            data.append(self.data[i])
            if num_tokens >= self.batch_size:
                batches.append(data)
                num_tokens, data = 0, []
        
        if not self.train or num_tokens > self.batch_size/2:
            batches.append(data)
        
        if self.train:
            random.shuffle(batches)
        
        for batch in batches:
            yield batchify(batch, self.vocabs, self.unk_rate)
