#!/usr/bin/env python
# coding: utf-8
from smatch import AMR
from AMRGraph import AMRGraph, number_regexp
from collections import Counter

def read_file(filename):
    # read preprocessed amr file (basically follows the original LDC format)
    f = open(filename)
    token, lemma, pos, ner = [], [], [], []
    amrs = []
    while True:
        for x in f:
            if x.startswith('# ::tok'):
                break
        else:
            break
        token.append(x.strip().split()[2:])
        for x in f:
            if x.startswith('# ::lem'):
                break
        lemma.append(x.strip().split()[2:])
        for x in f:
            if x.startswith('# ::pos'):
                break
        pos.append(x.strip().split()[2:])
        for x in f:
            if x.startswith('# ::ner'):
                break
        ner.append(x.strip().split()[2:])
        x = AMR.get_amr_line(f)
        if not x:
            break
        amr = AMR.parse_AMR_line(x)
        myamr = AMRGraph(amr)
        amrs.append(myamr)
    f.close()
    assert len(token) == len(lemma) == len(pos) == len(ner) == len(amrs), (len(token),len(lemma),len(pos),len(ner),len(amrs))
    print ('read from %s, %d amrs'%(filename, len(token)))
    return amrs, token, lemma, pos, ner

class LexicalMap(object):
    # build our lexical mapping (from token/lemma to concept), useful for copy mechanism.
    def __init__(self, fname):
        mappings= dict()

        def update(x):
            _tok = x[0]
            x = x[1:]
            if ' ' in _tok:
                return
            for _conc in x[::-1]:
                if _conc == 'name':
                     continue
                if len(_conc)>2 and _conc[0] ==  _conc[-1] == '"':
                    continue
                if (number_regexp.match(_tok) is not None):
                    continue
                if _tok not in mappings:
                    mappings[_tok] = Counter()
                mappings[_tok][_conc] += 1
                return

        last_token = None
        cur = []
        for line in open(fname).readlines():
            token, concept = line.strip().split('\t')

            concept = AMRGraph.normalize(concept)

            if token != last_token:
                if cur:
                    update(cur)
                cur = [token, concept]
            else:
                cur.append(concept)
            last_token = token
        if cur:
            update(cur)
        ret = dict()
        for token in mappings:
            cnt = mappings[token]
            conc, num = cnt.most_common()[0]
            if num >= 2:
                ret[token] = conc

        self.mappings = ret
        self.copyings = dict()

    #TODO!!!!!!!!!!!!!
    #we will copy lu's map and rules
    #cp_seq, mp_seq, token2idx, idx2token = lex_map.get(lemma, token, vocabs['predictable_concept'])
    def get_concepts(self, lem, tok, vocab=None):
        cp_seq, mp_seq = [], []
        new_tokens = set()
        for le, to in zip(lem, tok):
            cp = self.copyings.get(le, le+'_')
            mp = self.mappings.get(le, le)
            cp_seq.append(cp)
            mp_seq.append(mp)

        if vocab is None:
            return cp_seq, mp_seq

        for cp, mp in zip(cp_seq, mp_seq):
            if vocab.token2idx(cp) == vocab.unk_idx:
                new_tokens.add(cp)
            if vocab.token2idx(mp) == vocab.unk_idx:
                new_tokens.add(mp)
        nxt = vocab.size
        token2idx, idx2token = dict(), dict()
        for x in new_tokens:
            token2idx[x] = nxt
            idx2token[nxt] = x
            nxt += 1
        return cp_seq, mp_seq, token2idx, idx2token 

def collect_sense(amr, sense_table):
    for k, v in amr.name2concept_sensed.items():
        if v in AMRGraph.remove_sense_map:
            no_sense_v = AMRGraph.remove_sense_map[v]
        else:
            no_sense_v = v

        if no_sense_v not in sense_table:
            sense_table[no_sense_v] = Counter()
        sense_table[no_sense_v][v] += 1

def collect_wiki(amr, wiki_table):
    for src, wiki_des in amr.wiki:
        if wiki_des == '-':
            continue
        if src not in amr.edges:
            print ('bad wiki: %s has no outgoing edge'%src)
            continue
        name_des = None
        for rel, des in amr.edges[src]:
            if rel == 'name':
                if name_des is not None:
                    print ('bad wiki: multiple names for wiki')
                name_des = des
        if name_des is None:
            print ('bad wiki: %s has no :name edge'%wiki_des)
            continue
        full_name = []
        for rel, des in amr.edges[name_des]:
            if not rel.startswith('op'):
                print ('bad wiki: non-opX relation found', wiki_des, rel, des)
                continue
            full_name.append((int(rel[2:]), amr.name2concept[des].rstrip('_')))
        full_name = '_'.join(([x[1] for x in sorted(full_name)]))
        if full_name not in wiki_table:
            wiki_table[full_name] = Counter()
        wiki_table[full_name][wiki_des] += 1

def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n'%(x,y))

if __name__ == "__main__":
    amrs, token, lemma, pos, ner = read_file('../preprocessing/2017/train.txt_processed_preprocess')
    lexical_map = LexicalMap('../preprocessing/common/out_stanford')


    # collect wiki and sense from the training data
    sense_table = dict()
    wiki_table = dict()
    for amr in amrs:
        collect_sense(amr, sense_table)
        collect_wiki(amr, wiki_table)

    for no_sense_v in sense_table:
        v, _ = sense_table[no_sense_v].most_common()[0]
        sense_table[no_sense_v] = v
    
    for v in amr.remove_sense_map:
        no_sense_v = amr.remove_sense_map[v]
        if no_sense_v not in sense_table:
            sense_table[no_sense_v] = no_sense_v+'-01'
    
    with open('sense_table', 'w') as fo:
        for no_sense_v in sense_table:
            fo.write(no_sense_v+'\t'+sense_table[no_sense_v]+'\n')

    for x in wiki_table:
        wiki, _  = wiki_table[x].most_common()[0]
        wiki_table[x] = wiki
    
    with open('wiki_table', 'w') as fo:
        for x in wiki_table:
            fo.write(x+'\t'+wiki_table[x]+'\n')

    # collect concepts and relations
    conc = []
    rel = []
    predictable_conc = []
    for i in range(10):
        # run 10 times random sort to get the priorities of different types of edges
        for amr, lem, tok in zip(amrs, lemma, token):
            concept, edge, not_ok = amr.root_centered_sort()
            lexical_concepts = set()
            cp_seq, mp_seq = lexical_map.get_concepts(lem, tok)
            for lc, lm in zip(cp_seq, mp_seq):
                lexical_concepts.add(lc)
                lexical_concepts.add(lm)
            
            if i == 0:
                predictable_conc.append([ c for c in concept if c not in lexical_concepts])
                conc.append(concept)
            rel.append([e[-1] for e in edge])

    # make vocabularies
    token_vocab, token_char_vocab = make_vocab(token, char_level=True)
    lemma_vocab, lemma_char_vocab = make_vocab(lemma, char_level=True)
    pos_vocab = make_vocab(pos)
    ner_vocab = make_vocab(ner)
    conc_vocab, conc_char_vocab = make_vocab(conc, char_level=True)

    predictable_conc_vocab = make_vocab(predictable_conc)
    rel_vocab = make_vocab(rel)

    print ('make vocabularies')
    write_vocab(token_vocab, 'tok_vocab')
    write_vocab(token_char_vocab, 'word_char_vocab')
    write_vocab(lemma_vocab, 'lem_vocab')
    write_vocab(lemma_char_vocab, 'lem_char_vocab')
    write_vocab(pos_vocab, 'pos_vocab')
    write_vocab(ner_vocab, 'ner_vocab')
    write_vocab(conc_vocab, 'concept_vocab')
    write_vocab(conc_char_vocab, 'concept_char_vocab')
    write_vocab(predictable_conc_vocab, 'predictable_concept_vocab')
    write_vocab(rel_vocab, 'rel_vocab')
