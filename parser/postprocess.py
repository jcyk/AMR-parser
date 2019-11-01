import penman
from AMRGraph import is_attr_form
import re
import networkx as nx
import numpy as np

class PostProcessor(object):
    def __init__(self, sense_table, wiki_table, rel_vocab):
        self.amr = penman.AMRCodec()
        self.sense_table = dict()
        for line in open(sense_table).readlines():
            x, y = line.strip().split('\t')
            self.sense_table[x] = y
        
        self.wiki_table = dict()
        for line in open(wiki_table).readlines():
            x, y = line.strip().split('\t')
            self.wiki_table[x] = y
        self.rel_vocab = rel_vocab

    def to_triple_MSCG(self, res_concept, res_relation):
        """mininum spanning connected subgraph algorithm"""
        names = []
        graph = nx.Graph()
        ret = []
        for i, c in enumerate(res_concept):
            if not is_attr_form(c):
                name = c[0] + str(i)
                ret.append((name, 'instance', c))
            else:
                if c.endswith('_'):
                    name = '"'+c[:-1]+'"'
                else:
                    name = c
                name = name + '@attr%d@'%i
            names.append(name)
            graph.add_node(i, name=name)

        edges = []
        for i, j, p, r in res_relation:
            for r_i, p_i in enumerate(r):
                r_i = self.rel_vocab.idx2token(r_i)
                gain = np.log(p * p_i / (1. -p + 1e-9))
                if r_i.endswith('_reverse_'):
                    edges.append((i, r_i[:-9], j, gain))
                else:
                    edges.append((j, r_i, i, gain))

        edges.sort(key=lambda x: -x[-1])
        grouped_by_src = set()
        grouped_by_src_rel = set()
        grouped_by_src_tgt = set()
        grouped_by_rel_tgt = set()
        step = 0
        for e in edges:
            step += 1
            i, rel, j, gain = e
            if gain < 0 and nx.is_connected(graph):
                break
            if gain < 0 and nx.has_path(graph, i, j):
                continue
            # simple
            if (i, j) in grouped_by_src_tgt:
                continue
            # at most one tgt for one rel
            if (i, rel) in grouped_by_src_rel:
                continue
            if (rel, j) in grouped_by_rel_tgt:
                continue
            if is_attr_form(res_concept[i]) or is_attr_form(res_concept[j]):
                # attr has at most one neighbor
                if is_attr_form(res_concept[i]) and is_attr_form(res_concept[j]):
                    continue
                else:
                    if is_attr_form(res_concept[i]) and (i in grouped_by_src):
                        continue
                    if is_attr_form(res_concept[j]) and (j in grouped_by_src):
                        continue
            grouped_by_src.add(i)
            grouped_by_src.add(j)
            grouped_by_src_rel.add((i, rel))
            grouped_by_rel_tgt.add((rel, j))
            grouped_by_src_tgt.add((i, j))
            grouped_by_src_tgt.add((j, i))
            ret.append((names[i], rel, names[j]))
            graph.add_edge(i, j)
        assert nx.is_connected(graph)

        return ret

    def to_triple(self, res_concept, res_relation):
        """ res_concept: list of strings
            res_relation: list of (dep:int, head:int, arc_prob:float, rel_prob:list(vocab))
        """
        ret = []
        names = []
        for i, c in enumerate(res_concept):
            if not is_attr_form(c):
                name = c[0] + str(i)
                ret.append((name, 'instance', c))
            else:
                if c.endswith('_'):
                    name = '"'+c[:-1]+'"'
                else:
                    name = c
                name = name + '@attr%d@'%i
            names.append(name)

        grouped_relation = dict()
        for i, j, p, r in res_relation:
            r = self.rel_vocab.idx2token(np.argmax(np.array(r)))
            grouped_relation[i] = grouped_relation.get(i, []) + [(j, p, r)]
        for i, c in enumerate(res_concept):
            if i ==0:
                continue
            max_p, max_j, max_r = 0., 0, None
            for j, p, r in grouped_relation[i]:
                assert j < i
                if is_attr_form(res_concept[j]):
                    continue
                if p >=0.5:
                    if not is_attr_form(res_concept[i]):
                        if r.endswith('_reverse_'):
                            ret.append((names[i], r[:-9], names[j]))
                        else:
                            ret.append((names[j], r, names[i]))
                if p > max_p:
                    max_p = p
                    max_j = j
                    max_r = r

            if max_p < 0.5 or is_attr_form(res_concept[i]):
                if max_r.endswith('_reverse_'):
                    ret.append((names[i], max_r[:-9], names[max_j]))
                else:
                    ret.append((names[max_j], max_r, names[i]))
        return ret

    def patch_wsd(self, x):
        ret = []
        for src, rel, des in x:
            if rel == 'instance':
                des = self.sense_table.get(des, des)
            ret.append((src, rel, des))
        return ret

    def patch_wiki(self, x):
        wiki = dict()
        hold = dict()
        for src, rel, des in x:
            if rel == 'name':
                wiki[des] = src
                hold[des] = []
        for src, rel, des in x:
            if src in wiki and rel.startswith('op'):
                part = re.sub(r'@attr\d+@', '', des)
                if len(part)>2 and part[0] == '"' and part[-1] == '"':
                    part = part[1:-1]
                hold[src].append((int(rel[2:]), part))
        i = len(x)
        wiki_triple = []
        for wiki_name in wiki:
            src = wiki[wiki_name]
            full_name = '_'.join(([p[1] for p in sorted(hold[wiki_name])]))
            if full_name in self.wiki_table:
                des = '"'+self.wiki_table[full_name]+'"'
            else:
                des = '-'
            des = des + '@attr%d@'%i
            wiki_triple.append((src, 'wiki', des))
            i +=1
        return x + wiki_triple

    def get_string(self, x):
        return self.amr.encode(penman.Graph(x), top=x[0][0])
    
    def postprocess(self, concept, relation):
        mstr = self.get_string(self.patch_wiki(self.patch_wsd(self.to_triple(concept, relation))))
        return re.sub(r'@attr\d+@', '', mstr)
