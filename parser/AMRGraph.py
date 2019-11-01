# encoding=utf8
import re
import random

number_regexp = re.compile(r'^-?(\d)+(\.\d+)?$')

attr_value_set = set(['-', '+', 'interrogative', 'imperative', 'expressive'])

def get_remove_sense_map():
    sense_suffix = re.compile(r'.+\-[0-9]+$')
    conc = [line.split()[0] for line in open('../preprocessing/common/propbank-frame-arg-descr.txt').readlines()]
    for line in open('../preprocessing/common/verbalization-list-v1.06.txt').readlines():
        if line.startswith('MAYBE-VERBALIZE') or line.startswith('VERBALIZE'):
            conc.append(line.strip().split()[-1])
    ret = dict()
    for c in conc:
        if sense_suffix.match(c) is not None:
            ret[c] = c[:c.rfind('-')]
    return ret

def is_attr_form(x):
    return (x in attr_value_set or x.endswith('_') or number_regexp.match(x) is not None)

class AMRGraph(object):

    remove_sense_map = get_remove_sense_map()

    def normalize(conc):
        # lowercase and remove sense
        conc = conc.lower()
        return AMRGraph.remove_sense_map.get(conc, conc)

    def __init__(self, smatch_amr):
        # transform amr from original smatch format into our own data structure
        instance_triple, attribute_triple, relation_triple = smatch_amr.get_triples()
        self.root = smatch_amr.root
        self.nodes = set()
        self.edges = dict()
        self.reversed_edges = dict()
        self.undirected_edges = dict()
        self.name2concept = dict()
        self.name2concept_sensed = dict()
        self.wiki = []
        for x in instance_triple:
            if is_attr_form(x[2]):
                print ('bad concept', x)
            self.name2concept_sensed[x[1]] = x[2].lower()
            self.name2concept[x[1]] = AMRGraph.normalize(x[2])
            self.nodes.add(x[1])
        for x in attribute_triple:
            if x[0] == 'TOP':
                continue
            if x[0]=='wiki':
                self.wiki.append((x[1], x[2].rstrip('_')))
                continue
            if not is_attr_form(x[2]):
                print ('bad attribute', x)
            name = "%s_attr_%d"%(x[2], len(self.name2concept))
            self.name2concept[name] = x[2].lower()
            self._add_edge(x[0], x[1], name)
        for x in relation_triple:
            self._add_edge(x[0], x[1], x[2])

    def __len__(self):
        return len(self.name2concept)

    def _add_edge(self, rel, src, des):
        self.nodes.add(src)
        self.nodes.add(des)
        self.edges[src] = self.edges.get(src, []) + [(rel, des)]
        self.reversed_edges[des] = self.reversed_edges.get(des, []) + [(rel, src)]
        self.undirected_edges[src] = self.undirected_edges.get(src, []) + [(rel, des)]
        self.undirected_edges[des] = self.undirected_edges.get(des, []) + [(rel + '_reverse_', src)]

    def root_centered_sort(self, rel_order=None):
        queue = [self.root]
        visited = set(queue)
        step = 0
        while len(queue) > step:
            src = queue[step]
            step += 1
            if src not in self.undirected_edges:
                continue

            random.shuffle(self.undirected_edges[src])
            if rel_order is not None:
                # Do some random thing here for performance enhancement
                if random.random() < 0.5:
                    self.undirected_edges[src].sort(key=lambda x: -rel_order(x[0]) if (x[0].startswith('snt') or x[0].startswith('op') ) else -1)
                else:
                    self.undirected_edges[src].sort(key=lambda x: -rel_order(x[0]))
            for rel, des in self.undirected_edges[src]:
                if des in visited:
                    continue
                else:
                    queue.append(des)
                    visited.add(des)
        not_connected = len(queue) != len(self.nodes)
        assert (not not_connected)
        name2pos = dict(zip(queue, range(len(queue))))

        visited = set()
        edge = []
        for x in queue:
            if x not in self.undirected_edges:
                continue
            for r, y in self.undirected_edges[x]:
                if y in visited:
                    r = r[:-9] if r.endswith('_reverse_') else r+'_reverse_'
                    edge.append((name2pos[x], name2pos[y], r)) # x -> y: r
            visited.add(x)
        return [self.name2concept[x] for x in queue], edge, not_connected
