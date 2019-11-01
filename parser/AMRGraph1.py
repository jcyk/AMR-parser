# encoding=utf8
class AMRGraph(object):
    def __init__(self, smatch_amr):
        # transform amr from original smatch format into our own data structure
        instance_triple, attribute_triple, relation_triple = smatch_amr.get_triples()
        self.size = len(instance_triple) + len(attribute_triple) - 1
        self.root = smatch_amr.root
        self.nodes = set()
        self.undirected_edges = dict()
        self.name2concept = dict()
        self.instance_triple = []
        self.attribute_triple = []
        self.relation_triple = []
        for x in instance_triple:
            self.name2concept[x[1]] = x[2]
            self.nodes.add(x[1])
            self.instance_triple.append(x)
        for x in attribute_triple:
            if x[0] == 'TOP':
                self.attribute_triple.append(x)
                continue
            name = "%s_attr_%d"%(x[2], len(self.name2concept))
            self.name2concept[name] = x[2]
            self._add_edge(x[0], x[1], name)
            self.attribute_triple.append((x[0], x[1], name))
        for x in relation_triple:
            self._add_edge(x[0], x[1], x[2])
            self.relation_triple.append(x)

    def __len__(self):
        return len(self.name2concept)

    def _add_edge(self, rel, src, des):
        self.nodes.add(src)
        self.nodes.add(des)
        self.undirected_edges[src] = self.undirected_edges.get(src, []) + [(rel, des)]
        self.undirected_edges[des] = self.undirected_edges.get(des, []) + [(rel + '_reverse_', src)]

    def root_centered_sort(self):
        level = dict()
        level[self.root] = 0
        queue = [self.root]
        visited = set(queue)
        step = 0
        while len(queue) > step:
            src = queue[step]
            step += 1
            if src not in self.undirected_edges:
                continue

            for rel, des in self.undirected_edges[src]:
                if des in visited:
                    continue
                else:
                    level[des] = level[src] + 1
                    queue.append(des)
                    visited.add(des)
        not_connected = len(queue) != len(self.nodes)
        if not_connected:
            for x in self.nodes:
                if x not in queue:
                    level[x] = 999999
        return level
