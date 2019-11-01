# encoding=utf8
class AMRGraph(object):
    def __init__(self, smatch_amr):
        # transform amr from original smatch format into our own data structure
        instance_triple, attribute_triple, relation_triple, _ = smatch_amr.get_triples()
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


    def trim(self, levels, weighted, prefix):

        def score_by_level(l):
            if weighted:
                return max(5-l, 1)
            else:
                return 1
        if levels == -1:
            levels = float('inf')
        level = self.root_centered_sort()
        instance_triple, attribute_triple, relation_triple = [], [], []
        for x in self.instance_triple:
            if level[x[1]] <= levels:
                w = score_by_level(level[x[1]])
                instance_triple.append((x[0], x[1], x[2], w))
        for x in self.attribute_triple:
            if x[0] == 'TOP':
                w = score_by_level(level[x[1]])
                attribute_triple.append((x[0], x[1], x[2], w))
                continue
            if level[x[1]] <= levels and level[x[2]] <= levels:
                w = min(score_by_level(level[x[1]]), score_by_level(level[x[2]]))
                attribute_triple.append((x[0], x[1], self.name2concept[x[2]], w))
        for x in self.relation_triple:
            if level[x[1]] <= levels and level[x[2]] <= levels:
                w = min(score_by_level(level[x[1]]), score_by_level(level[x[2]]))
                relation_triple.append((x[0], x[1], x[2], w))

        node_map_dict = {}
        for i in range(0, len(instance_triple)):
            node_map_dict[instance_triple[i][1]] = prefix + str(i)
        tot_w = 0
        for i, v in enumerate(instance_triple):
            instance_triple[i] = (v[0], node_map_dict[v[1]], v[2], v[-1])
            tot_w += v[-1]
        for i, v in enumerate(attribute_triple):
            attribute_triple[i] = (v[0], node_map_dict[v[1]], v[2], v[-1])
            tot_w += v[-1]
        for i, v in enumerate(relation_triple):
            relation_triple[i] = (v[0], node_map_dict[v[1]], node_map_dict[v[2]], v[-1])
            tot_w += v[-1]
        return instance_triple, attribute_triple, relation_triple, tot_w

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