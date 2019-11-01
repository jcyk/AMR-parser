def get_alignments_from_file(fname):
    alignments = []
    with open(fname) as f:
        for line in f.readlines():
            if line.startswith('# ::lem'):
                toks = line.strip().split()[2:]
            if line.startswith('# ::node'):
                x = line.strip().split('\t')
                if len(x) != 4:
                    continue
                concept, pos = x[-2], x[-1]
                s, e  = pos.split('-')
                assert int(s) < int(e) <= len(toks)
                alignments.append((toks[int(s):int(e)], concept))
    return alignments

if __name__ == "__main__":

    res = get_alignments_from_file("2017/train.patch.txt_processed_preprocess.stanford.no_wiki.tamr.txt")
    with open('out_stanford', 'w') as fo:
        for x, y in res:
            line = ' '.join(x) + '\t' + y + '\n'
            fo.write(line)