# MUST BE WITH PYTHON2
# encoding=utf8
# 1. number connecting
# 2. urls
import os, sys, re

super_url = "https://www.com.html"

def further_preprocess(fname, train):
    num = re.compile(r'^([\d\.]+),((million)|(billion)|(trillion)|([\d\.]+))$')
    with open(fname+'_preprocess', 'w') as fo:
        for line in open(fname).readlines():
            if "\xc2\xa0" in line:
                if 'href' not in line:
                    line = line.replace('\xc2\xa0', ',') #9\xc2\xa01/2 => 9,1/2
                else:
                    line = line.replace("\xc2\xa0", '-')
            if line.startswith('# ::snt '):
                fo.write(line)
                continue
            if train or line.startswith('# ::tok ') or line.startswith('# ::lem '):
                new_xs = []
                for x in line.strip().split():
                    if ('http:' in x) or ('www.' in x) or ('https:' in x):
                        if line.startswith('# ::tok ') or line.startswith('# ::lem '):
                            new_xs.append(super_url)
                        else:
                            assert x[0] == '"', x
                            e = x.rfind('"')
                            assert e>0, x
                            new_xs.append('"'+super_url+'"'+x[e+1:])
                    else:
                        if ( line.startswith('# ::tok ') or line.startswith('# ::lem ')) and (num.match(x) is not None):
                            c_num = x.split(',')
                            if c_num[-1] == 'million':
                                x = str(int(float(''.join(c_num[:-1]))*1000000))
                            elif c_num[-1] == 'billion':
                                x = str(int(float(''.join(c_num[:-1]))*1000000000))
                            elif c_num[-1] == 'trillion':
                                x = str(int(float(''.join(c_num[:-1]))*1000000000000))
                            else:
                                x = ''.join(c_num)
                        new_xs.append(x)

                line = ' '.join(new_xs) + '\n'
            fo.write(line)

if __name__ == "__main__":
    dir_name = sys.argv[1]
    for fname in os.listdir(dir_name):
        if not fname.endswith('_processed'):
            continue
        further_preprocess(os.path.join(dir_name, fname), ('train' in fname))