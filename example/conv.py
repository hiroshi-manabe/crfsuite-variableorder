import sys, re

order_list = [[0, 1], [0, 1], [0, 1, 2], [0, 2], [0], [0], [0]]

def output_data(f_out, seq, max_order_list, feature_map, generate_features):
    filler_length = 10
    length = len(seq)
    seq = [['', '']] * filler_length + seq
    seq.extend([['', ''], ['', '']] * filler_length)
    label_seq = ['__BOS_EOS__'];

    for i in range(filler_length, length + filler_length):
        fs = []
        fs.append('LABEL')
        fs.append('W0_%s' % seq[i][1])
        fs.append('W-1_%s' % seq[i-1][1])
        fs.append('W+1_%s' % seq[i+1][1])
        fs.append('W-10_%s_%s' % (seq[i-1][1], seq[i][1]))
        fs.append('W0+1_%s_%s' % (seq[i][1], seq[i+1][1]))
        fs.append('W-2-1_%s_%s' % (seq[i-2][1], seq[i-1][1]))
        fs.append('W-2-10_%s_%s_%s' % (seq[i-2][1], seq[i-1][1], seq[i][1]))
        fs.append('W-3-2-1_%s_%s_%s' % (seq[i-3][1], seq[i-2][1], seq[i-1][1]))
        if i < length + filler_length:
            for j in range(1, 11):
                if len(seq[i][1]) >= j:
                    fs.append('suf%d_%s' % (j, seq[i][1][-j:]))
                if len(seq[i][1]) >= j:
                    fs.append('pre%d_%s' % (j, seq[i][1][:j]))
            if re.search(r'\d', seq[i][1]):
                fs.append('CONTAIN_NUMBER')
            if re.search(r'[A-Z]', seq[i][1]):
                fs.append('CONTAIN_UPPER')
            if re.search(r'\-', seq[i][1]):
                fs.append('CONTAIN_HYPHEN')
        if seq[i][0] == '__BOS_EOS__':
            fs = fs[:9]
        f_out.write(encode('%s\t%s\n' % (seq[i][0], '\t'.join(fs))))

        label_seq.append(seq[i][0])
        if generate_features:
            for attr_num in range(len(fs)):
                if fs[attr_num] not in feature_map:
                    feature_map[fs[attr_num]] = set()
                for j in order_list[attr_num] if attr_num < len(order_list) else range(1):
                    if (i - filler_length) - j >= -1:
                        feature_map[fs[attr_num]].add(tuple(label_seq[:-j-2:-1]))

    f_out.write('\n')

def encode(x):
    x = x.replace('\\', '\\\\')
    x = x.replace(':', '\\:')
    x = x.replace('#', '\\#')
    return x

filename_dics = {
    'train' : {
        'in' : 'train.txt',
        'out' : 'train_data.txt',
        'generate_features' : True
        },
    'test' : {
        'in' : 'test.txt',
        'out' : 'test_data.txt',
        'generate_features' : False
        },
}

filename_features = 'features.txt'
feature_map = {}

for (mode, filename_dic) in filename_dics.iteritems():
    if 'in' not in filename_dic or 'out' not in filename_dic:
        raise 'You must provide filenames for both input and output.'
    print filename_dic['in'] + '\n'
    f_in = open(filename_dic['in'], 'r')
    f_out = open(filename_dic['out'], 'wb')
    seq = []
    sentence_num = 0
    label_set = set()

    for line in f_in:
        line = line.strip('\n')

        if len(line) == 0:
            seq.append(['__BOS_EOS__', '__BOS_EOS__'])
            output_data(f_out, seq, order_list, feature_map, filename_dic['generate_features'])
            seq = []
        else:
            label_and_word = line.split('\t')
            seq.append(label_and_word)
            label_set.add(label_and_word[0])

    f_in.close()
    f_out.close()

for label in label_set:
    feature_map['LABEL'].add(('__BOS_EOS__', label))
    feature_map['LABEL'].add((label, '__BOS_EOS__'))

f_feature_out = open(filename_features, 'w')
for attr in feature_map:
    for label_seq in feature_map[attr]:
        f_feature_out.write(encode('%s\t%s\n' % (attr, '\t'.join(label_seq))))
f_feature_out.close()


