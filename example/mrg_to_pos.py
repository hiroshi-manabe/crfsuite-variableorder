import re, os

#mrg_dir = '/path/to/PennTreeBank3/parsed/mrg/wsj/'
mrg_dir = 'D:/cygwin/home/admin/PennTreeBank3/parsed/mrg/wsj/'

datasets = {
    'train.txt' : ['%02d' % x for x in range(0, 19)],
    'test.txt' : ['%02d' % x for x in range(22, 25)]
}

for dataset in datasets:
    is_first = True
    f_out = open(dataset, 'wb')
    for dir in datasets[dataset]:
        for file in os.listdir(dir):
            f_in = open(mrg_dir + dir + '/' + file)
            for line in f_in:
                if line[0] == '(':
                    if not is_first:
                        f_out.write("\n")
                    is_first = False
                for m in re.finditer(r'\(([^\(\)]+) ([^\(\)]+)\)', line):
                    str = m.group(1)
                    if str == '-NONE-': continue
                    if str == '-LRB-' : str = '('
                    if str == '-RRB-' : str = ')'
                    f_out.write('%s\t%s\n' % (str, m.group(2)))
            f_in.close
    f_out.write("\n")
    f_out.close





