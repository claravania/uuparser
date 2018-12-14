# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2018-10-15 15:39:25
# @Last Modified by:   claravania
# @Last Modified time: 2018-12-14 11:28:25


import os
import sys
import codecs


tb_dir = sys.argv[1]
max_sent = int(sys.argv[2])

filenames = os.listdir(tb_dir)
trainfile = ''
num_sents = 0
for f in filenames:
	if f.endswith('ud-dev.conllu'):
		trainfile = os.path.join(tb_dir, f)
		print 'Reading ', trainfile
		sents = []
		sent = []
		num_sents = 0
		with codecs.open(trainfile, 'r', encoding='utf-8') as fin:
			for line in fin:
				line = line.strip()
				if line == '' and len(sent) > 0:
					sents.append(sent)
					num_sents += 1
					sent = []

				else:
					sent.append(line)

fout1 = codecs.open(trainfile + '.new-dev', 'w', encoding='utf-8')
fout2 = codecs.open(trainfile + '.new-rest', 'w', encoding='utf-8')
num_train_sents = 0
num_dev_sents = 0
print 'Treebank sents', num_sents
for i in range(0, num_sents):
	if num_train_sents < max_sent:
		for x in sents[i]:
			fout1.write(unicode(x) + '\n')
		fout1.write('\n')
		num_train_sents += 1
	else:
		fout1.close()
		for x in sents[i]:
			fout2.write(unicode(x) + '\n')
		fout2.write('\n')
		num_dev_sents += 1

fout2.close()

print 'Train: ', num_train_sents
print 'Dev: ', num_dev_sents
print 'Total sents', num_train_sents + num_dev_sents






