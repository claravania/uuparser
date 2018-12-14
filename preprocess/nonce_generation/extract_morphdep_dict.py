# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2018-11-20 19:21:19
# @Last Modified by:   claravania
# @Last Modified time: 2018-11-26 16:29:12


import os
import sys
import codecs
import random

from collections import defaultdict


def create_morph_vocab(train_file, pos):
	"""
	Create a dictionary of words according to its morphology&dep labels
	key: (xfeats, dep_labels)
	value: list of words
	"""
	vocab = defaultdict(int)
	num_sents = 0
	with codecs.open(train_file, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip().lower()
			if line.startswith('#') or line == '':
				continue
			tok = line.split('\t')
			if '-' in tok[0] or '.' in tok[0]:
				continue

			if tok[0] == '1':
				num_sents += 1

			assert len(tok) == 10

			wordform = tok[1]
			lemma = tok[2]
			upos = tok[3]
			xfeats = tok[5]
			dep_labels = tok[7].split(':')[0]
			
			if upos in pos:
				key = (wordform, lemma, upos, xfeats, dep_labels)
				vocab[key] += 1
		
	return vocab, num_sents


tb_dir = sys.argv[1]
out_dir = sys.argv[2]
pos = ['noun', 'verb', 'adj']
for fname in os.listdir(tb_dir):
	if fname.endswith('train.conllu'):
		train_vocab, num_sents = create_morph_vocab(os.path.join(tb_dir, fname), pos)
		fout = codecs.open(os.path.join(out_dir, fname + '.morphdict'), 'w', encoding='utf-8')

		for key in sorted(train_vocab.keys()):
			out_line = list(key) + [str(train_vocab[key])]
			fout.write('\t'.join(out_line) + '\n')

fout.close()
		

			