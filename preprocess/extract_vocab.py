# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2018-10-28 15:33:07
# @Last Modified by:   claravania
# @Last Modified time: 2018-10-28 15:41:24



import os
import sys
import codecs
import operator

from collections import defaultdict


def create_vocab(train_file):

	word_vocab = defaultdict(int)
	char_vocab = defaultdict(int)
	ngram_vocab = defaultdict(int)
	tok_freq = defaultdict(int)
	with codecs.open(train_file, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line.startswith('#') or line == '':
				continue
			tok = line.split('\t')
			if '-' in tok[0] or '.' in tok[0]:
				continue

			wordform = tok[1].lower()
			upos = tok[3].lower()
			word_vocab[(wordform, upos)] += 1
			tok_freq['word'] += 1
			for ch in wordform:
				char_vocab[ch] += 1
				tok_freq['char'] += 1

			seq = '^' + wordform + '$'
			n = 3
			ngram_counts = len(seq) - n + 1
			for i in range(ngram_counts):
				ngram = seq[i:i + n]
				ngram_vocab[ngram] += 1
				tok_freq['ngram'] += 1

	vocab = {'word': word_vocab, 'char': char_vocab, 'ngram': ngram_vocab}
	return vocab, tok_freq


ud_dir = sys.argv[1]  # directory of UD

for tb in os.listdir(ud_dir):
	tb_dir = os.path.join(ud_dir, tb)
	for fname in os.listdir(tb_dir):
		if not fname.endswith('train.conllu'):
			continue

		vocab, tok_freq = create_vocab(os.path.join(tb_dir, fname))
		sorted_vocab = sorted(vocab['word'].items(), key=operator.itemgetter(0))
		fvocab = codecs.open(os.path.join(tb_dir, fname + '.vcb'), 'w', encoding='utf-8')
		for v, k in sorted_vocab:
			fvocab.write(str(v) + '\t' + str(k) + '\n')
