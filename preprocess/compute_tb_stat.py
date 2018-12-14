# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2018-10-26 11:29:14
# @Last Modified by:   claravania
# @Last Modified time: 2018-12-10 18:06:43

import os
import sys
import codecs

from collections import defaultdict


def create_vocab(train_file):

	word_vocab = defaultdict(int)
	char_vocab = defaultdict(int)
	ngram_vocab = defaultdict(int)
	tok_freq = defaultdict(int)
	num_sents = 0

	with codecs.open(train_file, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line.startswith('#') or line == '':
				continue
			tok = line.split('\t')
			if '-' in tok[0] or '.' in tok[0]:
				continue

			wid = tok[0]
			if wid == '1':
				num_sents += 1

			wordform = tok[1]
			word_vocab[wordform] += 1
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
	return vocab, tok_freq, num_sents


def get_num_sents(infile):

	num_sents = 0
	with codecs.open(infile, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line.startswith('#') or line == '':
				continue
			tok = line.split('\t')
			if '-' in tok[0] or '.' in tok[0]:
				continue

			wid = tok[0]
			if wid == '1':
				num_sents += 1

	return num_sents


ud_dir = sys.argv[1]  # directory of UD
tb_list = [
	  'UD_North_Sami-Giella',
	  'UD_Finnish-TDT',
	  'UD_Portuguese-Bosque',
	  'UD_Turkish-IMST',
	  'UD_Galician-TreeGal',
	  'UD_Kazakh-KTB',
		]
for tb in tb_list:
	tb_dir = os.path.join(ud_dir, tb)
	train_sents, dev_sents, test_sents = 0, 0, 0
	for fname in os.listdir(tb_dir):
		if fname.endswith('ud-train.conllu'):
			vocab, tok_freq, train_sents = create_vocab(os.path.join(tb_dir, fname))
			print tb, tok_freq['word']
		
		elif fname.endswith('ud-dev.conllu'):
			dev_sents = get_num_sents(os.path.join(tb_dir, fname))

		elif fname.endswith('ud-test.conllu'):
			test_sents = get_num_sents(os.path.join(tb_dir, fname))			


	print tb, tok_freq['word'], len(vocab['word']), train_sents, dev_sents, test_sents