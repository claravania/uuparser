# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2018-10-18 16:55:28
# @Last Modified by:   claravania
# @Last Modified time: 2018-12-14 11:28:35

from __future__ import division
import os
import json
import codecs

from collections import defaultdict



def isDigit(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def create_vocab(train_file):

	word_vocab = defaultdict(int)
	char_vocab = defaultdict(int)
	ngram_vocab = defaultdict(int)
	with codecs.open(train_file, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if line.startswith('#') or line == '':
				continue
			tok = line.split('\t')
			if '-' in tok[0] or '.' in tok[0]:
				continue

			wordform = tok[1]
			if not isDigit(wordform):
				word_vocab[wordform] += 1
			for ch in wordform:
				if not isDigit(ch):
					char_vocab[ch] += 1

			seq = '^' + wordform + '$'
			n = 3
			ngram_counts = len(seq) - n + 1
			for i in range(ngram_counts):
				ngram = seq[i:i + n]
				if not isDigit(ngram):
					ngram_vocab[ngram] += 1

	vocab = {'word': word_vocab, 'char': char_vocab, 'ngram': ngram_vocab}
	return vocab


def compute_overlap(vocab1, vocab2):

	stats = {}
	for vtype in vocab1:
		overlap = set(vocab1[vtype].keys()).intersection(set(vocab2[vtype].keys()))
		total = set(vocab1[vtype].keys()).union(set(vocab2[vtype].keys()))

		print(len(overlap), len(total))
		stats[vtype] = round(len(overlap) * 100 / len(total), 2)

	return stats


lang_pairs = [
	# ('et', 'fi'),
	('it_isdtt', 'es_ancora')
]

ud_iso_file = codecs.open('./src/utils/ud_iso.json',encoding='utf-8')
iso_dict = json.loads(ud_iso_file.read())
iso2tb = {}

for item in iso_dict:
	iso = iso_dict[item]
	iso2tb[iso] = item

ud_dir = '../data/ud_5k'
for iso1, iso2 in lang_pairs:
	tb1 = os.path.join(ud_dir, iso2tb[iso1], iso1 + '-ud-train.conllu')
	tb2 = os.path.join(ud_dir, iso2tb[iso2], iso2 + '-ud-train.conllu')

	tb1_vocab = create_vocab(tb1)
	tb2_vocab = create_vocab(tb2)

	stats = compute_overlap(tb1_vocab, tb2_vocab)

	print(iso1, iso2)
	for s in stats:
		print(s, ':', stats[s])
	print()
