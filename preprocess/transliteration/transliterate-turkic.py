# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2018-10-15 11:03:20
# @Last Modified by:   claravania
# @Last Modified time: 2018-12-14 11:29:12


import codecs
import os
import sys


def load_dict(alph_dict):
	_dict = {}
	with codecs.open(alph_dict, encoding='utf-8') as f:
		for line in f:
			tok = line.strip().split('\t')
			_dict[tok[0]] = tok[1]
	return _dict


mapping_dict_file = sys.argv[1]
in_file = sys.argv[2]
out_file = sys.argv[3]

mapping_dict = load_dict(mapping_dict_file)


fout = codecs.open(out_file, 'w', encoding='utf-8')

count = 0
with codecs.open(in_file, encoding='utf-8') as fin:
	for line in fin:
		count += 1

		if count % 10000 == 0 and count > 0:
			print(count)

		tok = line.strip()
		if tok.startswith('#') or tok == '':
			fout.write(unicode(line.strip()) + '\n')
			continue

		tok = tok.split('\t')
		tok[1] = ''.join([mapping_dict[ch] if ch in mapping_dict else ch for ch in tok[1]])
		tok[2] = ''.join([mapping_dict[ch] if ch in mapping_dict else ch for ch in tok[2]])
		entry = '\t'.join(tok) + '\n'
		fout.write(unicode(entry))


fout.close()