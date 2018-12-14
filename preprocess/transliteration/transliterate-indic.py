# -*- coding: utf-8 -*-
# @Author: claravania
# @Date:   2018-10-15 11:03:20
# @Last Modified by:   claravania
# @Last Modified time: 2018-11-08 17:22:37


import codecs
import os


from indictrans import Transliterator


ud_dir = '../data/ud-treebanks-v2.2'
hi_tb = 'UD_Urdu-UDTB'



trn = Transliterator(source='urd', target='eng')
filenames = os.listdir(os.path.join(ud_dir, hi_tb))

for f in filenames:
	if f.endswith('.conllu') or f.endswith('.conllu.sample'):
		
		fname = os.path.join(ud_dir, hi_tb, f)
		print 'Reading ' + fname

		ftrn = fname + '.en'
		fout = codecs.open(ftrn, 'w', encoding='utf-8')

		count = 0
		with codecs.open(fname, encoding='utf-8') as fin:
			for line in fin:
				count += 1

				if count % 10000 == 0 and count > 0:
					print(count)
				tok = line.strip()
				if tok.startswith('#') or tok == '':
					fout.write(unicode(tok) + '\n')
					continue

				tok = tok.split('\t')
				if len(tok) == 1 or '.' in tok[0] or '-' in tok[0]:
					fout.write(unicode(tok) + '\n')
					continue

				else:
					tok[1] = trn.transform(tok[1]) or '_'
					tok[2] = trn.transform(tok[2]) or '_'
					entry = '\t'.join(tok) + '\n'
					fout.write(unicode(entry))


		fout.close()