import os
import sys
import copy
import codecs
import random

from collections import defaultdict
from itertools import chain, combinations


class ConllEntry:
    def __init__(self, idx, form, lemma, upos, pos, feats=None, parent_id=None, relation=None, deps=None, misc=None):

        self.idx = idx
        self.form = form
        self.upos = upos
        self.pos = pos
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc


    def set_lexical(self, new_form, new_lemma):
    	self.form = new_form
    	self.lemma = new_lemma


    def __str__(self):
        values = [str(self.idx), self.form, self.lemma, \
                  self.upos,\
                  self.pos,\
                  self.feats, str(self.parent_id), self.relation, \
                  self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])



def all_subsets(ss):
	return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


def load_dict(fname):
	vocab = defaultdict(set)
	with codecs.open(fname, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			wordform, lemma, upos, xfeats, deprel, freq = line.split('\t')
			if xfeats != '_':
				key = (upos, xfeats, deprel)
				vocab[key].add((wordform, lemma))
	return vocab


morphdict_file = sys.argv[1]
infile = sys.argv[2]
outfile = sys.argv[3]

total_sents = 0
max_nonce_sents = 5
total_nonce_sents = 0


morph_dict = load_dict(morphdict_file)
fout = codecs.open(outfile, 'w', encoding= "utf-8")
with codecs.open(infile, 'r', encoding="utf-8") as f:
	sent = []
	idx2shuffle = []
	for line in f:
		line = line.lower().strip()
		if line.startswith('#') or line == '': continue
		
		tok = line.split('\t')
		assert len(tok) == 10
		if '-' in tok[0] or '.' in tok[0]: continue

		if tok[0] == '1' and sent:
			total_sents += 1
			if len(sent) <= 50:
				subsets = list(all_subsets(idx2shuffle))
				subsets = [s for s in subsets if len(s) > 0]
				num_nonce = min(len(subsets)-1, max_nonce_sents)
				random.shuffle(subsets)

				nonce_sents = []
				for i in range(num_nonce):
					nonce = copy.deepcopy(sent)
					for j in subsets[i]:
						j -= 1  # j is index in conllu file (starts with 1, so we need to adjust with the array index)
						curr_key = (sent[j].upos, sent[j].feats, sent[j].relation)
						candidates = list(morph_dict[curr_key])[:]
						random.shuffle(candidates)
						for form, lemma in candidates:
							if form != nonce[j].form:
								nonce[j].set_lexical(form, lemma)
								break
					nonce_sents.append(nonce)

				
				for i, sent in enumerate(nonce_sents):
					for token in sent:
						fout.write(unicode(token) + '\n')
					fout.write('\n')
					total_nonce_sents += 1
				
			sent = []
			idx2shuffle = []

		token = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7].split(':')[0], tok[8], tok[9])
		key = (token.upos, token.feats, token.relation)
		if key in morph_dict and len(morph_dict[key]) > 1:
			idx2shuffle.append(token.idx)
		sent.append(token)

print 'Total original sentences:', str(total_sents)
print 'Total nonce sentences:', str(total_nonce_sents)

		
	



