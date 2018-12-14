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


def transliterate_embeddings(mapping_dict, input_embedding_file, output_embedding_file):
    
    output_embedding = codecs.open(out_embedding_file, 'w', encoding="utf-8")
    
    num_tokens = 0
    dim = -1
    with codecs.open(input_embedding_file, 'r', encoding='utf-8') as f:
	    for i, line in enumerate(f):
	        line = line.strip().split()
	        if i == 0:
	        	dim = int(line[1])
	        	new_line = ' '.join(line) + '\n'
	        	output_embedding.write(unicode(new_line))
	        else:
		        if len(line) != dim + 1: 
		            continue
		        else:
		        	word = line[0]
		        	trans = ''.join([mapping_dict[ch] if ch in mapping_dict else ch for ch in word])
		        	new_line = trans + ' ' + ' '.join(line[1:]) + '\n'
		        	output_embedding.write(unicode(new_line))
		        	num_tokens += 1


mapping_dict_file = sys.argv[1]
in_embedding_file = sys.argv[2]
out_embedding_file = sys.argv[3]

mapping_dict = load_dict(mapping_dict_file)
transliterate_embeddings(mapping_dict, in_embedding_file, out_embedding_file)
