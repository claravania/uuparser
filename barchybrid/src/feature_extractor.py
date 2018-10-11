import dynet as dy
import random, codecs
from bilstm import BiLSTM

class FeatureExtractor(object):
    def __init__(self,model,wordsCount,rels,langs,words,ch,nnvecs,options):
        """
        Options handling
        """
        self.model = model
        if langs:
            self.langs = {lang: ind+1 for ind, lang in enumerate(langs)} # +1 for padding vector
        else:
            self.langs = None
        self.nnvecs = nnvecs
        self.multiling = options.multiling #and options.use_lembed
        self.external_embedding = None
        if options.external_embedding is not None:
            self.get_external_embeddings(options.external_embedding,model)
        self.disable_bilstm = options.disable_bilstm
        self.disable_second_bilstm = options.disable_second_bilstm

        """sharing"""
        self.shareBiLSTM = options.shareBiLSTM
        self.shareWordLookup = options.shareWordLookup
        self.shareCharLookup = options.shareCharLookup
        self.shareCharBiLSTM = options.shareCharBiLSTM
        self.word_lembed = options.lembed_word
        self.char_lembed = options.lembed_char

        """dims"""
        self.word_emb_size = options.word_emb_size
        self.char_emb_size = options.char_emb_size
        self.lstm_output_size = options.lstm_output_size
        self.char_lstm_output_size = options.char_lstm_output_size
        self.lang_emb_size = options.lang_emb_size

        lstm_input_size = self.word_emb_size + (self.edim if self.external_embedding is\
                          not None else 0) + (self.lang_emb_size if self.word_lembed else 0)\
                          + 2 * self.char_lstm_output_size

        """UTILS"""
        self.wordsCount = wordsCount
        self.irels = rels

        if self.multiling and not self.shareWordLookup:
            w2i = {}
            for lang in self.langs:
                 w2i[lang] = {w: i for i, w in enumerate(words[lang])}
            self.vocab = {}
            for lang in self.langs:
                self.vocab[lang] = {word: ind+2 for word, ind in w2i[lang].iteritems()}

        else:
            w2i = {w: i for i, w in enumerate(words)}
            self.vocab = {word: ind+2 for word, ind in w2i.iteritems()} # +2 for MLP padding vector and OOV vector

        if not self.multiling or self.shareCharLookup:
            self.chars = {char: ind+1 for ind, char in enumerate(ch)} # +1 for OOV vector
        else:
            self.chars = {}
            for lang in self.langs:
                self.chars[lang] = {char: ind+1 for ind, char in enumerate(ch[lang])}
        self.rels = {word: ind for ind, word in enumerate(rels)}

        """BILSTMS"""
        #word
        if not self.multiling or self.shareBiLSTM:
            if not self.disable_bilstm:
                self.bilstm1 = BiLSTM(lstm_input_size, self.lstm_output_size, model,
                                      dropout_rate=0.33)
                if not self.disable_second_bilstm:
                    self.bilstm2 = BiLSTM(2* self.lstm_output_size, self.lstm_output_size, model,
                                          dropout_rate=0.33)
            else:
                self.lstm_output_size = int(lstm_input_size * 0.5)
        else:
            self.bilstm1= {}
            self.bilstm2= {}
            for lang in self.langs:
                self.bilstm1[lang] = BiLSTM(lstm_input_size, self.lstm_output_size, model,
                                      dropout_rate=0.33)
                self.bilstm2[lang] = BiLSTM(2* self.lstm_output_size, self.lstm_output_size, model,
                                            dropout_rate=0.33)

        #char
        if self.char_lembed:
            char_in_dims = self.char_emb_size + self.lang_emb_size
        else:
            char_in_dims = self.char_emb_size

        if not self.multiling or self.shareCharBiLSTM:
            self.char_bilstm = BiLSTM(char_in_dims,self.char_lstm_output_size,self.model,dropout_rate=0.33)
        else:
            self.char_bilstms = {}
            for lang in self.langs:
                self.char_bilstms[lang] = BiLSTM(char_in_dims,self.char_lstm_output_size,self.model,dropout_rate=0.33)

        """LOOKUPS"""
        if not self.multiling or self.shareCharLookup:
            self.clookup = self.model.add_lookup_parameters((len(ch) + 1, self.char_emb_size))
        else:
            self.clookups = {}
            for lang in self.langs:
                self.clookups[lang] = self.model.add_lookup_parameters((len(ch[lang]) + 1, self.char_emb_size))

        if not self.multiling or self.shareWordLookup:
            self.wlookup = self.model.add_lookup_parameters((len(words) + 2, self.word_emb_size))
        else:
            self.wlookups = {}
            for lang in self.langs:
                self.wlookups[lang] = self.model.add_lookup_parameters((len(words[lang]) + 2, self.word_emb_size))

        if self.multiling and self.lang_emb_size > 0:
            self.langslookup = model.add_lookup_parameters((len(langs) + 1, self.lang_emb_size))


        """Padding"""
        self.word2lstm = model.add_parameters((self.lstm_output_size * 2, lstm_input_size))
        self.word2lstmbias = model.add_parameters((self.lstm_output_size *2))
        self.chPadding = model.add_parameters((self.char_lstm_output_size *2))

    def get_char_vec(self,word,dropout,lang=None,langvec=None):
        if word.form == "*root*":
            word.chVec = self.chPadding.expr() # use the padding vector if it's the word token
        else:
            char_vecs = []
            for char in word.form:
                if lang:
                    cvec = self.clookups[lang][self.chars[lang].get(char,0)]
                else:
                    cvec = self.clookup[self.chars.get(char,0)]
                if langvec:
                    char_vecs.append(dy.concatenate([langvec,cvec]))
                else:
                    char_vecs.append(cvec)
            if lang:
                word.chVec = self.char_bilstms[lang].get_sequence_vector(char_vecs,dropout)
            else:
                word.chVec = self.char_bilstm.get_sequence_vector(char_vecs,dropout)

    def Init(self):
        #TODO: This function makes me cry
        #I'm not sure how necessary it is to get different padding vecs
        evec = self.elookup[1] if self.external_embedding is not None else None
        paddingLangVec = self.langslookup[0] if self.multiling and self.lang_emb_size > 0 else None
        if not self.multiling or self.shareWordLookup:
            paddingWordVec = self.wlookup[1]
            #import ipdb;ipdb.set_trace()
            self.paddingVec = dy.tanh(self.word2lstm.expr() * dy.concatenate(filter(None,
                                                                              [paddingWordVec,
                                                                               evec,
                                                                               self.chPadding.expr(),
                                                                               paddingLangVec if self.word_lembed else None]))
                                                                              + self.word2lstmbias.expr() )
            self.empty = self.paddingVec if self.nnvecs == 1 else dy.concatenate([self.paddingVec for _ in xrange(self.nnvecs)])
        else:
            paddingWordVecs = {}
            self.paddingVecs = {}
            self.emptyVecs = {}
            for lang in self.langs:
                paddingWordVecs[lang] = self.wlookups[lang][1]
                self.paddingVecs[lang] = dy.tanh(self.word2lstm.expr() * dy.concatenate(filter(None,
                                                                                        [paddingWordVecs[lang],
                                                                                         evec,
                                                                                         self.chPadding.expr(),
                                                                                         paddingLangVec if self.word_lembed else None]))
                                                                                          + self.word2lstmbias.expr() )
                self.emptyVecs[lang] = self.paddingVecs[lang] if self.nnvecs == 1 else dy.concatenate([self.paddingVecs[lang] for _ in xrange(self.nnvecs)])

    def getWordEmbeddings(self, sentence, train, get_vectors=False):

        lang = sentence[0].language_id

        for root in sentence:
            #word
            if not self.multiling or self.shareWordLookup:
                wordcount = float(self.wordsCount.get(root.norm, 0))
            else:
                wordcount = float(self.wordsCount[lang].get(root.norm, 0))

            noDropFlag =  not train or (random.random() < (wordcount/(0.25+wordcount)))
            if not self.multiling or self.shareWordLookup:
                root.wordvec = self.wlookup[int(self.vocab.get(root.norm, 0)) if noDropFlag else 0]
            else:
                root.wordvec = self.wlookups[lang][int(self.vocab[lang].get(root.norm, 0)) if noDropFlag else 0]

            if self.multiling and self.word_lembed:
                root.langvec = self.langslookup[self.langs[root.language_id]] if self.lang_emb_size > 0 else None
            else:
                root.langvec = None

            #char
            if not self.multiling or self.shareCharBiLSTM:
                if self.char_lembed:
                    langVec = self.langslookup[self.langs[lang]]
                    self.get_char_vec(root,train, langvec=langvec)
                else:
                    self.get_char_vec(root,train)

            else:
                self.get_char_vec(root,train, lang=lang)

            if self.external_embedding is not None:
                if not noDropFlag and random.random() < 0.5:
                    root.evec = self.elookup[0]
                elif root.form in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.form]]
                elif root.norm in self.external_embedding:
                    root.evec = self.elookup[self.extrnd[root.norm]]
                else:
                    root.evec = self.elookup[0]
            else:
                root.evec = None

            root.vec = dy.concatenate(filter(None, [root.wordvec,
                                                    root.evec,
                                                    root.chVec,
                                                    root.langvec]))

        if not self.multiling or self.shareBiLSTM:
            self.bilstm1.set_token_vecs(sentence,train)
            self.bilstm2.set_token_vecs(sentence,train)
        else:
            self.bilstm1[lang].set_token_vecs(sentence,train)
            self.bilstm2[lang].set_token_vecs(sentence,train)

        if get_vectors:
            data_vec = list()
            for token in sentence:
                data_tuple = (token.cpos, token.feats, token.chVec.value(), token.vec.value())
                data_vec.append(data_tuple)
            return data_vec 


    def get_external_embeddings(self,external_embedding_file,model):
        external_embedding_fp = codecs.open(external_embedding_file,'r',encoding='utf-8')
        external_embedding_fp.readline()
        self.external_embedding = {}
        for line in external_embedding_fp:
            line = line.strip().split()
            self.external_embedding[line[0]] = [float(f) for f in line[1:]]

        external_embedding_fp.close()

        self.edim = len(self.external_embedding.values()[0])
        self.noextrn = [0.0 for _ in xrange(self.edim)] #???
        self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
        self.elookup = model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
        for word, i in self.extrnd.iteritems():
            self.elookup.init_row(i, self.external_embedding[word])
        self.extrnd['*PAD*'] = 1
        self.extrnd['*INITIAL*'] = 2

        print 'Load external embedding. Vector dimensions', self.edim
