import os
import sys
import utils
import copy
import random

# TODO: this whole file is now quite hacky - used to be mostly useful for
# pseudoProj


class OptionsManager(object):

    def __init__(self, options):
        """
        input: parser options
        object to harmonise the way we deal with the parser
        """

        print 'Using external embedding:', options.external_embedding
        self.deal_with_multiling(options)

        if options.include and not options.datadir:
            raise Exception(
                "You need to specify the data dir to include UD languages")

        if not options.predict:
            if not options.include and not options.trainfile:
                raise Exception(
                    "If not using the --include option, you must specify your training data with --trainfile")
        else:
            if not options.include and not options.testfile:
                raise Exception(
                    "If not using the --include option, you must specify your test data with --testfile")
            if not options.modeldir:
                # set model directory to output directory by default
                options.modeldir = options.outdir

        if not options.outdir:
            raise Exception(
                "You must specify an output directory via the --outdir option")
        # create output directory if it doesn't exist
        elif not os.path.exists(options.outdir):
            print "Creating output directory " + options.outdir
            os.mkdir(options.outdir)

        if not options.predict and not (options.rlFlag or options.rlMostFlag or options.headFlag):
            raise Exception(
                "Must include either head, rl or rlmost (For example, if you specified --disable-head and --disable-rlmost, you must specify --userl)")

        if options.rlFlag and options.rlMostFlag:
            print 'Warning: Switching off rlMostFlag to allow rlFlag to take precedence'
            options.rlMostFlag = False

        # TODO: maybe add more sanity checks

        # this is now useless
        options.drop_nproj = False

        options.multi_monoling = False  # set default
        self.iterations = 1  # set default
        self.conllu = True  # default

        if not options.include:  # must specifiy explicitly train
            treebank = utils.Treebank(options.trainfile,
                                      options.devfile, options.testfile)
            treebank.iso_id = None
            treebank.outdir = options.outdir
            treebank.modeldir = options.modeldir
            # just one model specified by train/dev and/or test
            if options.predict:
                if not options.testfile:
                    raise Exception("--testfile must be specified")
                elif not os.path.exists(options.testfile):
                    raise Exception(
                        "Test file " + options.testfile + " not found")
                else:
                    self.conllu = (os.path.splitext(options.testfile.lower())[
                                   1] == '.conllu')  # test if file in conllu format
                    treebank.test_gold = options.testfile
            else:
                self.prepareDev(treebank, options)
                if options.devfile:
                    self.conllu = (os.path.splitext(
                        options.devfile.lower())[1] == '.conllu')
                elif options.create_dev:
                    self.conllu = (os.path.splitext(
                        options.trainfile.lower())[1] == '.conllu')

            if options.debug:
                self.createDebugData(treebank, options)

            # make it a list of one element just for the sake of consistency
            # with the "include" case
            self.languages = [treebank]

        else:
            self.conllu = True  # file is in conllu format
            # languages requested by the user via the include flag
            language_list = utils.parse_list_arg(options.include)
            json_treebanks = utils.conll_dir_to_list(language_list, options.datadir, options.shared_task,  # list of the available treebanks
                                                     options.shared_task_datadir)
#            self.languages = [lang for lang in json_treebanks if lang.iso_id in language_list]
            treebank_dict = {lang.iso_id: lang for lang in json_treebanks}
            self.languages = []
            for lang in language_list:
                if lang in treebank_dict:
                    self.languages.append(treebank_dict[lang])
                else:
                    print "Warning: skipping invalid language code " + lang

            if options.multiling:
                if options.predict:
                    model = "%s/%s" % (options.modeldir,
                                       '-'.join(options.include.split()) + '.model')
                    # in multilingual case need model to be found in first
                    # language specified
                    if not os.path.exists(model):
                        raise Exception(
                            "Model not found. Path tried: %s" % model)
                if options.model_selection:  # can only do model selection for monolingual case
                    print "Warning: model selection on dev data not available for multilingual case"
                    options.model_selection = False
            else:
                options.multi_monoling = True
                self.iterations = len(self.languages)

            for lang_index in xrange(len(self.languages)):
                
                language = self.languages[lang_index]

                language.outdir = "%s/%s" % (options.outdir, language.iso_id)

                # create language-specific output folder if it doesn't exist
                if not os.path.exists(language.outdir):
                    print "Creating language-specific output directory " + language.outdir
                    os.mkdir(language.outdir)
                else:
                    print("Warning: language-specific subdirectory " + language.outdir
                          + " already exists, contents may be overwritten")

                if not options.predict:
                    self.prepareDev(language, options)

                if options.debug:  # it is important that prepareDev be called before createDebugData
                    self.createDebugData(language, options)

                if options.predict and options.multi_monoling:
                    language.modeldir = "%s/%s" % (options.modeldir,
                                                   language.iso_id)
                    # import pdb
                    # pdb.set_trace()
                    model = "%s/%s" % (language.modeldir,
                                       options.include.split()[lang_index] + '.model')
                    
                    # in multilingual case need model to be found in first
                    # language specified
                    if not os.path.exists(model):
                        if not options.shared_task:
                            raise Exception(
                                "Model not found. Path tried: %s" % model)
                        else:
                            # find model for the language in question
                            for otherl in json_treebanks:
                                if otherl.lcode == language.lcode:
                                    if otherl.lcode == otherl.iso_id:
                                        language.modeldir = "%s/%s" % (
                                            options.modeldir, otherl.iso_id)

    # creates dev data by siphoning off a portion of the training data (when necessary)
    # sets up treebank for prediction and model selection on dev data
    def prepareDev(self, treebank, options):
        # even if options.pred_dev is True, might change treebank.pred_dev to
        # False later if no dev data available
        treebank.pred_dev = options.pred_dev
        treebank.model_selection = False
        if not treebank.devfile or not os.path.exists(treebank.devfile):
            if options.create_dev:  # create some dev data from the training data
                traindata = list(utils.read_conll(
                    treebank.trainfile, treebank.iso_id))
                tot_sen = len(traindata)
                if tot_sen > options.min_train_sents:  # need to have at least min_train_sents to move forward
                    # location for the new dev file
                    dev_file = os.path.join(
                        treebank.outdir, 'dev-split' + '.conllu')
                    # location for the new train file
                    train_file = os.path.join(
                        treebank.outdir, 'train-split' + '.conllu')
                    dev_len = int(0.01 * options.dev_percent * tot_sen)
                    print("Taking " + str(dev_len) + " of " + str(tot_sen)
                          + " sentences from training data as new dev data for " + treebank.name)
                    random.shuffle(traindata)
                    devdata = traindata[:dev_len]
                    # write the new dev data to file
                    utils.write_conll(dev_file, devdata)
                    # put the rest of the training data in a new file too
                    traindata = traindata[dev_len:]
                    utils.write_conll(train_file, traindata)
                    # update some variables with the new file locations
                    treebank.dev_gold = dev_file
                    treebank.devfile = dev_file
                    treebank.trainfile = train_file
                else:  # not enough sentences
                    print("Warning: not enough sentences in training data to create dev set for "
                          + treebank.name + " (minimum required --min-train-size: " + str(options.min_train_sents) + ")")
                    treebank.pred_dev = False
            else:  # option --create-dev not set
                print("Warning: No dev data for " + treebank.name
                      + ", consider adding option --create-dev to create dev data from training set")
                treebank.pred_dev = False
        if options.model_selection:
            # epoch (final by default), score of best dev epoch
            treebank.dev_best = [options.epochs, 0]
            if treebank.pred_dev:
                treebank.model_selection = True
            else:
                print "Warning: can't do model selection for " + treebank.name + " as prediction on dev data is off"

    # if debug options is set, we read in the training, dev and test files as appropriate, cap the number of sentences and store
    # new files with these smaller data sets
    def createDebugData(self, treebank, options):
        ext = '.conllu' if self.conllu else '.conll'
        print 'Creating smaller data sets for debugging'
        if not options.predict:
            traindata = list(utils.read_conll(
                treebank.trainfile, treebank.iso_id, maxSize=options.debug_train_sents, hard_lim=True))
            # location for the new train file
            train_file = os.path.join(treebank.outdir, 'train-debug' + ext)
            # write the new dev data to file
            utils.write_conll(train_file, traindata)
            treebank.trainfile = train_file
            if treebank.devfile and os.path.exists(treebank.devfile) and options.pred_dev:
                devdata = list(utils.read_conll(
                    treebank.devfile, treebank.iso_id, maxSize=options.debug_dev_sents, hard_lim=True))
                # location for the new dev file
                dev_file = os.path.join(treebank.outdir, 'dev-debug' + ext)
                # write the new dev data to file
                utils.write_conll(dev_file, devdata)
                treebank.dev_gold = dev_file
                treebank.devfile = dev_file
        else:
            testdata = list(utils.read_conll(
                treebank.testfile, treebank.iso_id, maxSize=options.debug_test_sents, hard_lim=True))
            # location for the new dev file
            test_file = os.path.join(treebank.outdir, 'test-debug' + ext)
            # write the new dev data to file
            utils.write_conll(test_file, testdata)
            treebank.test_gold = test_file
            treebank.testfile = test_file

    def deal_with_multiling(self, options):
        if options.word == 'shared':
            options.shareBiLSTM = True
        elif options.word == 'not_shared':
            options.shareBiLSTM = False
            options.shareWordLookup = False
        elif options.word == 'shared_lembed':
            options.shareBiLSTM = True
            options.lembed_word = True

        if options.mlp == 'shared':
            options.shareMLP = True
        elif options.mlp == 'not_shared':
            options.shareMLP = False
        elif options.mlp == 'shared_lembed':
            options.shareMLP = True
            options.lembed_config = True

        if options.char == 'shared':
            options.shareCharLookup = True
        elif options.char == 'not_shared':
            options.shareCharLookup = False
            options.shareCharBiLSTM = False
        elif options.char == 'share_lembed':
            options.char = True
            options.lembed_char = True
