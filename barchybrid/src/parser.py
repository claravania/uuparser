from optparse import OptionParser, OptionGroup
from arc_hybrid import ArcHybridLSTM
from options_manager import OptionsManager
import pickle, utils, os, time, sys, copy, itertools, re, random
from shutil import copyfile
import codecs


def run(om,options,i):

    if options.multiling:
        outdir = options.outdir
    else:
        cur_treebank = om.languages[i]
        outdir = cur_treebank.outdir

    if options.shared_task:
        outdir = options.shared_task_outdir

    if not options.predict: # training

        fineTune = False
        start_from = 1
        if options.continueModel is None:
            continueTraining = False
        else:
            continueTraining = True
            trainedModel = options.continueModel
            if options.fineTune:
                fineTune = True
            else:
                start_from = options.first_epoch - 1

        if not continueTraining:
            print 'Preparing vocab'
            if options.multiling:
                path_is_dir=True,
                words, w2i, pos, cpos, rels, langs, ch = utils.vocab(om.languages,\
                                                                     path_is_dir,
                                                                     options.shareWordLookup,\
                                                                     options.shareCharLookup)

            else:
                words, w2i, pos, cpos, rels, langs, ch = utils.vocab(cur_treebank.trainfile)

            paramsfile = os.path.join(outdir, options.params)
            with open(paramsfile, 'w') as paramsfp:
                print 'Saving params to ' + paramsfile
                pickle.dump((words, w2i, pos, rels, cpos, langs,
                             options, ch), paramsfp)
                print 'Finished collecting vocab'
        else:
            paramsfile = os.path.join(outdir, options.params)
            with open(paramsfile, 'rb') as paramsfp:
                print 'Load params from ' + paramsfile
                words, w2i, pos, rels, cpos, langs, options, ch = pickle.load(paramsfp)
                print 'Finished loading vocab'

        max_epochs = options.first_epoch + options.epochs
        print 'Initializing blstm arc hybrid:'
        parser = ArcHybridLSTM(words, pos, rels, cpos, langs, w2i,
                               ch, options)

        if continueTraining:
            if not fineTune: 
                # continue training only, not doing fine tuning
                options.first_epoch = start_from + 1
                max_epochs = options.epochs
            else:
                # fine tune model
                options.first_epoch = options.epochs + 1
                max_epochs = options.first_epoch + 15
                print 'Fine tune model for another', max_epochs - options.first_epoch, 'epochs'

            parser.Load(trainedModel)
            

        best_multi_las = -1
        best_multi_epoch = 0
        
        if continueTraining:
            train_stats = codecs.open(os.path.join(outdir, 'train.stats'), 'a', encoding='utf-8')
        else:
            train_stats = codecs.open(os.path.join(outdir, 'train.stats'), 'w', encoding='utf-8')
                
        for epoch in xrange(options.first_epoch, max_epochs + 1):

            print 'Starting epoch ' + str(epoch)

            if options.multiling:
                traindata = list(utils.read_conll_dir(om.languages, "train", options.max_sentences))
            else:
                traindata = list(utils.read_conll(cur_treebank.trainfile, cur_treebank.iso_id,options.max_sentences))

            parser.Train(traindata)
            train_stats.write(unicode('Epoch ' + str(epoch) + '\n'))
            print 'Finished epoch ' + str(epoch)

            model_file = os.path.join(outdir, options.model + '.tmp')
            parser.Save(model_file)

            if options.pred_dev: # use the model to predict on dev data
                if options.multiling:
                    pred_langs = [lang for lang in om.languages if lang.pred_dev] # languages which have dev data on which to predict
                    for lang in pred_langs:
                        lang.outfilename = os.path.join(lang.outdir, 'dev_epoch_' + str(epoch) + '.conllu')
                        print "Predicting on dev data for " + lang.name
                    devdata = utils.read_conll_dir(pred_langs,"dev")
                    pred = list(parser.Predict(devdata))

                    if len(pred)>0:
                        utils.write_conll_multiling(pred,pred_langs)
                    else:
                        print "Warning: prediction empty"
                    
                    if options.pred_eval:
                        total_las = 0
                        for lang in pred_langs:
                            print "Evaluating dev prediction for " + lang.name
                            las_score = utils.evaluate(lang.dev_gold, lang.outfilename,om.conllu)
                            total_las += las_score
                            train_stats.write(unicode('Dev LAS ' + lang.name + ': ' + str(las_score) + '\n'))
                        if options.model_selection:
                            if total_las > best_multi_las:
                                best_multi_las = total_las
                                best_multi_epoch = epoch 

                else: # monolingual case
                    if cur_treebank.pred_dev:
                        print "Predicting on dev data for " + cur_treebank.name
                        devdata = utils.read_conll(cur_treebank.devfile, cur_treebank.iso_id)
                        cur_treebank.outfilename = os.path.join(outdir, 'dev_epoch_' + str(epoch) + ('.conll' if not om.conllu else '.conllu'))
                        pred = list(parser.Predict(devdata))
                        utils.write_conll(cur_treebank.outfilename, pred)
                        if options.pred_eval:
                            print "Evaluating dev prediction for " + cur_treebank.name
                            las_score = utils.evaluate(cur_treebank.dev_gold, cur_treebank.outfilename, om.conllu)
                            if options.model_selection:
                                if las_score > cur_treebank.dev_best[1]:
                                    cur_treebank.dev_best = [epoch, las_score]
                                    train_stats.write(unicode('Dev LAS ' + cur_treebank.name + ': ' + str(las_score) + '\n'))
                                    

            if epoch == max_epochs: # at the last epoch choose which model to copy to barchybrid.model
                if not options.model_selection:
                    best_epoch = options.epochs # take the final epoch if model selection off completely (for example multilingual case)
                else:
                    if options.multiling:
                        best_epoch = best_multi_epoch
                    else:
                        best_epoch = cur_treebank.dev_best[0] # will be final epoch by default if model selection not on for this treebank
                        if cur_treebank.model_selection:
                            print "Best dev score of " + str(cur_treebank.dev_best[1]) + " found at epoch " + str(cur_treebank.dev_best[0])

                bestmodel_file = os.path.join(outdir,"barchybrid.model.tmp")
                model_file = os.path.join(outdir,"barchybrid.model")
                if fineTune:
                    model_file = os.path.join(outdir,"barchybrid.tuned.model")
                print "Best epoch: " + str(best_epoch)
                print "Copying " + bestmodel_file + " to " + model_file
                copyfile(bestmodel_file,model_file)

        train_stats.close()

    else: #if predict - so

        # import pdb;pdb.set_trace()
        eval_type = options.evaltype
        print "Eval type: ", eval_type
        if eval_type == "train":
            if options.multiling:
                for l in om.languages:
                    l.test_gold = l.test_gold.replace('test', 'train')
            else:
                cur_treebank.testfile = cur_treebank.trainfile
                cur_treebank.test_gold = cur_treebank.trainfile

        elif eval_type == "dev":
            if options.multiling:
                for l in om.languages:
                    l.test_gold = l.test_gold.replace('test', 'dev')
            else:
                cur_treebank.testfile = cur_treebank.devfile
                cur_treebank.test_gold = cur_treebank.devfile

        if options.multiling:
            modeldir = options.modeldir
            if options.fineTune:
                prefix = [os.path.join(outdir, os.path.basename(l.test_gold) + '-tuned') for l in om.languages] 
            else:
                prefix = [os.path.join(outdir, os.path.basename(l.test_gold)) for l in om.languages] 
        else:
            modeldir = om.languages[i].modeldir
            if options.fineTune:
                prefix = os.path.join(outdir, os.path.basename(cur_treebank.testfile)) + '-tuned'
            else:
                prefix = os.path.join(outdir, os.path.basename(cur_treebank.testfile))

        if not options.extract_vectors:
            prefix = None


        params = os.path.join(modeldir, options.params)
        print 'Reading params from ' + params
        with open(params, 'r') as paramsfp:
            words, w2i, pos, rels, cpos, langs, stored_opt, ch = pickle.load(paramsfp)

            parser = ArcHybridLSTM(words, pos, rels, cpos, langs, w2i,
                               ch, stored_opt)

            if options.fineTune:
                options.model = options.model.replace('.model', '.tuned.model')
            model = os.path.join(modeldir, options.model)
            parser.Load(model)

            if options.multiling:
                testdata = utils.read_conll_dir(om.languages, eval_type)
            else:
                testdata = utils.read_conll(cur_treebank.testfile, cur_treebank.iso_id)

            ts = time.time()

            if options.multiling:
                for l in om.languages:
                    l.outfilename = os.path.join(outdir, eval_type + "-" + l.outfilename)
                pred = list(parser.Predict(testdata, prefix))
                utils.write_conll_multiling(pred,om.languages)
            else:
                if cur_treebank.outfilename:
                    cur_treebank.outfilename = os.path.join(outdir, eval_type + "-" + cur_treebank.outfilename)
                else:
                    cur_treebank.outfilename = os.path.join(outdir, 'out' + ('.conll' if not om.conllu else '.conllu'))
                utils.write_conll(cur_treebank.outfilename, parser.Predict(testdata, prefix))

            te = time.time()

            if options.pred_eval:
                if options.multiling:
                    for l in om.languages:
                        print "Evaluating on " + l.name
                        score = utils.evaluate(l.test_gold, l.outfilename, om.conllu)
                        print "Obtained LAS F1 score of %.2f on %s" %(score, l.name)
                else:
                    print "Evaluating on " + cur_treebank.name
                    score = utils.evaluate(cur_treebank.test_gold, cur_treebank.outfilename, om.conllu)
                    print "Obtained LAS F1 score of %.2f on %s" %(score,cur_treebank.name)

            print 'Finished predicting'

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--outdir", metavar="PATH", help='Output directory')
    parser.add_option("--datadir", metavar="PATH",
        help="Input directory with UD train/dev/test files; obligatory if using --include")
    parser.add_option("--modeldir", metavar="PATH",
        help='Directory where models will be saved, defaults to same as --outdir if not specified')
    parser.add_option("--params", metavar="FILE", default="params.pickle", help="Parameters file")
    parser.add_option("--model", metavar="FILE", default="barchybrid.model",
        help="Load/Save model file")

    group = OptionGroup(parser, "Experiment options")
    group.add_option("--include", metavar="LIST", help="List of languages by ISO code to be run \
if using UD. If not specified need to specify trainfile at least. When used in combination with \
--multiling, trains a common parser for all languages. Otherwise, train monolingual parsers for \
each")
    group.add_option("--trainfile", metavar="FILE", help="Annotated CONLL(U) train file")
    group.add_option("--devfile", metavar="FILE", help="Annotated CONLL(U) dev file")
    group.add_option("--testfile", metavar="FILE", help="Annotated CONLL(U) test file")
    group.add_option("--evaltype", type="string", default="test", help="type of eval: train, dev, test")
    group.add_option("--epochs", type="int", metavar="INTEGER", default=30,
        help='Number of epochs')
    group.add_option("--predict", help='Parse', action="store_true", default=False)
    group.add_option("--fineTune", help='Parse', action="store_true", default=False)
    group.add_option("--multiling", action="store_true", default=False,
        help='Train a multilingual parser with language embeddings')
    group.add_option("--max-sentences", type="int", metavar="INTEGER",
        help='Only train using n sentences per epoch', default=-1)
    group.add_option("--create-dev", action="store_true", default=False,
        help='Create dev data if no dev file is provided')
    group.add_option("--min-train-sents", type="int", metavar="INTEGER", default=1000,
        help='Minimum number of training sentences required in order to create a dev file')
    group.add_option("--dev-percent", type="float", metavar="FLOAT", default=5,
        help='Percentage of training data to use as dev data')
    group.add_option("--disable-pred-dev", action="store_false", dest="pred_dev", default=True,
        help='Disable prediction on dev data after each epoch')
    group.add_option("--disable-pred-eval", action="store_false", dest="pred_eval", default=True,
        help='Disable evaluation of prediction on dev data')
    group.add_option("--disable-model-selection", action="store_false",
        help="Disable choosing of model from best/last epoch", dest="model_selection", default=True)
    group.add_option("--use-default-seed", action="store_true",
        help="Use default random seed for Python", default=False)
    group.add_option("--extract_vectors", action="store_true",
        help="Extract intermediate representations of the model (predict mode only)", default=False)

    group.add_option("--continue", dest="continueTraining", action="store_true", default=False)
    group.add_option("--continueModel", metavar="FILE",
        help="Load model file, when continuing to train a previously trained model")
    group.add_option("--first-epoch", type="int", metavar="INTEGER", default=1)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Parser options")
    group.add_option("--disable-oracle", action="store_false", dest="oracle", default=True,
        help='Use the static oracle instead of the dynamic oracle')
    group.add_option("--disable-head", action="store_false", dest="headFlag", default=True,
        help='Disable using the head of word vectors fed to the MLP')
    group.add_option("--disable-rlmost", action="store_false", dest="rlMostFlag", default=True,
        help='Disable using leftmost and rightmost dependents of words fed to the MLP')
    group.add_option("--userl", action="store_true", dest="rlFlag", default=False)
    group.add_option("--k", type="int", metavar="INTEGER", default=3,
        help="Number of stack elements to feed to MLP")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Neural network options")
    group.add_option("--dynet-seed", type="int", metavar="INTEGER",
        help="Random seed for Dynet")
    group.add_option("--dynet-mem", type="int", metavar="INTEGER",
        help="Memory to assign Dynet in MB", default=512)
    group.add_option("--learning-rate", type="float", metavar="FLOAT",
        help="Learning rate for neural network optimizer", default=0.001)
    group.add_option("--char-emb-size", type="int", metavar="INTEGER",
        help="Character embedding dimensions", default=24)
    group.add_option("--char-lstm-output-size", type="int", metavar="INTEGER",
        help="Character BiLSTM dimensions", default=50)
    group.add_option("--word-emb-size", type="int", metavar="INTEGER",
        help="Word embedding dimensions", default=100)
    group.add_option("--lang-emb-size", type="int", metavar="INTEGER",
        help="Language embedding dimensions", default=12)
    group.add_option("--lstm-output-size", type="int", metavar="INTEGER",
        help="Word BiLSTM dimensions", default=125)
    group.add_option("--mlp-hidden-dims", type="int", metavar="INTEGER",
        help="MLP hidden layer dimensions", default=100)
    group.add_option("--mlp-hidden2-dims", type="int", metavar="INTEGER",
        help="MLP second hidden layer dimensions", default=0)
    group.add_option("--external-embedding", metavar="FILE", help="External embeddings")
    group.add_option("--activation", help="Activation function in the MLP", default="tanh")
    group.add_option("--disable-bilstm", action="store_true", default=False,
        help='disable the BiLSTM feature extactor')
    group.add_option("--disable-second-bilstm", action="store_true", default=False,
                     help='disable the BiLSTM feature extactor')
    group.add_option("--disable-lembed", action="store_false", dest="use_lembed",
        help='disable the use of a language embedding when in multilingual mode', default=True)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Debug options")
    group.add_option("--debug", action="store_true",
        help="Run parser in debug mode, with fewer sentences", default=False)
    group.add_option("--debug-train-sents", type="int", metavar="INTEGER",
        help="Number of training sentences in --debug mode", default=150)
    group.add_option("--debug-dev-sents", type="int", metavar="INTEGER",
        help="Number of dev sentences in --debug mode", default=100)
    group.add_option("--debug-test-sents", type="int", metavar="INTEGER",
        help="Number of test sentences in --debug mode", default=50)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Shared task options")
    group.add_option("--shared_task", action="store_true", dest="shared_task", default=False)
    group.add_option("--shared_task_datadir", dest="shared_task_datadir", default="EXP")
    group.add_option("--shared_task_outdir", dest="shared_task_outdir", default="EXP")
    parser.add_option_group(group)

    """
    Multilingual Options
    """
    multiling_opt = OptionGroup(parser, "Options for parameter sharing in\
                                multilingual mode")
    multiling_opt.add_option("--separate_mlps", action='store_false',\
                             default=True,dest='shareMLP')
    multiling_opt.add_option("--separate_word_lookups", action='store_false',\
                             default=True, dest='shareWordLookup')
    multiling_opt.add_option("--separate_char_lookup", action='store_false',\
                             default=True, dest='shareCharLookup')
    multiling_opt.add_option("--separate_bilstms", action='store_false',\
                             default=True,dest='shareBiLSTM')
    multiling_opt.add_option("--separate_char_bilstms", action='store_false',\
                             default=True,dest='shareCharBiLSTM')
    multiling_opt.add_option("--lembed_word",action='store_true', default=False)
    multiling_opt.add_option("--lembed_config",action='store_true', default=False)
    multiling_opt.add_option("--lembed_char",action='store_true', default=False)

    """
    Facilitate the naming
    """
    multiling_opt.add_option("--word", type="string",default='shared')
    multiling_opt.add_option("--mlp", type="string",default='shared')
    multiling_opt.add_option("--char", type="string",default='shared')

    parser.add_option_group(multiling_opt)

    (options, args) = parser.parse_args()

    # really important to do this before anything else to make experiments reproducible
    utils.set_seeds(options)

    om = OptionsManager(options)
    for i in range(om.iterations):
        run(om,options,i)
