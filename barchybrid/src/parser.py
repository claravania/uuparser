from optparse import OptionParser, OptionGroup
from arc_hybrid import ArcHybridLSTM
from options_manager import OptionsManager
from shutil import copyfile

import pickle
import utils
import os
import time
import sys
import copy
import itertools
import re
import random
import codecs
import pdb


def run(om, options, i):

    if options.multiling:
        outdir = options.outdir
    else:
        cur_treebank = om.languages[i]
        outdir = cur_treebank.outdir

    if options.shared_task:
        outdir = options.shared_task_outdir

    model_name = ''
    if not options.predict:  # training mode
        print 'Preparing vocab'
        if options.multiling:
            path_is_dir = True
            words, w2i, pos, cpos, rels, langs, ch = utils.vocab(om.languages,
                                                                 path_is_dir,
                                                                 options.shareWordLookup,
                                                                 options.shareCharLookup)
            model_name = '-'.join(options.include.split())

        else:
            words, w2i, pos, cpos, rels, langs, ch = utils.vocab(
                cur_treebank.trainfile)
            model_name = options.include.split()[i]

        paramsfile = os.path.join(outdir, options.params)
        with open(paramsfile, 'w') as paramsfp:
            print 'Saving params to ' + paramsfile
            pickle.dump((words, w2i, pos, rels, cpos, langs,
                         options, ch), paramsfp)
            print 'Finished collecting vocab'

        print 'Initializing blstm arc hybrid:'
        parser = ArcHybridLSTM(words, pos, rels, cpos, langs, w2i,
                               ch, options)
        if options.continueModel is not None:
            parser.Load(options.continueModel)

        best_dev_las = 0.0
        best_epoch = 0
        patience = 3
        model_file = os.path.join(outdir, model_name + '.model')
        for epoch in xrange(options.first_epoch, options.first_epoch + options.epochs):

            print '\n'
            print '===================='
            print 'Starting epoch ' + str(epoch)
            print 'Patience ' + str(patience)

            if options.multiling:
                traindata = list(utils.read_conll_dir(
                    om.languages, "train", options.max_sentences))
            else:
                traindata = list(utils.read_conll(
                    cur_treebank.trainfile, cur_treebank.iso_id, options.max_sentences))


            parser.Train(traindata)
            print 'Finished epoch ' + str(epoch) + '\n'
            

            if options.pred_dev:  # use the model to predict on dev data
                if options.multiling:
                    # languages which have dev data on which to predict
                    pred_langs = [
                        lang for lang in om.languages if lang.pred_dev]
                    for lang in pred_langs:
                        lang.outfilename = os.path.join(
                            lang.outdir, 'dev_epoch_' + str(epoch) + '.conllu')
                        print "Predicting on dev data for " + lang.name

                    devdata = utils.read_conll_dir(pred_langs, "dev")
                    pred = list(parser.Predict(devdata))
                    
                    if len(pred) > 0:
                        utils.write_conll_multiling(pred, pred_langs)
                    else:
                        print "Warning: prediction empty"
                    
                    if options.pred_eval:
                        dev_las = 0
                        for lang in pred_langs:
                            print "Evaluating dev prediction for " + lang.name
                            dev_las += utils.evaluate(lang.dev_gold,
                                                      lang.outfilename, om.conllu)
                        
                        print "*********Total LAS: ", str(dev_las)
                        if dev_las > best_dev_las:
                            parser.Save(model_file)
                            best_epoch = epoch
                            best_dev_las = dev_las
                            patience = 3
                        else:
                            patience -= 1


                else:  # monolingual case
                    if cur_treebank.pred_dev:

                        dev_las = 0
                        print "Predicting on dev data for " + cur_treebank.name
                        devdata = utils.read_conll(
                            cur_treebank.devfile, cur_treebank.iso_id)
                        cur_treebank.outfilename = os.path.join(
                            outdir, 'dev_epoch_' + str(epoch) + ('.conll' if not om.conllu else '.conllu'))
                        pred = list(parser.Predict(devdata))
                        utils.write_conll(cur_treebank.outfilename, pred)
                        
                        if options.pred_eval:
                            print "Evaluating dev prediction for " + cur_treebank.name
                            dev_las = utils.evaluate(
                                cur_treebank.dev_gold, cur_treebank.outfilename, om.conllu)
                            print "LAS: ", str(dev_las)

                            if dev_las > best_dev_las:
                                parser.Save(model_file)
                                best_epoch = epoch
                                best_dev_las = dev_las
                                patience = 3
                            else:
                                patience -= 1

                            # if options.model_selection:
                            #     if score > cur_treebank.dev_best[1]:
                            #         cur_treebank.dev_best = [epoch, score]
                            #         parser.Save(model_file)


            if patience == 0:
                print ''
                print 'No improvement on development set, stop training'
                print 'Best LAS' + str(best_dev_las)
                print 'Best epoch' + str(best_epoch)
                print ''
                break


            # CV: comment this as now we only save the best model
            # if epoch == options.epochs: # at the last epoch choose which model to copy to barchybrid.model
            #     if not options.model_selection:
            #         best_epoch = options.epochs # take the final epoch if model selection off completely (for example multilingual case)
            #     else:
            #         best_epoch = cur_treebank.dev_best[0] # will be final epoch by default if model selection not on for this treebank
            #         if cur_treebank.model_selection:
            # print "Best dev score of " + str(cur_treebank.dev_best[1]) + "
            # found at epoch " + str(cur_treebank.dev_best[0])

            #     bestmodel_file = os.path.join(outdir, model_name + ".model" + str(best_epoch))
            #     model_file = os.path.join(outdir, model_name + ".model")

            #     print "Copying " + bestmodel_file + " to " + model_file
            #     copyfile(bestmodel_file, model_file)


    else:  # if predict - so

        if options.multiling:
            modeldir = options.modeldir
            model_name = '-'.join(options.include.split()) + '.model'
        else:
            modeldir = om.languages[i].modeldir
            model_name = options.include.split()[i] + '.model'

        params = os.path.join(modeldir, options.params)
        print 'Reading params from ' + params
        with open(params, 'r') as paramsfp:
            words, w2i, pos, rels, cpos, langs, stored_opt, ch = pickle.load(
                paramsfp)


            parser = ArcHybridLSTM(words, pos, rels, cpos, langs, w2i,
                                   ch, stored_opt)

            model = os.path.join(modeldir, model_name)
            parser.Load(model)

            if options.multiling:
                testdata = utils.read_conll_dir(om.languages, "test")
            else:
                # this is very hacky to allow predictions on dev set
                cur_treebank.testfile = options.testfile
                cur_treebank.test_gold = options.testfile
                testdata = utils.read_conll(
                    cur_treebank.testfile, cur_treebank.iso_id)

            ts = time.time()

            if options.multiling:
                for l in om.languages:
                    l.outfilename = os.path.join(outdir, l.outfilename)

                pred = list(parser.Predict(testdata))
                utils.write_conll_multiling(pred, om.languages)
            else:
                if cur_treebank.outfilename:
                    cur_treebank.outfilename = os.path.join(
                        outdir, cur_treebank.outfilename)
                else:
                    cur_treebank.outfilename = os.path.join(
                        outdir, 'out' + ('.conll' if not om.conllu else '.conllu'))
                utils.write_conll(cur_treebank.outfilename,
                                  parser.Predict(testdata))

            te = time.time()

            if options.pred_eval:
                if options.multiling:
                    for l in om.languages:
                        print "Evaluating on " + l.name
                        l.test_gold = l.test_gold.replace('test', 'dev')
                        score = utils.evaluate(l.test_gold, l.outfilename, om.conllu)
                        print "Obtained LAS F1 score of %.2f on %s" % (score, l.name)
                else:
                    print "Evaluating on " + cur_treebank.name
                    score = utils.evaluate(
                        cur_treebank.test_gold, cur_treebank.outfilename, om.conllu)
                    print "Obtained LAS F1 score of %.2f on %s" % (score, cur_treebank.name)

            print 'Finished predicting'

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--outdir", metavar="PATH", help='Output directory')
    parser.add_option("--datadir", metavar="PATH",
                      help="Input directory with UD train/dev/test files; obligatory if using --include")
    parser.add_option("--modeldir", metavar="PATH",
                      help='Directory where models will be saved, defaults to same as --outdir if not specified')
    parser.add_option("--params", metavar="FILE",
                      default="params.pickle", help="Parameters file")
    parser.add_option("--model", metavar="FILE", default="barchybrid.model",
                      help="Load/Save model file")

    group = OptionGroup(parser, "Experiment options")
    group.add_option("--include", metavar="LIST", help="List of languages by ISO code to be run \
if using UD. If not specified need to specify trainfile at least. When used in combination with \
--multiling, trains a common parser for all languages. Otherwise, train monolingual parsers for \
each")
    group.add_option("--trainfile", metavar="FILE",
                     help="Annotated CONLL(U) train file")
    group.add_option("--devfile", metavar="FILE",
                     help="Annotated CONLL(U) dev file")
    group.add_option("--testfile", metavar="FILE",
                     help="Annotated CONLL(U) test file")
    group.add_option("--epochs", type="int", metavar="INTEGER", default=30,
                     help='Number of epochs')
    group.add_option("--predict", help='Parse',
                     action="store_true", default=False)
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

    group.add_option("--continue", dest="continueTraining",
                     action="store_true", default=False)
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
    group.add_option("--userl", action="store_true",
                     dest="rlFlag", default=False)
    group.add_option("--k", type="int", metavar="INTEGER", default=3,
                     help="Number of stack elements to feed to MLP")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Neural network options")
    group.add_option("--dynet-seed", type="int", metavar="INTEGER",
                     help="Random seed for Dynet")
    group.add_option("--dynet-mem", type="int", metavar="INTEGER",
                     help="Memory to assign Dynet in MB", default=512)
    group.add_option("--dynet-gpu", action="store_true", default=False,
                     help="flag for using GPU")
    group.add_option("--dynet-devices", type="str", default="",
                     help="CPU/GPU devices to use")
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
    group.add_option("--external-embedding", metavar="FILE",
                     help="External embeddings")
    group.add_option(
        "--activation", help="Activation function in the MLP", default="tanh")
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
    group.add_option("--shared_task", action="store_true",
                     dest="shared_task", default=False)
    group.add_option("--shared_task_datadir",
                     dest="shared_task_datadir", default="EXP")
    group.add_option("--shared_task_outdir",
                     dest="shared_task_outdir", default="EXP")
    parser.add_option_group(group)

    """
    Multilingual Options
    """
    multiling_opt = OptionGroup(parser, "Options for parameter sharing in\
                                multilingual mode")
    multiling_opt.add_option("--separate_mlps", action='store_false',
                             default=True, dest='shareMLP')
    multiling_opt.add_option("--separate_word_lookups", action='store_false',
                             default=True, dest='shareWordLookup')
    multiling_opt.add_option("--separate_char_lookup", action='store_false',
                             default=True, dest='shareCharLookup')
    multiling_opt.add_option("--separate_bilstms", action='store_false',
                             default=True, dest='shareBiLSTM')
    multiling_opt.add_option("--separate_char_bilstms", action='store_false',
                             default=True, dest='shareCharBiLSTM')
    multiling_opt.add_option(
        "--lembed_word", action='store_true', default=False)
    multiling_opt.add_option(
        "--lembed_config", action='store_true', default=False)
    multiling_opt.add_option(
        "--lembed_char", action='store_true', default=False)

    """
    Facilitate the naming
    """
    multiling_opt.add_option("--word", type="string", default='shared')
    multiling_opt.add_option("--mlp", type="string", default='shared')
    multiling_opt.add_option("--char", type="string", default='shared')

    parser.add_option_group(multiling_opt)

    (options, args) = parser.parse_args()

    # really important to do this before anything else to make experiments
    # reproducible
    utils.set_seeds(options)

    om = OptionsManager(options)
    for i in range(om.iterations):
        run(om, options, i)
