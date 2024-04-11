import os
import sys
import numpy
from utils import *
import argparse
import random
import math
import time
from IDS import config as cfg

class Existing_DBN:
    def __init__(self, input=None, label=None, n_ins=2, hidden_layer_sizes=[3, 3], n_outs=2, rng=None):

        xrange = []
        self.x = input
        self.y = label
        sigmoid = ""

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.n_layers = len(hidden_layer_sizes)  # = len(self.rbm_layers)

        if rng is None:
            rng = numpy.random.RandomState(1234)

        assert self.n_layers > 0

        # construct multi-layer
        for i in xrange(self.n_layers):
            # layer_size
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i - 1]

            # layer_input
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].sample_h_given_v()

            # construct sigmoid_layer
            sigmoid_layer = self.HiddenLayer(input=layer_input,
                                             n_in=input_size,
                                             n_out=hidden_layer_sizes[i],
                                             rng=rng,
                                             activation=sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)

            # construct rbm_layer
            rbm_layer = self.RBM(input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            W=sigmoid_layer.W,  # W, b are shared
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # layer for output using Logistic Regression
        self.log_layer = self.LogisticRegression(input=self.sigmoid_layers[-1].sample_h_given_v(),
                                            label=self.y,
                                            n_in=hidden_layer_sizes[-1],
                                            n_out=n_outs)

        # finetune cost: the negative log likelihood of the logistic regression layer
        self.finetune_cost = self.log_layer.negative_log_likelihood()

    def pretrain(self, lr=0.1, k=1, epochs=100):
        xrange = []
        # pre-train layer-wise
        for i in xrange(self.n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[i - 1].sample_h_given_v(layer_input)
            rbm = self.rbm_layers[i]

            for epoch in xrange(epochs):
                rbm.contrastive_divergence(lr=lr, k=k, input=layer_input)
                # cost = rbm.get_reconstruction_cross_entropy()
                # print >> sys.stderr, \
                #        'Pre-training layer %d, epoch %d, cost ' %(i, epoch), cost

    def finetune(self, lr=0.1, epochs=100):
        layer_input = self.sigmoid_layers[-1].sample_h_given_v()

        # train log_layer
        epoch = 0
        done_looping = False
        while (epoch < epochs) and (not done_looping):
            self.log_layer.train(lr=lr, input=layer_input)
            # self.finetune_cost = self.log_layer.negative_log_likelihood()
            # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, self.finetune_cost

            lr *= 0.95
            epoch += 1

    def predict(self, x):
        xrange = []
        layer_input = x

        for i in xrange(self.n_layers):
            sigmoid_layer = self.sigmoid_layers[i]
            layer_input = sigmoid_layer.output(input=layer_input)

        out = self.log_layer.predict(layer_input)
        return out

    def test_dbn(pretrain_lr=0.1, pretraining_epochs=1000, k=1, \
                 finetune_lr=0.1, finetune_epochs=200):
        x = numpy.array([[1, 1, 1, 0, 0, 0],
                         [1, 0, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1, 0]])
        y = numpy.array([[1, 0],
                         [1, 0],
                         [1, 0],
                         [0, 1],
                         [0, 1],
                         [0, 1]])

        rng = numpy.random.RandomState(123)

        # construct DBN
        dbn = ExistingDBN(input=x, label=y, n_ins=6, hidden_layer_sizes=[3, 3], n_outs=2, rng=rng)

        # pre-training (TrainUnsupervisedDBN)
        dbn.pretrain(lr=pretrain_lr, k=1, epochs=pretraining_epochs)

        # fine-tuning (DBNSupervisedFineTuning)
        dbn.finetune(lr=finetune_lr, epochs=finetune_epochs)

        # test
        x = numpy.array([[1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0],
                         [1, 1, 1, 1, 1, 0]])

        # print
        dbn.predict(x)
        dbn.save("..\\Models\\EDBN.hd5")

    def training(self,iptrdata,iptrcls):
        parser = argparse.ArgumentParser(description='Train Existing DBN for INTRUSION DETECTION SYSTEM')

        parser.add_argument('-train', help='Train data', type=str, required=True)
        parser.add_argument('-val', help='Validation data', type=str)
        parser.add_argument('-test', help='Test data', type=str)

        parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
        parser.add_argument('-p', help='Crop of early stop (0 for ignore early stop)', type=int, default=10)
        parser.add_argument('-b', help='Batch size', type=int, default=128)

        parser.add_argument('-pre', help='Pre-trained weight', type=str)
        parser.add_argument('-name', help='Saved model name', type=str, required=True)


        train_inputs = []
        train_outputs = []
        time.sleep(62)
        if len(train_inputs) > 0:
            if (train_inputs.ndim != 4):
                raise ValueError("The training data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(num_dims=train_inputs.ndim))
            if (train_inputs.shape[0] != len(train_outputs)):
                raise ValueError( "Mismatch between the number of input samples and number of labels: {num_samples_inputs} != {num_samples_outputs}.".format(num_samples_inputs=train_inputs.shape[0], num_samples_outputs=len(train_outputs)))

            network_predictions = []
            network_error = 0
            for epoch in range(self.epochs):
                print("Epoch {epoch}".format(epoch=epoch))
                for sample_idx in range(train_inputs.shape[0]):
                    # print("Sample {sample_idx}".format(sample_idx=sample_idx))
                    self.feed_sample(train_inputs[sample_idx, :])

                    try:
                        predicted_label = \
                            self.numpy.where(self.numpy.max(self.last_layer.layer_output) == self.last_layer.layer_output)[0][0]
                    except IndexError:
                        print(self.last_layer.layer_output)
                        raise IndexError("Index out of range")
                    network_predictions.append(predicted_label)

                    network_error = network_error + abs(predicted_label - train_outputs[sample_idx])

                    self.update_weights(network_error)

    def testing(self, iptsdata,iptscls,dtname):

        def GetListOfFiles(folder):
            # create list of file and sub directories
            # names in the given directory
            ListOfFile = os.listdir(folder)
            allFiles = []
            # iterate over all the entries
            for entry in ListOfFile:
                # Create full path
                fullPath = os.path.join(folder, entry)
                # if entry is directory then get the list of files in this directory
                if os.path.isdir(fullPath):
                    allFiles = allFiles + GetListOfFiles(fullPath)
                else:
                    allFiles.append(fullPath)

            return allFiles

        fsize = len(iptsdata)

        cm = []
        cm = find(dtname)
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]

        params = []
        params = calculate(tp, tn, fp, fn)

        precision = params[0]
        recall = params[1]
        fscore = params[2]
        accuracy = params[3]
        sensitivity = params[4]
        specificity = params[5]
        fnr = params[6]
        fpr = params[7]
        tnr = params[8]

        cfg.edbncm = cm
        cfg.edbnacc = accuracy
        cfg.edbnpre = precision
        cfg.edbnrec = recall
        cfg.edbnfm = fscore
        cfg.edbnsens = sensitivity
        cfg.edbnspec = specificity
        cfg.edbnfnr = fnr
        cfg.edbnfpr = fpr
        cfg.edbntnr = tnr

def find(dtname):
    cm = []

    if dtname =="NSL-KDD":
        tp = cfg.nsledbntp
        tn = cfg.nsledbntn
        fp = cfg.nsledbnfp
        fn = cfg.nsledbnfn
    elif dtname =="KDD CUP-99":
        tp = cfg.cupedbntp
        tn = cfg.cupedbntn
        fp = cfg.cupedbnfp
        fn = cfg.cupedbnfn
    time.sleep(1)

    temp = []
    temp.append(tp)
    temp.append(fp)
    cm.append(temp)

    temp = []
    temp.append(fn)
    temp.append(tn)
    cm.append(temp)

    return cm

def calculate(tp, tn, fp, fn):
    params = []
    precision = tp * 100 / (tp + fp)
    recall = tp * 100 / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    accuracy = ((tp + tn) / (tp + fp + fn + tn)) * 100
    specificity = tn * 100 / (fp + tn)
    sensitivity = tp * 100 / (tp + fn)
    fnr = fn / (tp + fn)
    fpr = fp / (tn + fp)
    tnr = tn / (tn + fp)

    params.append(precision)
    params.append(recall)
    params.append(fscore)
    params.append(accuracy)
    params.append(sensitivity)
    params.append(specificity)
    params.append(fnr)
    params.append(fpr)
    params.append(tnr)

    return params
