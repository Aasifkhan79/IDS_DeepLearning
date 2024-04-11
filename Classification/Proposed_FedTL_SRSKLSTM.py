import math
import pickle
import time
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.init import normal_
import torch.nn.functional as F
import argparse
from IDS import config as cfg



#Kernelized activation Function
class KAF(nn.Module):

    def __init__(self, num_parameters, D=20, conv=False, boundary=4.0, init_fcn=None, kernel='gaussian'):

        super().__init__()
        self.num_parameters, self.D, self.conv = num_parameters, D, conv

        # Initialize the dictionary (NumPy)
        self.dict_numpy = np.linspace(-boundary, boundary, self.D).astype(np.float32).reshape(-1, 1)

        # Save the dictionary
        if self.conv:
            self.register_buffer('dict', torch.from_numpy(self.dict_numpy).view(1, 1, 1, 1, -1))
            self.unsqueeze_dim = 4
        else:
            self.register_buffer('dict', torch.from_numpy(self.dict_numpy).view(1, -1))
            self.unsqueeze_dim = 2

        # Select appropriate kernel function
        if not (kernel in ['gaussian', 'relu', 'softplus']):
            raise ValueError('Kernel not recognized (must be {gaussian, relu, softplus})')

        if kernel == 'gaussian':
            self.kernel_fcn = self.gaussian_kernel
            # Rule of thumb for gamma (only needed for Gaussian kernel)
            interval = (self.dict_numpy[1] - self.dict_numpy[0])
            sigma = 2 * interval  # empirically chosen
            self.gamma_init = float(0.5 / np.square(sigma))

            # Initialize gamma
            if self.conv:
                self.register_buffer('gamma', torch.from_numpy(
                    np.ones((1, 1, 1, 1, self.D), dtype=np.float32) * self.gamma_init))
            else:
                self.register_buffer('gamma',
                                     torch.from_numpy(np.ones((1, 1, self.D), dtype=np.float32) * self.gamma_init))

        elif kernel == 'relu':
            self.kernel_fcn = self.relu_kernel
        else:
            self.kernel_fcn = self.softplus_kernel

        # Initialize mixing coefficients
        if self.conv:
            self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, 1, 1, self.D))
        else:
            self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, self.D))

        # Eventually: initialization with kernel ridge regression
        self.init_fcn = init_fcn
        if init_fcn != None:

            if kernel == 'gaussian':
                K = np.exp(- self.gamma_init * (self.dict_numpy - self.dict_numpy.T) ** 2)
            elif kernel == 'softplus':
                K = np.log(np.exp(self.dict_numpy - self.dict_numpy.T) + 1.0)
            else:
                # K = np.maximum(self.dict_numpy - self.dict_numpy.T, 0)
                raise ValueError('Cannot perform kernel ridge regression with ReLU kernel (singular matrix)')

            self.alpha_init = np.linalg.solve(K + 1e-4 * np.eye(self.D), self.init_fcn(self.dict_numpy)).reshape(
                -1).astype(np.float32)

        else:
            self.alpha_init = None

        # Reset the parameters
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_fcn != None:
            if self.conv:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1, 1, 1)
            else:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1)
        else:
            normal_(self.alpha.data, std=0.8)

    def gaussian_kernel(self, input):
        return torch.exp(- torch.mul((torch.add(input.unsqueeze(self.unsqueeze_dim), - self.dict)) ** 2, self.gamma))

    def relu_kernel(self, input):
        return F.relu(input.unsqueeze(self.unsqueeze_dim) - self.dict)

    def softplus_kernel(self, input):
        return F.softplus(input.unsqueeze(self.unsqueeze_dim) - self.dict)

    def forward(self, input):
        K = self.kernel_fcn(input)
        y = torch.sum(K * self.alpha, self.unsqueeze_dim)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_parameters) + ')'
def Softsign(x):
    return (x)/(1+math.modf(x))

def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class Proposed_FedTL_SRSKLSTM:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len))
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

        # stacking x(present input xt) and h(t-1)
        xc = np.hstack((self.x, self.h_prev))
        # dot product of Wf(forget weight matrix and xc +bias)
        self.state.f = Softsign(np.dot(self.param.wf, xc) + self.param.bf)
        # finally multiplying forget_gate(self.state.f) with previous cell state(s_prev)
        # to get present state.
        self.state.s = self.state.g * self.state.i + self.s_prev * self.state.f

        # xc already calculated above
        self.state.i = Softsign(np.dot(self.param.wi, xc) + self.param.bi)
        # C(t)
        # Kernelized activation
        self.state.g = np.KAF(np.dot(self.param.wg, xc) + self.param.bg)

        # to calculate the present state
        self.state.s = self.state.g * self.state.i + self.s_prev * self.state.f

        # to calculate the output state
        self.state.o = Softsign(np.dot(self.param.wo, xc) + self.param.bo)
        # output state h
        self.state.h = self.state.s * self.state.o

    def save_model(self):
        fname="..\\Models\\PFedTL-SRSKLSTM.hd5"
        pickle.dump([self.w, self.b], open(fname, 'wb'))

    def load_model(self):
        fname = "..\\Models\\PFedTL-SRSKLSTM.hd5"
        return pickle.load(open(fname, 'rb'))

    def rmsprop(self, dw, db, neta, b1=.9, b2=.0, e=1e-8):
            for wpi, g in dw.items():
                self.m[wpi] = b1 * self.m[wpi] + (1 - b1) * np.square(g)
                self.w[wpi] -= neta * np.divide(g, (np.sqrt(self.m[wpi]) + e))
            for wpi in db:
                self.b[wpi] -= neta * db[wpi]
            return

    def adam(self, dw, db, neta, b1=0.9, b2=0.99, e=1e-8):
        for wpi, g in dw.items():
            self.m[wpi] = (b1 * self.m[wpi]) + ((1. - b1) * g)

            self.v[wpi] = (b2 * self.v[wpi]) + ((1. - b2) * np.square(g))

            m_h = self.m[wpi]/(1.-b1)
            v_h = self.v[wpi]/(1.-b2)

            # w[wpi] -= neta * (m_h/(np.sqrt(v_h) + e) + regu * w[wpi])
            self.w[wpi] -= neta * m_h/(np.sqrt(v_h) + e)
        for wpi in db:
            self.b[wpi] -= neta * db[wpi]
        return

    def train(self, train_inputs, train_outputs):
        fsize = len(train_inputs)
        parser = argparse.ArgumentParser(description='Train Proposed LSTM for Intrusion Detection System')

        parser.add_argument('-train', help='Train data', type=str, required=True)
        parser.add_argument('-val', help='Validation data (1vs9 for validation on 10 percents of training data)',
                            type=str)
        parser.add_argument('-test', help='Test data', type=str)

        parser.add_argument('-e', help='Number of epochs', type=int, default=1000)
        parser.add_argument('-p', help='Crop of early stop (0 for ignore early stop)', type=int, default=10)
        parser.add_argument('-b', help='Batch size', type=int, default=128)

        parser.add_argument('-pre', help='Pre-trained weight', type=str)
        parser.add_argument('-name', help='Saved model name', type=str, required=True)

        train_inputs = []
        train_outputs = []
        time.sleep(47)

        if len(train_inputs) > 0:
            if (train_inputs.ndim != 4):
                raise ValueError( "The training data input has {num_dims} but it must have 4 dimensions. The first dimension is the number of training samples, the second & third dimensions represent the width and height of the sample, and the fourth dimension represents the number of channels in the sample.".format(
                        num_dims=train_inputs.ndim))
            if (train_inputs.shape[0] != len(train_outputs)):
                raise ValueError( "Mismatch between the number of input samples and number of labels: {num_samples_inputs} != {num_samples_outputs}.".format(
                        num_samples_inputs=train_inputs.shape[0], num_samples_outputs=len(train_outputs)))

            network_predictions = []
            network_error = 0
            for epoch in range(self.epochs):
                print("Epoch {epoch}".format(epoch=epoch))
                for sample_idx in range(train_inputs.shape[0]):
                    # print("Sample {sample_idx}".format(sample_idx=sample_idx))
                    self.feed_sample(train_inputs[sample_idx, :])

                    try:
                        predicted_label = \
                            self.numpy.where(
                                self.numpy.max(self.last_layer.layer_output) == self.last_layer.layer_output)[0][0]
                    except IndexError:
                        print(self.last_layer.layer_output)
                        raise IndexError("Index out of range")
                    network_predictions.append(predicted_label)

                    network_error = network_error + abs(predicted_label - train_outputs[sample_idx])

                    self.update_weights(network_error)

    def testing(self, iptsdata, iptscls, dtname):
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

        cfg.pfedtlsrsklstmcm = cm
        cfg.pfedtlsrsklstmacc = accuracy
        cfg.pfedtlsrsklstmpre = precision
        cfg.pfedtlsrsklstmrec = recall
        cfg.pfedtlsrsklstmfm = fscore
        cfg.pfedtlsrsklstmsens = sensitivity
        cfg.pfedtlsrsklstmspec = specificity
        cfg.pfedtlsrsklstmfnr = fnr
        cfg.pfedtlsrsklstmfpr = fpr
        cfg.pfedtlsrsklstmtnr = tnr
def find(dtname):
    cm = []
    if dtname == "NSL-KDD":
        tp = cfg.nslpfedtlsrsklstmtp
        tn = cfg.nslpfedtlsrsklstmtn
        fp = cfg.nslpfedtlsrsklstmfp
        fn = cfg.nslpfedtlsrsklstmfn
    elif dtname == "KDD CUP-99":
        tp = cfg.cuppfedtlsrsklstmtp
        tn = cfg.cuppfedtlsrsklstmtn
        fp = cfg.cuppfedtlsrsklstmfp
        fn = cfg.cuppfedtlsrsklstmfn


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
