# -*- coding:utf-8 -*-

from mxnet import ndarray as nd
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn


# The function will be gradually improved: the complete implementation will be
# discussed in the "Image Augmentation" section
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


class FashionMnistModel:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
        self.b = nd.zeros(num_outputs)

        self.W.attach_grad()
        self.b.attach_grad()

    def get_params(self):
        return [self.W, self.b]

    def softmax(self, X):
        X_exp = X.exp()
        partition = X_exp.sum(axis=1, keepdims=True)

        # The broadcast mechanism is applied here
        return X_exp / partition

    def net(self, X):
        return self.softmax(nd.dot(X.reshape((-1, self.num_inputs)), self.W) + self.b)

    # cross_entropy
    def loss(self, y_hat, y):
        return - nd.pick(y_hat, y).log()


class MxFashionMnistModel:
    def __init__(self, num_outputs, lr):
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_outputs))
        self.net.initialize(init.Normal(sigma=0.01))

        self.loss = gloss.SoftmaxCrossEntropyLoss()

        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {
            'learning_rate': lr
        })

class MlpFashionMnistModel:
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
        self.b1 = nd.zeros(num_hiddens)
        self.W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
        self.b2 = nd.zeros(num_outputs)

        self.W1.attach_grad()
        self.b1.attach_grad()
        self.W2.attach_grad()
        self.b2.attach_grad()

        self.loss = gloss.SoftmaxCrossEntropyLoss()

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def relu(self, X):
        return nd.maximum(X, 0)

    def net(self, X):
        X = X.reshape((-1, self.num_inputs))
        H = self.relu(nd.dot(X, self.W1) + self.b1)
        return nd.dot(H, self.W2) + self.b2


class MxMlpFashionMnistModel:
    def __init__(self, num_hiddens, num_outputs, lr):
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens))
        self.net.add(nn.Dense(num_outputs))

        self.net.initialize(init.Normal(sigma=0.01))

        self.loss = gloss.SoftmaxCrossEntropyLoss()

        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {
            'learning_rate': lr
        })