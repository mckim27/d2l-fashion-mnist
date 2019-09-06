# -*- coding:utf-8 -*-

import d2l
from mxnet import ndarray as nd
from mxnet import gluon, init, autograd
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


def l2_penalty(w):
    return(w**2).sum() / 2



def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    # In this case, all elements are dropped out
    if drop_prob == 1:
        return X.zeros_like()
    # print(nd.random.uniform(0, 1, X.shape))
    mask = nd.random.uniform(0, 1, X.shape) > drop_prob
    # print(mask)
    return mask * X / (1.0-drop_prob)

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

class MlpFashionMnistDropOutModel:
    def __init__(self, num_inputs, num_hiddens1, num_hiddens2, num_outputs, drop_prob1, drop_prob2):
        self.num_inputs = num_inputs
        self.num_hiddens1 = num_hiddens1
        self.num_hiddens2 = num_hiddens2
        self.num_outputs = num_outputs
        self.drop_prob1 = drop_prob1
        self.drop_prob2 = drop_prob2

        self.W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
        self.b1 = nd.zeros(num_hiddens1)

        self.W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
        self.b2 = nd.zeros(num_hiddens2)

        self.W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
        self.b3 = nd.zeros(num_outputs)

        self.W1.attach_grad()
        self.b1.attach_grad()
        self.W2.attach_grad()
        self.b2.attach_grad()
        self.W3.attach_grad()
        self.b3.attach_grad()

        self.loss = gloss.SoftmaxCrossEntropyLoss()

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def net(self, X):
        X = X.reshape(-1, self.num_inputs)

        H1 = (nd.dot(X, self.W1) + self.b1).relu()
        if autograd.is_training():
            H1 = dropout(H1, self.drop_prob1)

        H2 = (nd.dot(H1, self.W2) + self.b2).relu()
        if autograd.is_training():
            H2 = dropout(H2, self.drop_prob2)

        return nd.dot(H2, self.W3) + self.b3


class MxMlpFashionMnistDropOutModel:
    def __init__(self, num_hiddens1, num_hiddens2, num_outputs, drop_prob1, drop_prob2, lr):
        self.net = nn.Sequential()

        self.net.add(
            nn.Dense(num_hiddens1, activation='relu'),
            nn.Dropout(drop_prob1),
            nn.Dense(num_hiddens2, activation='relu'),
            nn.Dropout(drop_prob2),
            nn.Dense(num_outputs)
        )

        self.net.initialize(init.Normal(sigma=0.01))

        self.loss = gloss.SoftmaxCrossEntropyLoss()

        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {
            'learning_rate': lr
        })


class ConvLeNetModel:
    def __init__(self, lr):
        self.net = nn.Sequential()

        self.net.add(
            nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
            nn.AvgPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
            nn.AvgPool2D(pool_size=2, strides=2),
            # Dense will transform the input of the shape (batch size, channel,
            # height, width) into the input of the shape (batch size,
            # channel * height * width) automatically by default
            nn.Dense(120, activation='sigmoid'),
            nn.Dense(84, activation='sigmoid'),
            nn.Dense(10)
        )

        self.net.initialize(force_reinit=True, init=init.Xavier())

        self.loss = gloss.SoftmaxCrossEntropyLoss()

        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {
            'learning_rate': lr
        })