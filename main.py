# -*- coding:utf-8 -*-

import argparse
import d2l
from data import FashionMnistDataLoader
from model import FashionMnistModel, MxFashionMnistModel, evaluate_accuracy
from model import MlpFashionMnistModel, MxMlpFashionMnistModel
from model import MlpFashionMnistDropOutModel, MxMlpFashionMnistDropOutModel, ConvLeNetModel
from mxnet import autograd

num_epochs = 15
lr = 0.1
num_inputs = 28 * 28
num_hiddens = 256
num_hiddens2 = 256
num_outputs = 10
batch_size = 256

drop_prob1, drop_prob2 = 0.2, 0.5


def main():
    parser = argparse.ArgumentParser(description='D2L Fashion MNIST')
    parser.add_argument('--run_mode', type=str, nargs='?', default='mxnet', help='input run_mode. "raw" or "mxnet"')
    parser.add_argument('--net_mode', type=str, nargs='?', default='slp', help='input net_mode. "slp" or "mlp" or "mlp_drop"')
    args = parser.parse_args()
    run_mode = args.run_mode
    net_mode = args.net_mode

    print(f'run {run_mode} code ...')
    print(f'run {net_mode} ...')

    if run_mode == 'raw':
        if net_mode == 'slp':
            model = FashionMnistModel(num_inputs, num_outputs)
        elif net_mode == 'mlp':
            model = MlpFashionMnistModel(num_inputs, num_hiddens, num_outputs)
        else:
            model = MlpFashionMnistDropOutModel(num_inputs, num_hiddens, num_hiddens2, num_outputs, drop_prob1, drop_prob2)

        data_loader = FashionMnistDataLoader(batch_size)
        trainer = None
    else:
        if net_mode == 'slp':
            model = MxFashionMnistModel(num_outputs, lr)
        elif net_mode == 'mlp':
            model = MxMlpFashionMnistModel(num_hiddens, num_outputs, lr)
        elif net_mode == 'LeNet':
            model = ConvLeNetModel(lr)
        else:
            model = MxMlpFashionMnistDropOutModel(num_hiddens, num_hiddens2, num_outputs, drop_prob1, drop_prob2, lr)

        data_loader = FashionMnistDataLoader(batch_size)
        trainer = model.trainer

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in data_loader.train_iter:
            with autograd.record():
                y_hat = model.net(X)
                l = model.loss(y_hat, y).sum()
            l.backward()

            if trainer is None:
                d2l.sgd(model.get_params(), lr, batch_size)
            else:
                # This will be illustrated in the next section
                trainer.step(batch_size)

            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(data_loader.test_iter, model.net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


if __name__ == '__main__':
    main()
