# -*- coding:utf-8 -*-

from mxnet.gluon import data as gdata
import d2l

class FashionMnistDataLoader:
    def __init__(self, batch_size):
        self.mnist_train = gdata.vision.FashionMNIST(train=True)
        self.mnist_test = gdata.vision.FashionMNIST(train=False)

        print(f'len mnist_train : {len(self.mnist_train)}')
        print(f'len mnist_test : {len(self.mnist_test)}')

        features, label = self.mnist_train[0]
        print(f'check features data shape: {features.shape}, dtype: {features.dtype}')
        print(f'check label data: {label}, type: {type(label)}, dtype: {label.dtype}')

        self.train_iter, self.test_iter = d2l.load_data_fashion_mnist(batch_size)

    # This function has been saved in the d2l package for future use
    def get_fashion_mnist_labels(self, labels):
        text_labels = [
            't-shirt', 'trouser', 'pullover', 'dress', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
        return [text_labels[int(i)] for i in labels]

    # This function has been saved in the d2l package for future use
    def show_fashion_mnist(self, images, labels):
        d2l.use_svg_display()
        # Here _ means that we ignore (not use) variables
        _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))

        for f, img, lbl in zip(figs, images, labels):
            f.imshow(img.reshape((28, 28)).asnumpy())
            f.set_title(lbl)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)