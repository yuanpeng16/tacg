import numpy as np
from sympy.physics.units import length


def one_hot(a, output_nodes):
    ret = [0] * output_nodes
    ret[a] = 1
    return ret


class Dataset(object):
    def __init__(self):
        self.train_samples, self.test_samples = self.get_samples()

    def get_output1(self, x):
        z = [
            x[0] ^ x[1],
            x[1] ^ x[2]
        ]
        y = [z[1], z[0] ^ z[1]]
        y = [one_hot(yi, 2) for yi in y]
        return y

    def get_output(self, x):
        if x[0] == 0:
            z = x[1] % 2
        else:
            z = x[2] % 2
        y = [z, z]
        y = [one_hot(yi, 2) for yi in y]
        return y

    def get_samples(self):
        train_x = [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 0],
        ]
        test_x = [
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 1],
        ]

        train_y = [self.get_output(x) for x in train_x]
        train_y = np.asarray(train_y)
        train_y = np.transpose(train_y, [1, 0, 2])
        train_y = [train_y[0], train_y[1]]
        test_y = [self.get_output(x) for x in test_x]
        test_y = np.asarray(test_y)
        test_y = np.transpose(test_y, [1, 0, 2])
        test_y = [test_y[0], test_y[1]]
        return [np.asarray(train_x), train_y], [np.asarray(test_x), test_y]

    def get_train_samples(self, batch_size=0):
        if batch_size <= 0 or batch_size == len(self.train_samples[0]):
            return self.train_samples
        x, [y_0, y_1] = self.train_samples
        repeats = batch_size // len(x)
        x = np.repeat(x, repeats, 0)
        y_0 = np.repeat(y_0, repeats, 0)
        y_1 = np.repeat(y_1, repeats, 0)
        return [x, [y_0, y_1]]

    def get_test_samples(self):
        return self.test_samples
