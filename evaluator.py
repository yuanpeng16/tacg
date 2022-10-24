import tensorflow as tf


class Evaluator(object):
    def __init__(self, args, model, datasets):
        self.args = args
        self.model = model
        self.datasets = datasets

    def forward(self, x):
        y1_hat, y2_hat = [], []
        size = self.args.batch_size
        for i in range(0, len(x), size):
            j = min(i + size, len(x))
            y1, y2 = self.model(x[i:j])
            y1 = tf.argmax(y1, -1).numpy()
            y2 = tf.argmax(y2, -1).numpy()
            y1_hat.extend(y1)
            y2_hat.extend(y2)
        return y1_hat, y2_hat

    def get_accuracy(self, y_hat, y):
        n_samples = len(y[0])
        y1_hat_list = y_hat[0]
        y2_hat_list = y_hat[1]
        hit1, hit2, hit = 0, 0, 0
        for i in range(n_samples):
            y1_hat = y1_hat_list[i]
            y2_hat = y2_hat_list[i]
            h1 = y[0][i][y1_hat] == 1
            h2 = y[1][i][y2_hat] == 1
            if h1:
                hit1 += 1
            if h2:
                hit2 += 1
            if h1 and h2:
                hit += 1
        acc = hit / n_samples
        acc1 = hit1 / n_samples
        acc2 = hit2 / n_samples
        return acc1, acc2, acc

    def evaluate(self, x, y):
        y_hat = self.forward(x)
        return self.get_accuracy(y_hat, y)

    def evaluate_datasets(self, datasets):
        ret = []
        for dataset in datasets:
            ret.extend(
                self.evaluate(dataset[0], dataset[1]))
            ret.append("\t")
        return ret

    def evaluate_all(self):
        return self.evaluate_datasets(self.datasets)
