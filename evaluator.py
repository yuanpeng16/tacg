import tensorflow as tf


class Evaluator(object):
    def __init__(self, args, model, encoder, datasets):
        self.args = args
        self.model = model
        self.encoder = encoder
        self.datasets = datasets

    def forward(self, x):
        y1_hat = []
        size = self.args.batch_size
        for i in range(0, len(x), size):
            j = min(i + size, len(x))
            y1 = self.model(x[i:j], training=False)
            y1 = tf.argmax(y1, -1).numpy()
            y1_hat.extend(y1)
        return y1_hat

    def get_accuracy(self, y_hat, y):
        n_samples = len(y)
        y1_hat_list = y_hat
        y2_hat_list = y_hat
        hit1, hit2, hit = 0, 0, 0
        for i in range(n_samples):
            y1_hat = y1_hat_list[i]
            y2_hat = y2_hat_list[i]
            h1 = y[i][y1_hat] == 1
            h2 = y[i][y2_hat] == 1
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
        loss = self.model.evaluate(x, y, verbose=0)
        return loss[0], self.get_accuracy(y_hat, y)

    def evaluate_datasets(self, datasets):
        ret = []
        for dataset in datasets:
            ret.extend(
                self.evaluate(dataset[0], dataset[1]))
            ret.append("\t")
        return ret

    def evaluate_all(self):
        return self.evaluate_datasets(self.datasets)

    def get_hidden_representations(self):
        train = self.encoder(self.datasets[0][0], training=False).numpy()
        test = self.encoder(self.datasets[1][0], training=False).numpy()
        return train, test
