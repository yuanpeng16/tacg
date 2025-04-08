import tensorflow as tf


class Evaluator(object):
    def __init__(self, args, model, encoder, datasets):
        self.args = args
        self.model = model
        self.encoder = encoder
        self.datasets = datasets

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    def evaluate_datasets(self, datasets):
        ret = []
        for dataset in datasets:
            ret.extend(self.evaluate(dataset[0], dataset[1]))
            ret.append("\t")
        return ret

    def evaluate_all(self):
        return self.evaluate_datasets(self.datasets)

    def get_hidden_representations(self):
        train = self.encoder(self.datasets[0][0], training=False).numpy()
        test = self.encoder(self.datasets[1][0], training=False).numpy()
        return train, test
