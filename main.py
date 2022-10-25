import numpy as np
import argparse
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import Dataset
from models import get_model_generator
from evaluator import Evaluator


def set_random_seeds(parameter_random_seed, data_random_seed):
    random.seed(data_random_seed)
    np.random.seed(data_random_seed)
    tf.random.set_seed(parameter_random_seed)


def train(args, dg, model, ev):
    assert args.steps % args.log_interval == 0
    steps = args.steps // args.log_interval
    x_train, y_train = dg.get_train_samples(args.batch_size)
    for i in range(steps):
        model.fit(x_train, y_train, batch_size=args.batch_size,
                  epochs=args.log_interval, verbose=0)
        print((i + 1) * args.log_interval, *ev.evaluate_all())


def main(args):
    set_random_seeds(args.data_random_seed, args.parameter_random_seed)
    dg = Dataset()
    train_samples = dg.get_train_samples(6)
    test_samples = dg.get_test_samples()
    datasets = [train_samples, test_samples]

    mg = get_model_generator(args)
    model, encoder = mg.get_model()
    model.summary()

    ev = Evaluator(args, model, encoder, datasets)

    # train and evaluate
    print(0, *ev.evaluate_all())
    train(args, dg, model, ev)
    print("final", *ev.evaluate_all())

    # output hidden representations
    hidden_representations = ev.get_hidden_representations()
    markers = ['o', 'x']
    for hidden, marker in zip(hidden_representations, markers):
        assert len(hidden.shape) == 2
        assert hidden.shape[1] == 2
        x, y = np.transpose(hidden)
        plt.scatter(x, y, marker=marker)
        print(hidden)
    plt.savefig('logs/' + args.model + '/hidden_representations.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline',
                        help='Model type.')
    parser.add_argument('--data_random_seed', type=int, default=8,
                        help='Random seed.')
    parser.add_argument('--parameter_random_seed', type=int, default=7,
                        help='Random seed.')
    parser.add_argument('--steps', type=int, default=2000,
                        help='Steps.')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size.')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Alpha.')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta.')
    main(parser.parse_args())
