import numpy as np
import argparse
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os

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


def dump_hidden_representation(sorted_list, log_dir):
    with open(os.path.join(log_dir, "hidden.txt"), 'w') as f:
        for pair in sorted_list:
            for hidden in pair:
                f.write(str(tuple(hidden)) + '\n')
            f.write("\n")


def get_hidden_representations(ev, log_dir):
    tr, ts = ev.get_hidden_representations()
    sorted_list = [[tr[0], tr[3]], [tr[1], tr[4]], [tr[2], tr[5]], ts]
    sorted_list = np.asarray(sorted_list)
    markers = ['s', 's', '^', '^']
    colors = ['none', 'black', 'black', 'none']
    for hidden, marker, color in zip(sorted_list, markers, colors):
        assert len(hidden.shape) == 2
        assert hidden.shape[1] == 2
        x, y = np.transpose(hidden)
        plt.scatter(x, y, marker=marker, color=color, edgecolors='black')
    plt.savefig(os.path.join(log_dir, "hidden_representations.pdf"))
    dump_hidden_representation(sorted_list, log_dir)


def main(args):
    set_random_seeds(args.data_random_seed, args.parameter_random_seed)
    log_dir = os.path.join("logs", args.model, str(args.parameter_random_seed))

    dg = Dataset()
    train_samples = dg.get_train_samples(6)
    test_samples = dg.get_test_samples()
    datasets = [train_samples, test_samples]

    mg = get_model_generator(args)
    model, encoder = mg.get_model()

    ev = Evaluator(args, model, encoder, datasets)

    # train and evaluate
    print(0, *ev.evaluate_all())
    train(args, dg, model, ev)
    print("final", *ev.evaluate_all())

    # output hidden representations
    get_hidden_representations(ev, log_dir)


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
