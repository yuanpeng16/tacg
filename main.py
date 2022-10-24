import numpy as np
import argparse
import random
import tensorflow as tf

from dataset import Dataset
from models import get_model_generator
from evaluator import Evaluator


def set_random_seeds(parameter_random_seed, data_random_seed):
    random.seed(data_random_seed)
    np.random.seed(data_random_seed)
    tf.random.set_seed(parameter_random_seed)


def train(args, dg, model, ev):
    for i in range(args.steps):
        x_train, y_train = dg.get_train_samples(args.batch_size)
        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=1,
                  verbose=0)
        if i % args.log_interval == args.log_interval - 1:
            print(i + 1, *ev.evaluate_all())


def main(args):
    set_random_seeds(args.data_random_seed, args.parameter_random_seed)
    dg = Dataset()
    train_samples = dg.get_train_samples(6)
    test_samples = dg.get_test_samples()
    datasets = [train_samples, test_samples]

    mg = get_model_generator(args)
    model = mg.get_model()

    ev = Evaluator(args, model, datasets)

    # train and evaluate
    print(0, *ev.evaluate_all())
    train(args, dg, model, ev)
    print("final", *ev.evaluate_all())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='baseline',
                        help='Model type.')
    parser.add_argument('--data_random_seed', type=int, default=8,
                        help='Random seed.')
    parser.add_argument('--parameter_random_seed', type=int, default=7,
                        help='Random seed.')
    parser.add_argument('--steps', type=int, default=100,
                        help='Steps.')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval.')
    main(parser.parse_args())
