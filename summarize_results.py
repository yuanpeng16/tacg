import os
import argparse
import numpy as np


def read_file(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    line = lines[-1]
    term = line.strip().split(' ')[-1]
    acc = float(term[:-1])
    return acc


def output(name, mean, std):
    print(name, '&', mean, '$\\pm$', std, '\\\\')


def get_result(model, name):
    results = []
    for i in range(5):
        log_path = os.path.join('logs', model, str(i + 1), 'log.txt')
        acc = read_file(log_path)
        results.append(acc)
    results = np.asarray(results)
    mean = round(np.mean(results), 2)
    std = round(np.std(results), 2)
    output(name, mean, std)


def main(args):
    models = ['baseline', 'proposed', 'no_regularization', 'no_decoder', 'lack_data']
    names = ['Baseline', 'Proposed', 'No regularization', 'No structure', 'Lack training data']
    if args.task == 'lack' or args.task == 'attention':
        models = models[:4]
        names = names[:4]
    models = [os.path.join(args.task, x) for x in models]
    for model, name in zip(models, names):
        get_result(model, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='xor', help='Task type.')
    main(parser.parse_args())
