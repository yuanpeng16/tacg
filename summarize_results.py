import os
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


def main():
    models = ['baseline', 'proposed', 'no_regularization', 'no_decoder',
              'lack_data']
    names = ['Baseline', 'Proposed', 'No regularization', 'No decoder design',
             'Lack training data']
    for model, name in zip(models, names):
        get_result(model, name)


if __name__ == '__main__':
    main()
