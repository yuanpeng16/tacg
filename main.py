class Dataset(object):
    def __init__(self):
        self.train_samples, self.test_samples = self.get_samples()

    def get_output(self, x):
        if x[0] == 0:
            z = [x[1], x[1]^x[2]]
        else:
            z = [x[2], x[2]^x[1]]
        y = [z[1], z[0]^z[1]]
        return y

    def get_samples(self):
        train_x = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]]
        test_x = [
            [0, 1, 0],
            [1, 0, 1]]
        
        train_y = [self.get_output(x) for x in train_x]
        test_y = [self.get_output(x) for x in test_x]
        return [train_x, train_y], [test_x, test_y]

    def get_train_samples(self):
        return self.train_samples

    def get_test_samples(self):
        return self.test_samples



if __name__ == '__main__':
    d = Dataset()
    tr = d.get_train_samples()
    for x, y in zip(*tr):
        print(x, y)

    ts = d.get_test_samples()
    for x, y in zip(*ts):
        print(x, y)

