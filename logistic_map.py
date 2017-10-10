import numpy as np

# Logistic Map Difference Equation
def next_gen(x, rate):
    return rate * x * (1. - x)

def next_sample(x, rate, sequence_length):
    sample = []
    for _ in range(sequence_length):
        sample.append(x)
        x = next_gen(x, rate)
    return np.asarray(sample)

class LogisticMapGenerator(object):
    def __init__(self, rate, sequence_length):
        # self.rate = np.random.uniform() * 4
        self.rate = rate
        self.sequence_length = sequence_length

    def sample(self, batch_size):
        batch = []
        x_s = []
        for _ in range(batch_size):
            x = np.random.uniform()
            batch.append(next_sample(x, self.rate, self.sequence_length))
            x_s.append(x)
        return np.asarray(x_s)[:, None], np.asarray(batch)

if __name__ == '__main__':
    print(next_sample(0.1, 2.5, 10))
