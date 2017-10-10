import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from logistic_map import LogisticMapGenerator, next_sample

RATE = 2.5
SEQUENCE_LENGTH = 3
LEARNING_RATE = 1E-3
BATCH_SIZE = 128

def f_x(x, w1, w2, w3):
    h1 = tf.nn.relu(tf.matmul(x, w1))
    h2 = tf.nn.relu(tf.matmul(h1, w2))
    h3 = tf.matmul(h2, w3)
    return h3

# def f_x(x, rate):
#     return rate * x * (1. - x)

def main():
    generator = LogisticMapGenerator(RATE, SEQUENCE_LENGTH)
    x = tf.placeholder(tf.float32, shape=[None, 1])
    y = tf.placeholder(tf.float32, shape=[None, SEQUENCE_LENGTH])
    w1 = tf.get_variable("w1", [1, 16], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable("w2", [16, 16], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable("w3", (16, 1), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    rate = tf.get_variable("rate", [1], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('memoryless_static_rnn') as scope:
        cells = []
        cells.append(x)
        for idx, _ in enumerate(range(SEQUENCE_LENGTH - 1)):
            if idx == 0:
                # cell = f_x(x, rate)
                cell = f_x(x, w1, w2, w3)
            else:
                # cell = f_x(cell, rate)
                cell = f_x(cell, w1, w2, w3)
            cells.append(cell)
            scope.reuse_variables()

    cells = tf.stack(cells, axis=1)
    loss = tf.reduce_mean(tf.square(tf.squeeze(tf.stack(cells)) - y))
    training_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        # TRAIN
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            _input, _sequence = generator.sample(BATCH_SIZE)
            # _, loss_val, rate_val = sess.run([training_step, loss, rate], feed_dict={x: _input, y: _sequence})
            _, loss_val = sess.run([training_step, loss], feed_dict={x: _input, y: _sequence})
            if i % 200 == 0:
                # print('loss: {} - rate: {}\n'.format(loss_val, rate_val))
                print('loss: {}\n'.format(loss_val))

        # TEST
        value = 0.8
        _input_test = np.array([value])[None, :]
        output_rnn = sess.run(cells, feed_dict={x: _input_test})
        output_lmap = next_sample(value, RATE, SEQUENCE_LENGTH)

        plt.plot(np.squeeze(output_rnn))
        plt.plot(np.squeeze(output_lmap))
        plt.legend(['rnn output', 'logistic map output'], loc='lower right')
        # print('Estimanted rate: {} - Real rate: {}'.format(rate.eval(), RATE))
        plt.show()


if __name__ == '__main__':
    main()
