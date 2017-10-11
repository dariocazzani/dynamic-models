import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from logistic_map import LogisticMapGenerator, next_sample

RATE = 2.5
SEQUENCE_LENGTH = 5
LEARNING_RATE = 1E-4
BATCH_SIZE = 128
ORDER = 4
APPROXIMATION_EXPONENT = 3

"""
Approximate f_u --> Universal approximation theorem
https://en.wikipedia.org/wiki/Universal_approximation_theorem
a feed-forward network with a single hidden layer containing a finite number
of neurons (i.e., a multilayer perceptron), can approximate continuous
functions on compact subsets of R^n
"""
# Assuming order APPROXIMATION_EXPONENT is enough parameters
def approx_f_u(u, w1_approx, w2_approx, order):
	x_p = []
	for p in range(ORDER):#np.power(order, APPROXIMATION_EXPONENT)):
		x_p.append(u)
		x = tf.squeeze(tf.stack(x_p, axis=1))
	h1 = tf.nn.sigmoid(tf.matmul(x, w1_approx))
	return tf.matmul(h1, w2_approx)

def f_u(u, w1, order):
	# build nth order polynomial
	x_p = []
	for p in range(order):
		x_p.append(tf.pow(u, p+1))
		x = tf.squeeze(tf.stack(x_p, axis=1))
	return tf.matmul(x, w1)

def main():
	generator = LogisticMapGenerator(RATE, SEQUENCE_LENGTH)
	u = tf.placeholder(tf.float32, shape=[None, 1])
	y = tf.placeholder(tf.float32, shape=[None, SEQUENCE_LENGTH])
	w1 = tf.get_variable("w1", [ORDER, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
	w1_approx = tf.get_variable("w1_approx", [ORDER, ORDER**APPROXIMATION_EXPONENT], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
	w2_approx = tf.get_variable("w2_approx", [ORDER**APPROXIMATION_EXPONENT, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

	with tf.variable_scope('memoryless_static_rnn') as scope:
		cells = []
		cells.append(u)
		for idx, _ in enumerate(range(SEQUENCE_LENGTH - 1)):
			if idx == 0:
				# cell = f_u(u, w1, ORDER)
				cell = approx_f_u(u, w1_approx, w2_approx, ORDER)
			else:
				# cell = f_u(cell, w1, ORDER)
				cell = approx_f_u(cell, w1_approx, w2_approx, ORDER)
			cells.append(cell)
			scope.reuse_variables()

	cells = tf.stack(cells, axis=1)
	loss = tf.reduce_mean(tf.square(tf.squeeze(tf.stack(cells)) - y))
	training_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

	with tf.Session() as sess:
		# TRAIN
		sess.run(tf.global_variables_initializer())
		for i in range(20000):
			_input, _sequence = generator.sample(BATCH_SIZE)
			# _, loss_val, rate_val = sess.run([training_step, loss, rate], feed_dict={u: _input, y: _sequence})
			_, loss_val = sess.run([training_step, loss], feed_dict={u: _input, y: _sequence})
			if i % 200 == 0:
				# print('loss: {} - rate: {}\n'.format(loss_val, rate_val))
				print('loss: {}\n'.format(loss_val))

		# TEST
		value = 0.5
		output_rnn = sess.run(cells, feed_dict={u: np.asarray([[value], [value]])})
		output_lmap = next_sample(value, RATE, SEQUENCE_LENGTH)

		plt.plot(np.squeeze(output_rnn[0,:]))
		plt.plot(np.squeeze(output_lmap))
		plt.legend(['rnn output', 'logistic map output'], loc='lower right')
		# print('Estimanted rate: {} - Real rate: {}'.format(rate.eval(), RATE))
		plt.show()


if __name__ == '__main__':
	main()
