import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from logistic_map import LogisticMapGenerator, next_sample

RATE = 3.7
TRAIN_SEQUENCE_LENGTH = 3
TEST_SEQUENCE_LENGTH = 50
LEARNING_RATE = 1E-2
BATCH_SIZE = 128
ORDER = 4
EPOCHS = 50000

def f_u(u, w1, order):
	"""
	Build nth order polynomial
 	W1 * x + W2 * x^2 ... + Wn * x^n
	"""
	x_p = []
	for p in range(order):
		x_p.append(tf.pow(u, p+1))
		x = tf.squeeze(tf.stack(x_p, axis=1))
	return tf.matmul(x, w1)

def get_rnn_outputs(sess, cells, u, starting_value):
	"""
	Run inference for sequence_length steps given a starting_value
	"""
	total_loops = TEST_SEQUENCE_LENGTH // TRAIN_SEQUENCE_LENGTH + 1
	total_output_rnn = []
	for i in range(total_loops):
		if i == 0:
			output_rnn = sess.run(cells, feed_dict={u: np.asarray([[starting_value], [starting_value]])})
			output_rnn = np.squeeze(output_rnn[0,:])
			last_output = output_rnn[-2]
		else:
			output_rnn = sess.run(cells, feed_dict={u: np.asarray([[last_output], [last_output]])})
			output_rnn = np.squeeze(output_rnn[0,:])
			last_output = output_rnn[-2]
		total_output_rnn.extend(list(output_rnn))
	return np.squeeze(np.asarray(total_output_rnn)[:TEST_SEQUENCE_LENGTH])

def main():
	generator = LogisticMapGenerator(RATE, TRAIN_SEQUENCE_LENGTH)
	u = tf.placeholder(tf.float32, shape=[None, 1])
	y = tf.placeholder(tf.float32, shape=[None, TRAIN_SEQUENCE_LENGTH])
	w1 = tf.get_variable("w1", [ORDER, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

	with tf.variable_scope('memoryless_static_rnn') as scope:
		cells = []
		cell = u
		cells.append(cell)
		for idx, _ in enumerate(range(TRAIN_SEQUENCE_LENGTH - 1)):
			cell = f_u(cell, w1, ORDER)
			cells.append(cell)
			scope.reuse_variables()

	cells = tf.stack(cells, axis=1)
	loss = tf.reduce_mean(tf.square(tf.squeeze(tf.stack(cells)) - y))
	training_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

	with tf.Session() as sess:
		"""TRAIN"""
		sess.run(tf.global_variables_initializer())
		for epoch in range(EPOCHS):
			_input, _sequence = generator.sample(BATCH_SIZE)
			_, loss_val = sess.run([training_step, loss], feed_dict={u: _input, y: _sequence})
			if epoch % 2000 == 0:
				print('Epoch: {} = Loss: {}'.format(epoch, loss_val))
				print('Polynomial coefficients: {}\n'.format(np.squeeze(w1.eval())))

		"""TEST FOR A LONGER SEQUENCE"""
		starting_value = 0.5
		output_lmap = np.squeeze(next_sample(starting_value, RATE, TEST_SEQUENCE_LENGTH))
		output_rnn = get_rnn_outputs(sess, cells, u, starting_value)
		MSE = np.mean(np.square(output_lmap - output_rnn))
		print('Mean Squared error: {:.4f}'.format(MSE))

		"""PLOT TEST RESULTS"""
		plt.plot(output_rnn)
		plt.plot(output_lmap)
		plt.legend(['rnn output', 'logistic map output'], loc='lower right')
		plt.show()

if __name__ == '__main__':
	main()
