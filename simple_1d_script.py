import numpy as np
from numpy import linalg as LA
import tensorflow as tf
# import matplotlib.pyplot as plt
import itertools


def nn_run(k_squared, encoder_init_weights, decoder_init_weights,
	learning_rates, optimizers, encoder_activations, decoder_activations, init_weights_function, 
	init_bias_function, num_units_list, m, train_batch_size, mc_batch_size, num_epochs, x_stddev): 
	'''
	k_squared: k_squared value for cost function
	encoder_init_weights: a list of initial weights for encoder. 
	decoder_init_weights: a list of initial weights for decoder. 
	learning_rates: list of two learning rates (for encoder and decoder)
	optimizers: list of two optimizers (for encoder and decoder)
	encoder_activations: a list of activation functions (variable length)
	decoder_activations: a list of activation functions (variable length)
	init_weights_function: Weight initialization function. 
	init_bias_function: Bias initialization function
	num_units_list: list of unit numbers, of the form [m, ..., m] where m is dimension.
	m: Dimension of input/output 
	train_batch_size: Number of samples in a training batch 
	mc_batch_size: Number of samples in a Monte Carlo batch (for testing)
	num_epochs: Number of epoch steps to take in training
	x_stddev: The standard deviation of the x0 input 
	'''
	if not(k_squared): 
		k_squared = 0.04
	
	g1 = tf.Graph()

	x0 = tf.placeholder(tf.float32, [None, 1])
	z = tf.placeholder(tf.float32, [None, 1])
	
	num_encoder_layers = len(encoder_activations)
	num_decoder_layers = len(decoder_activations)

	# xavier_init = tf.glorot_uniform_initializer()

	#declare encoder
	encoder_params = []
	for i in range(num_encoder_layers): 
		w_name = 'W' + str(i + 1)
		b_name = 'b' + str(i + 1)
		if encoder_init_weights: 
			assert len(encoder_init_weights) == 2 * num_encoder_layers, 'Wrong number of initial weights!'
			init_weight, init_bias = encoder_init_weights[2 * i], encoder_init_weights[1 + (2 * i)]
			encoder_params.append(tf.Variable(initial_value=init_weight, name=w_name))
			encoder_params.append(tf.Variable(initial_value=init_bias, name=b_name))
		else: 
			fan_in, fan_out = num_units_list[i], num_units_list[i + 1]
			encoder_params.append(tf.Variable(init_weights_function([fan_in, fan_out]), name=w_name))
			encoder_params.append(tf.Variable(init_bias_function([fan_out]), name=b_name))

	#declare decoder
	decoder_params = []
	for j in range(num_decoder_layers): 
		total_index = j + num_encoder_layers
		w_name = 'W' + str(total_index + 1)
		b_name = 'b' + str(total_index + 1)
		if decoder_init_weights: 
			assert len(decoder_init_weights) == 2 * num_decoder_layers, 'Wrong number of initial weights!'
			init_weight, init_bias = decoder_init_weights[2 * j], decoder_init_weights[1 + (2 * j)]
			decoder_params.append(tf.Variable(initial_value=init_weight, name=w_name))
			decoder_params.append(tf.Variable(initial_value=init_bias, name=b_name))
		else: 
			fan_in, fan_out = num_units_list[total_index], num_units_list[total_index + 1]
			decoder_params.append(tf.Variable(init_weights_function([fan_in, fan_out]), name=w_name))
			decoder_params.append(tf.Variable(init_bias_function([fan_out]), name=b_name))

	#Encoder forward pass 
	current_hidden = x0 
	for i in range(num_encoder_layers): 
		current_weight, current_bias = encoder_params[2 * i], encoder_params[1 + (2 * i)]
		affine_forward = tf.add(tf.matmul(current_hidden, current_weight), current_bias)
		current_hidden = encoder_activations[i](affine_forward)

	u1 = current_hidden
	u1_cost = k_squared * tf.reduce_mean(tf.reduce_sum((u1)**2, axis=1))

	x1 = u1 + x0
	x1_noise = x1 + z
	current_hidden = x1_noise
	#Decoder foward pass 
	for j in range(num_decoder_layers): 
		current_weight, current_bias = decoder_params[2 * j], decoder_params[1 + (2 * j)]
		affine_forward = tf.add(tf.matmul(current_hidden, current_weight), current_bias)
		current_hidden = decoder_activations[i](affine_forward)

	u2 = current_hidden
	x2 = x1 - u2
	x2_cost = tf.reduce_mean(tf.reduce_sum((x2)**2, axis=1))

	wits_cost = x2_cost + u1_cost

	# Define gradients and optimizers 
	encoder_lr, decoder_lr = learning_rates[0], learning_rates[1]

	encoder_opt = optimizers[0](learning_rate = encoder_lr)
	decoder_opt = optimizers[1](learning_rate = decoder_lr)

	grads = tf.gradients(wits_cost, encoder_params + decoder_params)
	grads1 = grads[:len(encoder_params)]
	grads2 = grads[len(encoder_params):]

	train_op1 = encoder_opt.apply_gradients(zip(grads1, encoder_params))
	train_op2 = decoder_opt.apply_gradients(zip(grads2, decoder_params))
	train_op = tf.group(train_op1, train_op2)

	# Training here. 

	mc_x_batch = np.random.normal(size=(mc_batch_size, m), scale = x_stddev)
	mc_z_batch = np.random.normal(size=(mc_batch_size, m), scale = 1.0)
	mc_losses = []

	epoch_step = int(num_epochs/20)

	print('Beginning Training....')
	print('Training Batch Size: {}, MC Batch Size: {}'.format(train_batch_size, mc_batch_size))

	#declare testing stuf
	num_x0_points = 100
	test_averaging = 50
	x0_test = np.linspace(-3 * x_stddev, 3 * x_stddev, num=num_x0_points)
	z_test = np.random.normal(scale=1, size=num_x0_points)
	u1_test, u2_test, y2_test = np.zeros((1, num_x0_points)), np.zeros((1, num_x0_points)), np.zeros((1, num_x0_points))

	with tf.Session() as sess: 
		sess.run(tf.global_variables_initializer())
		for epoch in range(num_epochs): 
			x_batch = np.random.normal(size=(train_batch_size, m), scale = x_stddev)
			z_batch = np.random.normal(size=(train_batch_size, m), scale = 1.0)

			_, train_cost = sess.run([train_op, wits_cost], feed_dict = {x0: x_batch, z: z_batch})

		
			if epoch % epoch_step == 0: 
				mc_cost = sess.run([wits_cost], feed_dict={x0: mc_x_batch, z: mc_z_batch})
				print('Epoch {}, Cost {}, MC Cost: {}'.format(epoch, train_cost, mc_cost[0]))
		final_mc_cost = mc_cost[0]
		print('Epoch {}, Cost {}, MC Cost: {}'.format(epoch, train_cost, final_mc_cost))

		

		# print('Beginning testing....')
		# for i in range(num_x0_points):
		# 	u1t, u2t, y2t  = 0, 0, 0

		# 	#vignesh says: don't pass in y2 values
		# 	for _ in range(test_averaging):
		# 		u1tmp, u2tmp, y2tmp, x1tmp = sess.run(
		# 			[u1, u2, x1_noise, x1],
		# 			feed_dict={x0: x0_test[i].reshape((1, 1)),
		# 			z: np.array(np.random.normal(scale=1)).reshape((1, 1))}) #generate z on the fly.
		# 		u1t += u1tmp
		# 		u2t += u2tmp
		# 		y2t += y2tmp

		# 	u1_test[0, i] = u1t / test_averaging
		# 	u2_test[0, i] = u2t / test_averaging
		# 	y2_test[0, i] = y2t / test_averaging

	return final_mc_cost
					


def cartesian_product(*arrays): 
  return itertools.product(*arrays)


if __name__ == "__main__":
	k_squared_vals = [0.04]
	encoder_init_weights_lists = [[]]
	decoder_init_weights_lists = [[]]
	learning_rates_lists = [[5e-4, 5e-4], [5e-4, 0]]
	optimizers_lists = [[tf.train.GradientDescentOptimizer, tf.train.GradientDescentOptimizer]]
	encoder_activations_lists = [[tf.nn.sigmoid, tf.identity], [tf.nn.tanh, tf.identity]]
	decoder_activations_lists = [[tf.nn.sigmoid, tf.identity]]
	init_weights_functions = [tf.glorot_normal_initializer()]
	init_bias_functions = [tf.zeros_initializer()]
	num_units_lists = [[1, 10, 1, 10, 1]]
	m_list = [1]
	train_batch_sizes = [1000]
	mc_batch_sizes = [1000]
	num_epochs_list = [2000]
	x_stddev_list = [5]
	
	# train_net(200, 500, 100, 5)

	all_hyperparam_tuples = cartesian_product(k_squared_vals,
		encoder_init_weights_lists,
		decoder_init_weights_lists,
		learning_rates_lists,
		optimizers_lists,
		encoder_activations_lists,
		decoder_activations_lists,
		init_weights_functions,
		init_bias_functions,
		num_units_lists,
		m_list,
		train_batch_sizes,
		mc_batch_sizes,
		num_epochs_list,
		x_stddev_list)

	run_num = 1
	# seed = 85

	good_seeds = []
	good_losses = []
	for tup in all_hyperparam_tuples: 
		#Unroll the huge tuple of hyperparameters.
		k_squared, encoder_init_weights, decoder_init_weights, learning_rates, optimizers, encoder_activations, decoder_activations, init_weights_function, init_bias_function, num_units_list, m, train_batch_size, mc_batch_size, num_epochs, x_stddev = tup

		#Seed for reproducibility. Change seed every time. 
		# do we need a seed for the random seed generator? whoa...
		seed = np.random.randint(low=5, high=200)
		np.random.seed(seed)
		tf.set_random_seed(seed)

		print('RUN NUMBER {}'.format(run_num))
		print('Numpy/TF random seed {}'.format(seed)) #prints the seed here. 
		print('HYPERPARAMETERS ARE: ')
		print('k_squared, encoder_init_weights, decoder_init_weights,' +
			'learning_rates, optimizers, encoder_activations, decoder_activations, init_weights_function, ' + 
			'init_bias_function, num_units_list, m, train_batch_size, mc_batch_size, num_epochs, x_stddev')
		print(tup)
		print('-----------------------------------------------\n')
		final_cost = nn_run(k_squared, encoder_init_weights, decoder_init_weights,
			learning_rates, optimizers, encoder_activations, decoder_activations, init_weights_function, 
			init_bias_function, num_units_list, m, train_batch_size, mc_batch_size, num_epochs, x_stddev)
		print('-----------------------------------------------\n')
		run_num += 1