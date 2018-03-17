import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import matplotlib.pyplot as plt

def stack_weights(encoder_vars, decoder_vars): 
    '''
    Unpacks two lists of weights (encoder_vars, decoder_vars)
    and stacks them horizontally into a giant vector. 

    Helper function for weight_norm_update. 
    '''
    for i in range(len(encoder_vars)): 
        encoder_vars[i] = np.ndarray.flatten(encoder_vars[i])
        decoder_vars[i] = np.ndarray.flatten(decoder_vars[i])
    W1, b1, W2, b2 = encoder_vars[0], encoder_vars[1], encoder_vars[2], encoder_vars[3]
    W3, b3, W4, b4 = decoder_vars[0], decoder_vars[1], decoder_vars[2], decoder_vars[3]
    return np.hstack((W1, b1, W2, b2, W3, b3, W4, b4))


def weight_norm_update(weight_stack_1, weight_stack_2): 
	'''
	Computes the update l1, l2 norms for two vectors 
	(weight_stack_1, weight_stack_2).
	'''
    weight_diff = weight_stack_1 - weight_stack_2
    l2_norm = LA.norm(weight_diff) / LA.norm(weight_stack_1)
    l1_norm = np.sum(np.abs(weight_diff)) / np.sum(np.abs(weight_stack_1))
    return l1_norm, l2_norm

# def neural_net_run(m, k_sq, learning_rate, epochs, batch_size, x_stddev, 
#   encoder_activation_1, encoder_activation_2, decoder_activation_1, decoder_activation_2, 
#   num_units_1, num_units_2, decay, test_averaging, optimizer_func, skip_layer):
  '''
  A single run of the decoder network. Assume a fixed encoder which performs a piecewise
  constant function. 

  m = Dimensions
  k_sq = k_squared value for loss function
  learning_rate = constant LR. 
  epochs = number of epochs 
  batch_size = batch_size for NN training
  
  num_units_1 = number of units in hidden layer 1 
  num_units_2 = number of units in hidden layer 3 
  decay = learning rate decay 
  test_averaging = number of steps over which to average u1, u2, x1
  optimizer_func = optimizer from tensorflow
  skip_layer = A boolean which indicates whether the last layer sees a residual or not. 
  '''

def create_computational_graph(k_squared = 0.04, encoder_init_weights, decoder_init_weights,
	learning_rates, optimizers, encoder_activations, decoder_activations, init_weights_function, 
	init_bias_function, num_units_list): 
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
	num_units_list: list of unit numbers, of the form [m, ..., m]
	'''
	k_squared = 0.04
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
			encoder_params.append(tf.Variable(initial_value=init_weights_function([fan_in, fan_out]), name=w_name))
			encoder_params.append(tf.Variable(initial_value=init_bias_function([fan_out]), name=b_name))

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
			decoder_params.append(tf.Variable(initial_value=init_weights_function([fan_in, fan_out]), name=w_name))
			decoder_params.append(tf.Variable(initial_value=init_bias_function([fan_out]), name=b_name))

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

def train_net(k_squared = 0.04, encoder_init_weights, train_batch_size, mc_batch_size, num_epochs, x_stddev):
	create_computational_graph(k_squared, encoder_init_weights)

	all_u1 = []
	all_x2 = []
	all_u2 = []
	all_y2 = []
	l1_weight_updates = []
	l2_weight_updates = []

	train_cost = []
	weights_dict = {}
	prev_weights = np.zeros(shape=(1051,))

	mc_x_batch = np.random.normal(size=(mc_batch_size, 1), scale = x_stddev)
	mc_z_batch = np.random.normal(size=(mc_batch_size, 1), scale = 1.0)
	mc_losses = []

	epoch_step = int(num_epochs/50)

	print('Beginning Training....')
	print('Training Batch Size: {}, MC Batch Size: {}'.format(train_batch_size, mc_batch_size))

	with tf.Session() as sess: 
	    sess.run(tf.global_variables_initializer())
	    #training
	    for epoch in range(num_epochs): 
	        x_batch = np.random.normal(size=(train_batch_size, 1), scale = x_stddev)
	        z_batch = np.random.normal(size=(train_batch_size, 1), scale = 1.0)

	        _, train_cost = sess.run([train_op, wits_cost], feed_dict = {x0: x_batch, z: z_batch})

	        #Uncomment this when interested in weight norms. 
	        # _, cost, encoder_vars_tmp, decoder_vars_tmp  = sess.run([train_op, wits_cost, encoder_vars, decoder_vars], 
	        #                                                         feed_dict={x0: x_batch, z: z_batch})
	        train_cost.append(cost)
	        
	        mc_cost = sess.run([wits_cost], feed_dict={x0: mc_x_batch, z: mc_z_batch})
	        mc_losses.append(mc_cost[0])
	        
	        #Uncomment this when interested in weight norms. 
	        # next_weights = stack_weights(encoder_vars_tmp, decoder_vars_tmp)
	        # l1_update, l2_update = weight_norm_update(prev_weights, next_weights)
	        # l1_weight_updates.append(l1_update)
	        # l2_weight_updates.append(l2_update)
	        # prev_weights = next_weights

	        if epoch % epoch_step == 0: 
	            print('Epoch {}, Cost {}, MC Cost: {}'.format(epoch, cost, mc_cost[0]))