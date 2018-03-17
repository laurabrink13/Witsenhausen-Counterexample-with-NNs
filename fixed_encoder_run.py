import tensorflow as tf
import numpy as np
import uuid
import sys
import scipy.stats 
import matplotlib
import itertools 
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def neural_net_run(m, k_sq, learning_rate, epochs, batch_size, x_stddev, 
  encoder_activation_1, encoder_activation_2, decoder_activation_1, decoder_activation_2, 
  num_units_1, num_units_2, decay, test_averaging, optimizer_func, skip_layer):
  '''
  A single run of the decoder network. Assume a fixed encoder which performs a piecewise
  constant function. 

  m = Dimensions
  k_sq = k_squared value for loss function
  learning_rate = constant LR. 
  epochs = number of epochs 
  batch_size = batch_size for NN training
  x_stddev = standard deviation of x_0 
  encoder_activation_1 = activation function for layer 1
  encoder_activation_2 = activation function for layer 2
  decoder_activation_1 = activation function for layer 3
  decoder_activation_2 = activation function for layer 4
  num_units_1 = number of units in hidden layer 1 
  num_units_2 = number of units in hidden layer 3 
  decay = learning rate decay 
  test_averaging = number of steps over which to average u1, u2, x1
  optimizer_func = optimizer from tensorflow
  skip_layer = A boolean which indicates whether the last layer sees a residual or not. 
  '''

  #x1 is a placeholder because it is deterministic. We 
  #also pass in x0 in order to calculate u1. 
  

  x0 = tf.placeholder(tf.float32, [None, m])
  x1 = tf.placeholder(tf.float32, [None, m])
  z = tf.placeholder(tf.float32, [None, m])

  # f1 = lambda: tf.constant(-10)
  # f2 = lambda: tf.constant(-4)
  # f3 = lambda: tf.constant(-4)
  # f4 = lambda: tf.constant(10)

  # pw_step_func = tf.case({tf.less(x, -7): f1, tf.less(x, 0): f2, 
  #   tf.less(x, 7): f3, tf.greater(x, 7): f4}, 
  #   exclusive=True)

  # x1 = x0 #TODO apply piecewise linear function 
  
  u1 = x1 - x0 
  # u1_cost = tf.reduce_mean(tf.pow(tf.norm(u1,axis=1),2))

  u1_cost = tf.reduce_mean(tf.reduce_sum((u1)**2, axis=1))
  # u1_cost = (tf.norm(u1)**2) / batch_size #todo change this? not sure which is correct


  # The observed value for the second controller is the original controlled with noise
  y2 = tf.placeholder_with_default(x1 + z, [None, m])

  l3 = tf.layers.dense(
    y2, num_units_2, activation=decoder_activation_1, use_bias=True)
  if skip_layer: 
    l4 = tf.layers.dense(
      l3 + y2, m, activation=decoder_activation_2, use_bias=True) #CHANGED TO RESIDUAL LAYER
  else: 
    l4 = tf.layers.dense(
      l3, m, activation=decoder_activation_2, use_bias=True) #CHANGED TO RESIDUAL LAYER

  u2 = -l4
  x2 = x1 + u2

  # x2_cost = (tf.norm(x2) ** 2) / batch_size
  x2_cost = tf.reduce_mean(tf.reduce_sum((x2)**2, axis=1))

  # x2_cost = tf.reduce_mean(tf.pow(tf.norm(x2,axis=1),2))
  '''
  Note: Because the net has no control over x0, x1, or u1, 
  it doesn't matter whether k_sq * u1 is in the cost.

  However, this does mean the loss numbers are no longer comparable here. 
  Just means 
  '''
  #wits_cost = (k_sq * u1_cost) + x2_cost
  wits_cost = x2_cost
  adaptive_learning_rate = tf.placeholder_with_default(learning_rate, [])

  if optimizer_func == tf.train.AdamOptimizer: 
    optimizer = tf.train.AdamOptimizer(adaptive_learning_rate, epsilon=1e-4).minimize(wits_cost)
  else:
    optimizer = optimizer_func(adaptive_learning_rate).minimize(wits_cost)
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate=adaptive_learning_rate).minimize(wits_cost)

  init_op = tf.global_variables_initializer()
  print_step = int(epochs/50)

  with tf.Session() as sess:
      sess.run(init_op)
      x_train = np.random.normal(size=epochs * batch_size * m, scale=x_stddev)

    # Train for some epochs
      for step in range(epochs):
          x0_batch = x_train[step: step + (batch_size * m)].reshape((batch_size, m))
          x1_batch = pw_step_function(x0_batch)

          # Noise has variance 1
          z_batch = np.random.normal(size=(batch_size, m), scale=1)

          
          current_lr = learning_rate
          _, val = sess.run(
          [optimizer, wits_cost],
          feed_dict={x0: x0_batch, x1: x1_batch, z: z_batch,
               adaptive_learning_rate: current_lr, 
               #y1: y1_batch
               }) 

          if step % print_step == 0:
              print("step: {}, loss: {}, lr: {}".format(step, val, current_lr))

    # Test over a continuous range of X
    
      num_x0_points = 550
      x0_test = np.linspace(-3 * x_stddev, 3 * x_stddev, num=num_x0_points)
      # x0_test = np.hstack((np.linspace(-3 * x_stddev, -7.1, num=100), 
      #      np.linspace(-7.1, -6.9, num=50), 
      #      np.linspace(-6.9, -0.1, num=100),
      #      np.linspace(-0.1, 0.1, num=50), 
      #      np.linspace(0.1, 6.9, num=100),
      #      np.linspace(6.9, 7.1, num=50), 
      #      np.linspace(7.1, 3*x_stddev, num=100)))
      # np.linspace(-7.1, -6.9, num=100), np.linspace(-0.1, 0.1, num=100), np.linspace(6.9, 7.1, num=100)
      # x0_test = np.linspace(-3*x_stddev, 3*x_stddev, num=num_x0_points)
      y2_test = x0_test
      x1_test = pw_step_function(x0_test)
      z_test = np.random.normal(scale=1, size=num_x0_points)

      u1_test, u2_test, wits_cost_test = np.zeros((1, num_x0_points)), np.zeros((1, num_x0_points)), np.zeros((1, num_x0_points))
      wits_cost_total = 0.0 
      x0_distribution = scipy.stats.norm(loc=0.0, scale=x_stddev)
      total_density = 0.0
      for i in range(num_x0_points):
          u1t, u2t, wits_cost_t  = 0, 0, 0
          
          #vignesh says: don't pass in y2 values
          for _ in range(test_averaging):
            u1tmp, u2tmp, y2tmp,x1tmp, ztmp, wits_cost_tmp = sess.run(
                 [u1, u2, y2, x1,z, wits_cost],
                 feed_dict={x0: x0_test[i].reshape((1, 1)), x1: x1_test[i].reshape((1, 1)),
                 z: np.array(np.random.normal(scale=1)).reshape((1, 1))})
            wits_cost_t += wits_cost_tmp
            u1t += u1tmp
            u2t += u2tmp
            
          x0_density = x0_distribution.pdf(x0_test[i])
          total_density += x0_density
          scaled_wits_cost = wits_cost_t * x0_density
          wits_cost_test[0, i] = scaled_wits_cost / test_averaging
          u1_test[0, i] = u1t / test_averaging
          u2_test[0, i] = -u2t / test_averaging
          #x1_test[0, i] = x1t / test_averaging
      total_cost = np.sum(wits_cost_test) / total_density
      print('Mean loss over {} points is {}'.format(num_x0_points, total_cost))

      # PLOTTING. Unnecessary for now because we're just doing hyperparameter search
      # TODO fix activation function names
      # l1, = plt.plot(x0_test, u1_test[0], label="U1 Test")
      # plt.legend(handles=[l1])
      # plt.title("{} Unit NN With Activation Fns {}, {}".format(
      #   str(num_units_1), str(encoder_activation_1), str(decoder_activation_1)))
      # plt.savefig("figs/y1_tests/x0vsu1_nu1_{}_nu2_{}_ksq_{}_f1_{}_f3_{}.png".format(
      #   str(num_units_1), str(num_units_2) ,str(k_sq), 
      #   str(encoder_activation_1), str(decoder_activation_1)))

      # print('x0 points: {}'.format(x0_test))
      # print('x1 points: {}'.format(x1_test))
      # print('z points: {}'.format(z_test))
      # print('y1 points: {}'.format(x1_test + z_test))
      # print('u2 points: {}'.format(u2_test[0]))
      plt.clf()
      plt.plot(y2_test, u2_test[0], lw=0.5, c='green')
      plt.scatter(y2_test, u2_test[0], s=0.2, c='blue')
      plt.plot([-4, -4], [-10, 10], c='red')
      plt.plot([-1, -1], [-10, 10], c='red')
      plt.plot([2, 2], [-10, 10], c='red')
      plt.plot([5, 5], [-10, 10], c='red')
      
      # plt.legend(handles=[l1])
      plt.title("Y2 vs U2, {} Units, {}, {} Activations".format(
        str(num_units_2), str(decoder_activation_1), str(decoder_activation_2)))
      plt.savefig("figs/fixed_encoder_test/y2vsu2_nu1_{}_nu2_{}_ksq_{}_xstd_{}_f3_{}_f4_{}.png".format(
        str(num_units_1), str(num_units_2) ,str(k_sq), str(x_stddev),
        str(decoder_activation_1), str(decoder_activation_2)))

      # plt.clf()
      # l1, = plt.plot(y1_test, u2_test[0] + x1_test[0], label="X2 Test")
      # plt.legend(handles=[l1])
      # plt.title("{} Unit NN With Activation Fns, {}, {}".format(
      #   str(num_units_2), str(encoder_activation_1), str(decoder_activation_1)))
      # plt.savefig("figs/y1_tests/y1vsx2_nu1_{}_nu2_{}_ksq_{}_f1_{}_f3_{}.png".format(
      #   str(num_units_1), str(num_units_2) ,str(k_sq), 
      #   str(encoder_activation_1), str(decoder_activation_1)))

      # plt.clf()
      # l2, = plt.plot(x0_test, x1_test, label="deterministic x1")
      # plt.title("Deterministic Step Function")
      # plt.legend(handles=[l2])
      # plt.savefig("figs/fixed_encoder_test/x0vsx1_nu1_{}_nu2_{}_ksq_{}_xstd_{}_f3_{}_f4_{}.png".format(
      #   str(num_units_1), str(num_units_2) ,str(k_sq), str(x_stddev), 
      #   str(encoder_activation_1), str(decoder_activation_1)))

def pw_step_function(x_arr): 
  '''
  Performs a piecewise step operation on every element of x_arr.
  x_arr: A numpy 1D array of real numbers of shape (N, )
  returns: A numpy 1D array of shape (N, ) whose values are in [-10, -4, 4, 10]
  '''
  return np.piecewise(x_arr, [x_arr < -7, (x_arr >= -7) & (x_arr < 0), (x_arr >= 0) & (x_arr < 7), x_arr > 7],
    [-4, -1, 2, 5])

def cartesian_product(*arrays): 
  return itertools.product(*arrays)

if __name__ == "__main__":
  #learning_rates = [0.01, 0.001, 0.0001, 0.005]
  
  diemsions = [1]
  k_squared_vals = [0.04]
  learning_rates = [5e-5]
  num_epochs = [8000]
  batch_size = [100]
  x_stddeviations = [5]
  encoder_activation_1s = [tf.nn.relu]
  encoder_activation_2s = [tf.identity]
  decoder_activation_1s = [tf.nn.tanh, tf.nn.sigmoid]
  decoder_activation_2s = [tf.identity]
  num_units_1s = [150]
  num_units_2s = [30]
  decay_rates = [1 - 1e-3]
  test_average_sizes = [100]
  optimizers = [tf.train.AdamOptimizer]
  skip_layers = [True, False]
  
  run_num = 1

  all_hyperparam_tuples = cartesian_product(diemsions, k_squared_vals,
    learning_rates, num_epochs, batch_size, x_stddeviations, encoder_activation_1s,
    encoder_activation_2s, decoder_activation_1s, decoder_activation_2s, num_units_1s, 
    num_units_2s, decay_rates, test_average_sizes, optimizers, skip_layers)


  for tup in all_hyperparam_tuples: 
    #Unroll the huge tuple of hyperparameters.
    #I apologize for the very long line of code. RIP pep8 - Akhil
    m, k_sq, learning_rate, epochs, batch_size, x_stddev, encoder_activation_1, encoder_activation_2, decoder_activation_1, decoder_activation_2, num_units_1, num_units_2, decay, test_averaging, optimizer_func, skip_layer = tup
    seed = run_num + 50
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print('RUN NUMBER {}'.format(run_num))
    print('Numpy/TF random seed {}'.format(seed))
    print('HYPERPARAMETERS ARE: ')
    print('m, k_sq, learning_rate, epochs, batch_size, x_stddev, encoder_activation_1, encoder_activation_2, decoder_activation_1, '
      + 'decoder_activation_2, num_units_1, num_units_2, decay, test_averaging, optimizer_func, skip_layer')
    print(tup)
    print('-----------------------------------------------\n')
    neural_net_run(m, k_sq, learning_rate, epochs, batch_size, x_stddev, 
      encoder_activation_1, encoder_activation_2, decoder_activation_1, decoder_activation_2, 
      num_units_1, num_units_2, decay, test_averaging, optimizer_func, skip_layer)
    print('-----------------------------------------------\n')
    run_num += 1