import tensorflow as tf
import numpy as np
import uuid
import sys
import scipy.stats 
import matplotlib
import itertools 
matplotlib.use('Agg')

import matplotlib.pyplot as plt

#encoder_activation_1 <- activation_fn_1
##encoder_activation_2 (new)
#decoder_activation_1 <- activation_fn_2
##encoder_activation_2 (new)

def neural_net_run(m, k_sq, learning_rate, epochs, batch_size, x_stddev, 
  encoder_activation_1, encoder_activation_2, decoder_activation_1, decoder_activation_2, 
  num_units_1, num_units_2, decay, test_averaging, optimizer_func):
  '''
  A single run of the neural network. Used for hyerparameter search.
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
  '''
  x0 = tf.placeholder(tf.float32, [None, m])
  z = tf.placeholder(tf.float32, [None, m])

  l1 = tf.layers.dense(
    x0, num_units_1, activation=encoder_activation_1, use_bias=True)
  l2 = tf.layers.dense(
    l1, m, activation=encoder_activation_2, use_bias=True)

  u1 = l2
  u1_cost = (tf.norm(u1)**2) / batch_size

  # The observed value for the second controller is the original controlled with noise
  x1 = x0 + u1
  y1 = tf.placeholder_with_default(x1 + z, [None, m])

  l3 = tf.layers.dense(
    y1, num_units_2, activation=decoder_activation_1, use_bias=True)
  l4 = tf.layers.dense(
    l3, m, activation=decoder_activation_2, use_bias=True)

  u2 = -l4
  x2 = x1 + u2

  u2_cost = (tf.norm(x2) ** 2) / batch_size
  wits_cost = (k_sq * u1_cost) + u2_cost

  adaptive_learning_rate = tf.placeholder_with_default(learning_rate, [])

  if optimizer_func == tf.train.AdamOptimizer: 
    optimizer = tf.train.AdamOptimizer(adaptive_learning_rate, epsilon=1e-4).minimize(wits_cost)
  else:
    optimizer = optimizer_func(adaptive_learning_rate).minimize(wits_cost)
  # optimizer = tf.train.GradientDescentOptimizer(learning_rate=adaptive_learning_rate).minimize(wits_cost)

  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
      sess.run(init_op)
      x_train = np.random.normal(size=epochs * batch_size * m, scale=x_stddev)

    # Train for some epochs
      for step in range(epochs):
          x_batch = x_train[step: step + (batch_size * m)].reshape((batch_size, m))

          # Noise has variance 1
          z_batch = np.random.normal(size=(batch_size, m), scale=1)

          #NOTE: CHANGING DECAY SCHEDULE HERE
          # current_lr = learning_rate/(1 + (1 - decay)*step)
          #before
          #current_lr = learning_rate * (decay**step)
          # if np.random.randint(int(epochs/5)) == 25:
          #   #randomly perturb by taking a wildly large step. 
          #   print('RANDOM PERTURBATION AT STEP: {}'.format(step))
          #   current_lr = 0.5
          # else: 
          current_lr = learning_rate
          _, val = sess.run(
          [optimizer, wits_cost],
          feed_dict={x0: x_batch, z: z_batch,
               adaptive_learning_rate: current_lr})

          if step % 100 == 0:
              print("step: {}, loss: {}, lr: {}".format(step, val, current_lr))

    # Test over a continuous range of X
      num_x0_points = 300
      x0_test = np.linspace(-3*x_stddev, 3*x_stddev, num=300)
      y1_test = x0_test
      u1_test, u2_test, x1_test, wits_cost_test = np.zeros((1, 300)), np.zeros((1, 300)), np.zeros((1, 300)), np.zeros((1, 300))
      wits_cost_total = 0.0 
      x0_distribution = scipy.stats.norm(loc=0.0, scale=x_stddev)
      total_density = 0.0
      for i in range(300):
          u1t, u2t, x1t, wits_cost_t  = 0, 0, 0, 0
          # wits_cost_t  = 0
          for _ in range(test_averaging):
            u1tmp, u2tmp, x1tmp, wits_cost_tmp = sess.run(
                  [u1, u2, x1, wits_cost],
                  feed_dict={x0: x0_test[i].reshape((1, 1)),
                  z: np.random.normal(size=(1, 1), scale=1)
            })
            wits_cost_t += wits_cost_tmp
            u1t += u1tmp
            u2t += u2tmp
            x1t += x1tmp
          x0_density = x0_distribution.pdf(x0_test[i])
          total_density += x0_density
          scaled_wits_cost = wits_cost_t * x0_density
          wits_cost_test[0, i] = scaled_wits_cost / test_averaging
          u1_test[0, i] = u1t / test_averaging
          u2_test[0, i] = -u2t / test_averaging
          x1_test[0, i] = x1t / test_averaging
      total_cost = np.sum(wits_cost_test)
      print('Mean loss is {}'.format(total_cost / total_density))

      #PLOTTING. Unnecessary for now because we're just doing hyperparameter search
      #TODO fix activation function names
      # l1, = plt.plot(x0_test, u1_test[0], label="U1 Test")
      # plt.legend(handles=[l1])
      # plt.title("{} Unit NN With Activation Fn {}".format(
      #   str(num_units_1), str(encoder_activation_1)))
      # plt.savefig("figs/{}_{}_{}_u_1_{}.png".format(
      #   str(learning_rate), str(num_units_1), str(num_units_2) ,str(k_sq)))

      # plt.clf()
      # l1, = plt.plot(y1_test, u2_test[0], label="U2 Test")
      # plt.legend(handles=[l1])
      # plt.title("{} Unit NN With Activation Fn {}".format(
      #   str(num_units_2), str(decoder_activation_1)))
      # plt.savefig("figs/{}_{}_{}_u_2_{}.png".format(
      #   str(learning_rate), str(num_units_1), str(num_units_2) ,str(k_sq)))

      # plt.clf()
      # l2, = plt.plot(x0_test, x1_test[0], label="X1 Test")
      # plt.title("{} Unit NN With Activation Fn {}".format(
      #   str(num_units_1), str(decoder_activation_2)))
      # plt.legend(handles=[l2])
      # plt.savefig("figs/{}_{}_{}_x_1_{}.png".format(
      #   str(learning_rate), str(num_units_1), str(num_units_2) ,str(k_sq)))


def cartesian_product(*arrays): 
  return itertools.product(*arrays)

if __name__ == "__main__":
  #learning_rates = [0.01, 0.001, 0.0001, 0.005]
  
  diemsions = [1]
  k_squared_vals = [0.5]
  learning_rates = [5e-5]
  num_epochs = [500]
  batch_size = [1000]
  x_stddeviations = [5]
  encoder_activation_1s = [tf.nn.sigmoid]
  encoder_activation_2s = [tf.identity]
  decoder_activation_1s = [tf.nn.sigmoid]
  decoder_activation_2s = [tf.identity]
  num_units_1s = [150, 250]
  num_units_2s = [30, 100, 250]
  decay_rates = [1 - 1e-3]
  test_average_sizes = [100]
  optimizers = [tf.train.AdamOptimizer]

  
  run_num = 1
  
  
  

  all_hyperparam_tuples = cartesian_product(diemsions, k_squared_vals,
    learning_rates, num_epochs, batch_size, x_stddeviations, encoder_activation_1s,
    encoder_activation_2s, decoder_activation_1s, decoder_activation_2s, num_units_1s, 
    num_units_2s, decay_rates, test_average_sizes, optimizers)
  
  for tup in all_hyperparam_tuples: 
    print(tup)

  #   def neural_net_run(m, k_sq, learning_rate, epochs, batch_size, x_stddev, 
  # encoder_activation_1, encoder_activation_2, decoder_activation_1, decoder_activation_2, 
  # num_units_1, num_units_2, decay, test_averaging, optimizer_func):

  #TODO replace nested for loops with better function
  # for lr in learning_rates:
  #   for decay_rate in decay_rates:
  #     for n_units_1 in num_units_1s:
  #       for n_units_2 in num_units_2s:
  #         for activation_fn_1 in first_activations: 
  #           np.random.seed(run_num)
  #           print('RUN NUMBER {}'.format(run_num))
  #           print('Numpy random seed {}'.format(run_num))
  #           print('Initial learning rate {}'.format(lr))
  #           print('GD Optimizier {}'.format(optimizer))
  #           print('Decay rate: {} is UNUSED'.format(decay_rate))
  #           print('N_units_1: {}'.format(n_units_1))
  #           print('N_units_2: {}'.format(n_units_2))
  #           print('Activation_fn_1: {}, activation_fn_2: {}'.format(activation_fn_1, activation_fn_2))
  #           print('k_squared: {}'.format(k_squared))
  #           print('x0 standard deviation: {}'.format(x0_stddev))
  #           print('Number of epochs: {}'.format(num_epochs))
  #           print('-----------------------------------------------\n')
  #           neural_net_run(m = 1, k_sq = k_squared, learning_rate = lr, epochs = num_epochs, batch_size = 1000, 
  #             x_stddev = x0_stddev, activation_fn_1 = activation_fn_1, activation_fn_2 = activation_fn_2, num_units_1 = n_units_1, 
  #             num_units_2 = n_units_2, decay = decay_rate, test_averaging = 10000, optimizer_func = optimizer)
  #           print('-----------------------------------------------\n')
  #           run_num += 1
