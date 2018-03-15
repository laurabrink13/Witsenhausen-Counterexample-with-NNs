import tensorflow as tf
import numpy as np
import uuid
import sys
import scipy.stats 
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def neural_net_run(m, k_sq, learning_rate, epochs, batch_size, x_stddev, 
  activation_fn_1, activation_fn_2, num_units_1, num_units_2, 
  decay, test_averaging, optimizer_func):
  '''
  A single run of the neural network. Used for hyerparameter search.
  m = Dimensions
  k_sq = k_squared value for loss function
  learning_rate = constant LR. 
  epochs = number of epochs 
  batch_size = batch_size for NN training
  x_stddev = standard deviation of x_0 
  activation_fn_1 = activation function for layer 1, 2
  activation_fn_2 = activation function for layer 3, 4
  num_units_1 = number of units in hidden layer 1 
  num_units_2 = number of units in hidden layer 3 
  decay = learning rate decay 
  test_averaging = number of steps over which to average u1, u2, x1
  optimizer_func = optimizer from tensorflow
  '''
  x0 = tf.placeholder(tf.float32, [None, m])
  z = tf.placeholder(tf.float32, [None, m])

  l1 = tf.layers.dense(
    x0, num_units_1, activation=activation_fn_1, use_bias=True)
  l2 = tf.layers.dense(
    l1, m, activation=activation_fn_1, use_bias=True)

  u1 = l2
  u1_cost = (tf.norm(u1)**2) / batch_size

  # The observed value for the second controller is the original controlled with noise
  x1 = x0 + u1
  y1 = tf.placeholder_with_default(x1 + z, [None, m])

  l3 = tf.layers.dense(
    y1, num_units_2, activation=activation_fn_2, use_bias=True)
  l4 = tf.layers.dense(
    l3, m, activation=activation_fn_2, use_bias=True)

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
      # l1, = plt.plot(x0_test, u1_test[0], label="U1 Test")
      # plt.legend(handles=[l1])
      # plt.title("{} Unit NN With Activation Fn {}".format(
      #   str(num_units_1), str(activation_fn_1)))
      # plt.savefig("figs/{}_{}_{}_u_1_{}.png".format(
      #   str(learning_rate), str(num_units_1), str(num_units_2) ,str(k_sq)))

      # plt.clf()
      # l1, = plt.plot(y1_test, u2_test[0], label="U2 Test")
      # plt.legend(handles=[l1])
      # plt.title("{} Unit NN With Activation Fn {}".format(
      #   str(num_units_2), str(activation_fn_2)))
      # plt.savefig("figs/{}_{}_{}_u_2_{}.png".format(
      #   str(learning_rate), str(num_units_1), str(num_units_2) ,str(k_sq)))

      # plt.clf()
      # l2, = plt.plot(x0_test, x1_test[0], label="X1 Test")
      # plt.title("{} Unit NN With Activation Fn {}".format(
      #   str(num_units_1), str(activation_fn_1)))
      # plt.legend(handles=[l2])
      # plt.savefig("figs/{}_{}_{}_x_1_{}.png".format(
      #   str(learning_rate), str(num_units_1), str(num_units_2) ,str(k_sq)))

if __name__ == "__main__":
  #learning_rates = [0.01, 0.001, 0.0001, 0.005]
  learning_rates = [1e-5, 1e-4]
  num_units_1s = [150, 250]
  num_units_2s = [30, 150]
  optimizers = [tf.train.AdamOptimizer]
  activation_fn_1 = tf.nn.sigmoid
  activation_fn_2 = tf.nn.sigmoid

  k_squared = float(sys.argv[1])
  num_epochs = int(sys.argv[2])
  run_num = 1
  x0_stddev = 5
  decay_rates = [1 - 1e-3]

  for lr in learning_rates:
    for decay_rate in decay_rates:
      for n_units_1 in num_units_1s:
        for n_units_2 in num_units_2s:
          for optimizer in optimizers: 
            np.random.seed(run_num)
            print('RUN NUMBER {}'.format(run_num))
            print('Numpy random seed {}'.format(run_num))
            print('Initial learning rate {}'.format(lr))
            print('GD Optimizier {}'.format(optimizer))
            print('Decay rate: {} is UNUSED'.format(decay_rate))
            print('N_units_1: {}'.format(n_units_1))
            print('N_units_2: {}'.format(n_units_2))
            print('Activation_fn_1: {}, activation_fn_2: {}'.format(activation_fn_1, activation_fn_2))
            print('k_squared: {}'.format(k_squared))
            print('x0 standard deviation: {}'.format(x0_stddev))
            print('Number of epochs: {}'.format(num_epochs))
            print('-----------------------------------------------\n')
            neural_net_run(m = 1, k_sq = k_squared, learning_rate = lr, epochs = num_epochs, batch_size = 100, 
              x_stddev = x0_stddev, activation_fn_1 = activation_fn_1, activation_fn_2 = activation_fn_2, num_units_1 = n_units_1, 
              num_units_2 = n_units_2, decay = decay_rate, test_averaging = 100, optimizer_func = optimizer)
            print('-----------------------------------------------\n')
            run_num += 1