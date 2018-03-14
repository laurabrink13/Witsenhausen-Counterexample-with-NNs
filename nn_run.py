import tensorflow as tf
import numpy as np
import uuid
import matplotlib.pyplot as plt
import sys

# m = 1 # dimension
# k_sq = 0.5
# learning_rate = 0.01
# epochs = 10000
# batch_size = 100
# x_stddev = 3
# activation_fn_1 = tf.nn.sigmoid
# activation_fn = tf.nn.relu
# num_units_1 = 150
# num_units_2 = 30
# decay = 1 - 1e-10

def neural_net_run(m, k_sq, learning_rate, epochs, batch_size, x_stddev, 
  activation_fn_1, activation_fn_2, num_units_1, num_units_2, decay, test_averaging):
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

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=adaptive_learning_rate).minimize(wits_cost)

  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
      sess.run(init_op)
      x_train = np.random.normal(size=epochs * batch_size * m, scale=x_stddev)

    # Train for some epochs
      for step in range(epochs):
          x_batch = x_train[step: step + (batch_size * m)].reshape((batch_size, m))

          # Noise has variance 1
          z_batch = np.random.normal(size=(batch_size, m), scale=1)

          _, val = sess.run(
          [optimizer, wits_cost],
          feed_dict={x0: x_batch, z: z_batch,
               adaptive_learning_rate: learning_rate * (decay**step)})

          if step % 100 == 0:
              print("step: {}, value: {}".format(step, val))

    # Test over a continuous range of X
      x0_test = np.linspace(-2*x_stddev, 2*x_stddev, num=100)
      y1_test = x0_test
      u1_test, u2_test, x1_test = np.zeros((1, 100)), np.zeros((1, 100)), np.zeros((1, 100))

      for i in range(100):
          u1t, u2t, x1t = 0, 0, 0
          for _ in range(test_averaging):
              u1tmp, u2tmp, x1tmp = sess.run(
                  [u1, u2, x1],
                  feed_dict={x0: x0_test[i].reshape((1, 1)),
                  z: np.random.normal(size=(1, 1), scale=1),
                  y1: y1_test[i].reshape((1, 1))
             })

          u1t += u1tmp
          u2t += u2tmp
          x1t += x1tmp

          u1_test[0, i] = u1t / test_averaging
          u2_test[0, i] = -u2t / test_averaging
          x1_test[0, i] = x1t / test_averaging

      l1, = plt.plot(x0_test, u1_test[0], label="U1 Test")
      plt.legend(handles=[l1])
      plt.title("{} Unit NN With Activation Fn {}".format(
        str(num_units_1), str(activation_fn_1)))
      plt.savefig("figs/{}_{}_{}_u_1_{}.png".format(
        str(learning_rate), str(num_units_1), str(num_units_2) ,str(k_sq)))

      plt.clf()
      l1, = plt.plot(y1_test, u2_test[0], label="U2 Test")
      plt.legend(handles=[l1])
      plt.title("{} Unit NN With Activation Fn {}".format(
        str(num_units_2), str(activation_fn_2)))
      plt.savefig("figs/{}_{}_{}_u_2_{}.png".format(
        str(learning_rate), str(num_units_1), str(num_units_2) ,str(k_sq)))

      plt.clf()
      l2, = plt.plot(x0_test, x1_test[0], label="X1 Test")
      plt.title("{} Unit NN With Activation Fn {}".format(
        str(num_units_1), str(activation_fn_1)))
      plt.legend(handles=[l2])
      plt.savefig("figs/{}_{}_{}_x_1_{}.png".format(
        str(learning_rate), str(num_units_1), str(num_units_2) ,str(k_sq)))

if __name__ == "__main__":
  learning_rates = [0.01, 0.001, 0.0001, 0.005]
  num_units_1s = [100, 150, 200]
  num_units_2s = [20, 30, 40]

  k_squared = float(sys.argv[1])
  num_epochs = int(sys.argv[2])

  for lr in learning_rates:
    for n_units_1 in num_units_1s:
      for n_units_2 in num_units_2s:
        print('RUNNING FOR learning_rate: {}, num_units_1: {}, num_units_2: {}'.format(lr, n_units_1, n_units_2))
        print('-----------------------------------------------\n')
        neural_net_run(m = 1, k_sq = 0.5, learning_rate = lr, epochs = 100, batch_size = 100, 
          x_stddev = 3, activation_fn_1 = tf.nn.sigmoid, activation_fn_2 = tf.nn.sigmoid, num_units_1 = n_units_1, 
          num_units_2 = n_units_2, decay = 1 - 1e-10, test_averaging = 100)
        print('-----------------------------------------------\n')

