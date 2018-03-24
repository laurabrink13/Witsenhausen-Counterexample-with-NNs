import tensorflow as tf
import numpy as np
import uuid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#NOTE: NOT SURE IF THIS WORKS 
#-AKHIL

#python
METARESULTSFILEPATH = "2d/results/meta_basic_cloud1"
meta_results_file = METARESULTSFILEPATH + '.txt'
with open(meta_results_file, 'r') as f:
   data = f.readlines()
   f.close()

x0_list = []
x1_list = []

repeat_run_list = [6,7, 1000, 1001] #Enter the metaresultsfilepath and desired run numbers here to replicate
for line in data[1:]:
    clean_line = line.replace('\n', '')
#     print(clean_line)
    k, dat_path, checkpoint_path, avg_cost_test_rep, _ = clean_line.split(' ')

    if int(k) in repeat_run_list:
        with open(dat_path, 'rb') as f:
            hyperparameters_dict = pickle.load(f)
            f.close()
        # for key in hyperparameters_dict.keys()
        print(hyperparameters_dict.keys())
        print(np.mean(hyperparameters_dict['cost']))
        x0_list.append(hyperparameters_dict['x0'])
        x1_list.append(hyperparameters_dict['x1'])
#         print('x0: ', hyperparameters_dict['x0'])
#         print('x1: ', hyperparameters_dict['x1'])





# x_axis = x0_list[0][:, 0]
# y_axis = x0_list[0][:, 1]
x_test = x0_list[0]
x1_test = x1_list[0]
u1_test = np.array(x1_test - x0_list[0])
num_test_intervals = len(x1_list[0])

#BEGIN PLOTTING
from mpl_toolkits.mplot3d import Axes3D
# x_axis = np.linspace(-15, 15, num=num_test_intervals)
# y_axis = np.linspace(-15, 15, num=num_test_intervals)

# PLOT X1_0

def twod_plot_all(x0_test, x1_test, u1_test, num_test_intervals):  
	'''
	'''  
	x_axis, y_axis = x0_test[:, 0], x0_test[:, 1]
    fig = plt.figure()
    fig.suptitle('X1 first element')
    ax = fig.gca(projection='3d')
    for i in range(num_test_intervals):
            ax.scatter(x_axis[i], y_axis[i], x1_test[i][0])
    #         ax.scatter(x_axis[i], y_axis[j], x1_test[1,i,j])
    ax.set_xlabel('x0_0')
    ax.set_ylabel('x0_1')
    ax.set_zlabel('x1_0')
    plt.tight_layout()
    plt.show()

    # PLOT X1_1
    fig = plt.figure()
    fig.suptitle('X1 second element')
    ax = fig.gca(projection='3d')
    for i in range(num_test_intervals):
            ax.scatter(x_axis[i], y_axis[i], x1_test[i][1])
    #         ax.scatter(x_axis[i], y_axis[j], x1_test[1,i,j])
    ax.set_xlabel('x0_0')
    ax.set_ylabel('x0_1')
    ax.set_zlabel('x1_1')
    plt.tight_layout()
    plt.show()

    # PLOT U1_0
    fig = plt.figure()
    fig.suptitle('U1 first element')
    ax = fig.gca(projection='3d')
    for i in range(num_test_intervals):
            ax.scatter(x_axis[i], y_axis[i], u1_test[i][0])
    #         ax.scatter(x_axis[i], y_axis[j], x1_test[1,i,j])
    ax.set_xlabel('x0_0')
    ax.set_ylabel('x0_1')
    ax.set_zlabel('u1_0')
    plt.tight_layout()
    plt.show()

    # fig = plt.figure()
    fig = plt.figure()
    fig.suptitle('U1 second element')
    ax = fig.gca(projection='3d')
    for i in range(num_test_intervals):
            ax.scatter(x_axis[i], y_axis[i], u1_test[i][1])
    #         ax.scatter(x_axis[i], y_axis[j], x1_test[1,i,j])
    ax.set_xlabel('x0_0')
    ax.set_ylabel('x0_1')
    ax.set_zlabel('u1_1')
    plt.tight_layout()
    plt.show()