
# coding: utf-8

# In[1]:


import tensorflow as tf
from tqdm import tqdm
import numpy as np
import uuid
import sys
import scipy.stats 
import matplotlib
import itertools 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os 
import pickle


NUMSTEPS = 20 #For printing and saving to tensorboard

NUMBATCHESTESTING = 100 #Number of batches for testing
UNIQUEID=  uuid.uuid1().__str__()[:6]
TENSORBOARDPATH = "tensorboard-layers-api/basic_" 
RESULTSFILEPATH = "2d/results/dat/basic_"
METARESULTSFILEPATH = "2d/results/meta_basic_cloud1"

repeat_run_list = [6,7, 1000, 1001] #Enter the metaresultsfilepath and desired run numbers here to replicate
FIGPATH = "2d/figures/basic_"+ UNIQUEID
CHECKPOINTPATH = "2d/checkpoints/basic_"   
TOPK = 5 #Collect top TOPK hyperparameters in separate file

#For representative sampling
MAXSTDREP = 4. #Maximum standard deviations for represntative sampling
NUMPOINTSX0 = 10 #Sampling for x0
NUMPOINTSZ = 10
m = 2
mode = 2 # Set mode = 1 for testing over only x0 and setting z = 0

############## 
#Helper functions
##############

def cartesian_product(*arrays): 
    return itertools.product(*arrays)


def list_to_str(prefix, cur_list):
    cur_list_str = [prefix+str(elem) for elem in cur_list]
    return cur_list_str
    
    
def tup_to_str(tups):
    cur_str = ""
    for k,tup in enumerate(tups):
        cur_str += str(tup)
        if k != len(tups)-1:
            cur_str+=','
    return cur_str

def variable_summaries(var_str, var):
    '''Attach a lot of summaries to a Tensor (for TensorBoard visualization).'''
    with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev', stddev)
#         tf.summary.scalar('max', tf.reduce_max(var))
#         tf.summary.scalar('min', tf.reduce_min(var))
#         tf.summary.histogram('histogram-'+var_str, var)
        tf.summary.scalar('norm_'+var_str, tf.norm(var))


# In[3]:


def neural_net(param_string="", params={}, m=2, verbose=False, mode = 2):

    '''m: Dimension of x0 (default =1)
       param_string: String encoding information about hyperparameters
       
       params: parameter dictionary that contains
       num_batches: number of batches 
       batch_size: number of samples per batch
       sigma_x0: standard deviation of x0
       sigma_z: standard deviation of noise
       optimizer: the optimizer used to minimize cost
    '''
    tf.reset_default_graph()
  
    global_layer_num = 1
    
    seed = params['seed']
    np.random.seed(seed)
    tf.set_random_seed(seed)

    #Learning rate and optimizer
    optimizer_function = params['optimizer_function']
    learning_rate = float(params['learning_rate'])

    #Placeholders for inputs
    x0 = tf.placeholder(tf.float32, [None, m])
    z = tf.placeholder(tf.float32, [None, m])
    adaptive_learning_rate = tf.placeholder_with_default(learning_rate, [])
    


    ###################################################
    #Layers
    #First we will go from x0 -> h1 via first layer structure 
    #Then we go from h1 -> u1 via second layer structure
    #x1 = x0 - u1
    #y1 = x1 + z
    #y1 -> h2 via third hidden layer
    #h2 -> u2 via fourth layer
    #x2 = y1 - u2
    ####################################################

    #The layers   
    layer_structures  = params['layer_structures']
    ##########################        
    #First layer structure to get h1 from x0
    ##########################
    layer_structure = layer_structures[0]
    num_units = layer_structure[0]
    layer_activation = layer_structure[1]
    num_layers = layer_structure[2]
    net = x0 #Input to first layer of this structure
    for k in range(num_layers):
        net = tf.layers.dense(inputs=net, units=num_units, activation=layer_activation                              , use_bias=True, name = 'layer' + str(global_layer_num),kernel_initializer=tf.glorot_normal_initializer())
        
        #Add to tensor board summary
        with tf.variable_scope('layer' + str(global_layer_num), reuse=True):
            with tf.name_scope('weights'):
                w = tf.get_variable('kernel')
                b = tf.get_variable('bias')
                variable_summaries('w'+str(global_layer_num), w)
                variable_summaries('b'+str(global_layer_num), b)

        global_layer_num += 1       
    h1 = net

    ############################        
    #Second layer structure to get u1 from h1
    ############################

    layer_structure = layer_structures[1]
    num_units = layer_structure[0]
    layer_activation = layer_structure[1]
    num_layers = layer_structure[2]
    net = h1 #Input to first layer of this structure
    for k in range(num_layers):
        net = tf.layers.dense(inputs=net, units=m, activation=layer_activation,                              use_bias=True, name = 'layer' + str(global_layer_num),kernel_initializer=tf.glorot_normal_initializer())
        #Add to tensor board summary
        with tf.variable_scope('layer' + str(global_layer_num), reuse=True):
            with tf.name_scope('weights'):
                w = tf.get_variable('kernel')
                b = tf.get_variable('bias')
                variable_summaries('w'+str(global_layer_num), w)
                variable_summaries('b'+str(global_layer_num), b)
        global_layer_num += 1    
    u1 = net

    ################
    #Cost and optimizer
    
    x1 = x0 + u1
    y1 = x1 + z    
    
    ############################        
    #Third layer structure to get h2 from y1
    ############################

    layer_structure = layer_structures[2]
    num_units = layer_structure[0]
    layer_activation = layer_structure[1]
    num_layers = layer_structure[2]
    net = y1
    for k in range(num_layers):
        net = tf.layers.dense(inputs=net, units=num_units, activation=layer_activation                               ,use_bias=True, name = 'layer' + str(global_layer_num),kernel_initializer=tf.glorot_normal_initializer())
        #Add to tensor board summary
        with tf.variable_scope('layer' + str(global_layer_num), reuse=True):
            with tf.name_scope('weights'):
                w = tf.get_variable('kernel')
                b = tf.get_variable('bias')
                variable_summaries('w'+str(global_layer_num), w)
                variable_summaries('b'+str(global_layer_num), b)
        global_layer_num += 1    
    
    h2 = net
    
     ############################        
    #Fourth layer structure to get u2 from h2
    ############################

    layer_structure = layer_structures[3]
    num_units = layer_structure[0]
    layer_activation = layer_structure[1]
    num_layers = layer_structure[2]
    net = h2 #Input to first layer of this structure
    for k in range(num_layers):
        net = tf.layers.dense(inputs=net, units=m, activation=layer_activation                               ,use_bias=True, name = 'layer' + str(global_layer_num), kernel_initializer=tf.glorot_normal_initializer())
        #Add to tensor board summary
        with tf.variable_scope('layer' + str(global_layer_num), reuse=True):
            with tf.name_scope('weights'):
                w = tf.get_variable('kernel')
                b = tf.get_variable('bias')
                variable_summaries('w'+str(global_layer_num), w)
                variable_summaries('b'+str(global_layer_num), b)
        global_layer_num += 1    
    u2 =net
    
    x2 = x1 - u2
    
    k_squared = params['k_squared']
    stage1_cost = k_squared*tf.reduce_mean(tf.reduce_mean((u1)**2,axis=1))    
    stage2_cost = tf.reduce_mean(tf.reduce_mean((x2)**2,axis=1))
    cost = stage1_cost+ stage2_cost
    optimizer = optimizer_function(adaptive_learning_rate).minimize(cost)

    ###################
    #Tensor board summary
    ####################
    tf.summary.scalar("cost", cost)
    tf.summary.scalar("adaptive_learning_rate", adaptive_learning_rate)
    merged_summary_op = tf.summary.merge_all()   



    ######################################
    #Session
    #####################################
    with tf.Session() as sess:

        #Creates/saves in directory named tensorboard-layers-api in same directory as the .ipynb
        uniq_id = TENSORBOARDPATH +  UNIQUEID +"/" + param_string
        summary_writer = tf.summary.FileWriter(uniq_id, graph=tf.get_default_graph())


        #######################################
        #Training
        #Train the nn on training set constructed by sampling x0 and z independently from gaussian
        #distributions with means 0 and standard deviations sigma_x0 and sigma_z respectively
        #######################################

        if verbose is True:
            print("Training...")

        #Initialization
        
        run_num = params['run_num']

#         init_op = tf.global_variables_initializer()
#         sess.run(init_op) 
        
        #Save initializations to file
        saver = tf.train.Saver()
        cur_checkpoint_path = params['checkpoint_path']
        saver.restore(sess, cur_checkpoint_path)

        
        
        batch_size = params['batch_size']
        sigma_x0 = params['sigma_x0']
        sigma_z = params['sigma_z']
        max_epochs = params['max_epochs']
        learning_rate_decay= float(params['learning_rate_decay'])

        step_size= np.ceil(max_epochs/NUMSTEPS) #For printing and saving to tensor board

        ''''''
        #WARNING: Can only be used when marginals along dimensions are independent 
        train_costs = np.zeros((max_epochs,1))
        for step in range(max_epochs): #Splitting into multiple batches so as to not run into memory issues
            batch_x0 = np.random.normal(loc=np.zeros((m,)), scale = sigma_x0, size=[batch_size,m])
            batch_z = np.random.normal(loc=np.zeros((m,)), scale = sigma_z, size=[batch_size,m])


            #Adaptive learning rate 
            cur_adaptive_learning_rate = learning_rate*float(learning_rate_decay**step)


            _,batch_cost,summary = sess.run([optimizer,cost,merged_summary_op],       
                        feed_dict = {x0:batch_x0, z:batch_z, adaptive_learning_rate:cur_adaptive_learning_rate\
                                    })
            train_costs[step] = batch_cost

            if step % step_size == 0: 
                summary_writer.add_summary(summary, step)
                if verbose is True:
                    print("---Step: {}, Cost: {}".format(step,batch_cost))

        ######################################
        #Testing on sampled Gaussians
        #Evaluate the nn on test set constructed by sampling x0 and z independently from gaussian
        #distributions  with means 0 and standard deviations sigma_x0 and sigma_z respectively            
        ######################################################
        if verbose is True:
            print("Evaluating on sampled gaussians...")
        num_batches = params['num_batches']
        batch_size = params['batch_size']
        sigma_x0 = params['sigma_x0']
        sigma_z = params['sigma_z']
        ''''''
        #WARNING: Can only be used when marginals along dimensions are independent 
        test_costs = np.zeros((num_batches,1))
        for i in range(num_batches): #Splitting into multiple batches so as to not run into memory issues
            batch_x0 = np.random.normal(loc=np.zeros((m,)), scale = sigma_x0, size=[batch_size,m])
            batch_z = np.random.normal(loc=np.zeros((m,)), scale = sigma_z, size=[batch_size,m])

            batch_cost = sess.run(cost, feed_dict = {x0:batch_x0, z:batch_z})
            test_costs[i] = batch_cost
            if verbose is True:
                if i % step_size == 0:
                    print("---Batch: {}, Cost: {}".format(i,batch_cost))

        avg_test_cost = float(np.mean(test_costs, axis = 0))
        if verbose is True:
            print("Average test cost: {}".format(avg_test_cost))
        
        ##########################################################################
        ###Test over representative x0 and z landscape to get complete picture
        ##########################################################################
        
        if verbose is True:
            print("Evaluating on representative samples...")
        sigma_x0 = params['sigma_x0']
        sigma_z = params['sigma_z']

        sigma_x0 = params['sigma_x0']
        sigma_z = params['sigma_z']
        
        x0_grid = params['x0_grid']  
        
        if mode is 2:
            z_grid = params['z_grid']
            z_density = params['z_density']
        if mode is 1:
            z_grid = np.zeros((1,m))        
            z_density = np.ones((1,1))
            
        total_z_density = sum(z_density)
        x0_density = params['x_density']

        costs_test_rep_fat = np.zeros((x0_grid.shape[0], z_grid.shape[0]))
        x1_test_rep_fat = np.zeros((x0_grid.shape[0], x0_grid.shape[1], z_grid.shape[0]))
        x2_test_rep_fat = np.zeros((x0_grid.shape[0], x0_grid.shape[1], z_grid.shape[0]))
        u2_test_rep_fat = np.zeros((x0_grid.shape[0], x0_grid.shape[1], z_grid.shape[0]))

        for i in range(x0_grid.shape[0]):
            cur_x0 = np.reshape(x0_grid[i], [1, m])

            for j in range(z_grid.shape[0]):
                cur_z_density = z_density[j]
                cur_z = np.reshape(z_grid[j], [1,m])
                raw_cost,x1_tmp,x2_tmp,u2_tmp = sess.run([cost,x1,x2,u2], feed_dict = {x0:cur_x0, z:cur_z} )

                costs_test_rep_fat[i,j] = raw_cost*cur_z_density

                for k in range(x1_test_rep_fat.shape[1]):

                    x1_test_rep_fat[i,k,j] = cur_z_density*x1_tmp[0,k]
                    x2_test_rep_fat[i,k,j] = cur_z_density*x2_tmp[0,k]
                    u2_test_rep_fat[i,k,j] = cur_z_density*u2_tmp[0,k]



        x1_test_rep_thin = np.sum(x1_test_rep_fat, axis =-1)/total_z_density
        costs_test_rep_thin = np.sum(costs_test_rep_fat, axis = -1)/total_z_density
        x2_test_rep_thin = np.sum(x2_test_rep_fat, axis = -1)/total_z_density
        u2_test_rep_thin = np.sum(u2_test_rep_fat, axis = -1)/total_z_density


        return x0_grid, costs_test_rep_fat, x1_test_rep_fat, x2_test_rep_fat, u2_test_rep_fat


# In[2]:


##################
#Main Loop
 ##################

#Define the x and z grids for representative testing

meta_results_file = METARESULTSFILEPATH + '.txt'

with open(meta_results_file, 'r') as f:
   data = f.readlines()
   f.close()

for line in data[1:]:
 
   clean_line = line.replace('\n', '')
#     print(clean_line)
   k, dat_path, checkpoint_path, avg_cost_test_rep, _ = clean_line.split(' ')
   
   if int(k) in repeat_run_list:
       with open(dat_path, 'rb') as f:
           hyperparameters_dict = pickle.load(f)
           f.close()

       tup = hyperparameters_dict['tup']

       layer_structure = tup[0]
       k_squared = tup[1]
       learning_rate = tup[2]
       learning_rate_decay = tup[3]
       max_epoch = tup[4]
       batch_size = tup[5]
       sigma_x0 = tup[6]
       sigma_z = tup[7]
       optimizer_function = tup[8]
       seed = tup[9]

       z1d = np.linspace(-MAXSTDREP*sigma_z, MAXSTDREP*sigma_z, num=NUMPOINTSZ)
       x1d = np.linspace(-MAXSTDREP*sigma_x0, MAXSTDREP*sigma_x0, num=NUMPOINTSX0)

       #WARNING: ONLY FOR m = 2
       if m == 2:
           z_grid = np.array(list(cartesian_product(z1d,z1d)))
           x_grid = np.array(list(cartesian_product(x1d,x1d)))


       z_distribution = scipy.stats.norm(loc = 0.*np.ones(m), scale = np.ones(m)*sigma_z)
       z_density = np.product(z_distribution.pdf(z_grid), axis = 1)

       x_distribution = scipy.stats.norm(loc = 0.*np.ones(m), scale = np.ones(m)*sigma_x0)
       x_density = np.product(x_distribution.pdf(x_grid), axis = 1)



       tup_str = hyperparameters_dict['tup_str']
       params = hyperparameters_dict['params']
#         print(params['checkpoint_path'])
   #     params = {'run_num':k, 'x0_grid':x_grid, 'z_grid':z_grid, 'z_density':z_density, 'x_density':x_density, 'seed':seed, 'num_batches':NUMBATCHESTESTING, 'batch_size':batch_size, 'sigma_x0':sigma_x0, 'sigma_z':sigma_z,              'optimizer_function':optimizer_function, 'learning_rate':learning_rate, 'max_epochs':max_epoch,              'learning_rate_decay':learning_rate_decay, 'layer_structures':layer_structure , 'k_squared':k_squared}

   # #     print(tup_str)
   #         avg_test_cost, avg_test_cost_rep, x0_test_rep, u1_test_rep, u2_test_rep, x1_test_rep, x2_test_rep =     neural_net(param_string= tup_str, params=params, verbose = True)   


       x0_test_rep, cost_test_rep, x1_test_rep, x2_test_rep, u2_test_rep =         neural_net(param_string= tup_str, params=params, verbose = True,m = m, mode = mode)   
   
#     #TODO: Calculate average cost here over x0
   
#     avg_cost_test_rep = cost_test_rep[0]
   
#     hyperparam_results_dict = {}
#     hyperparam_results_dict['tup_str'] = tup_str
#     hyperparam_results_dict['cost'] = cost_test_rep
#     hyperparam_results_dict['x1'] = x1_test_rep
#     hyperparam_results_dict['x2']= x2_test_rep
#     hyperparam_results_dict['x0'] = x_grid
#     hyperparam_results_dict['u2']= u2_test_rep
   
#     hyperparam_results_list.append([avg_cost_test_rep, hyperparam_results_dict])
   
   
# #         hyperparam_results_list.append([tup_str,avg_test_cost, avg_test_cost_rep, x0_test_rep, u1_test_rep,                                   u2_test_rep, x1_test_rep, x2_test_rep])
#     results_file = RESULTSFILEPATH + UNIQUEID + '.dat'
#     with open(results_file, 'ab') as f:
#         pickle.dump(hyperparam_results_dict,f)
#         f.close()



