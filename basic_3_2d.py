

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



NUMSTEPS = 20 #For printing and saving to tensorboard

NUMBATCHESTESTING = 100 #Number of batches for testing
UNIQUEID=  uuid.uuid1().__str__()[:6]
TENSORBOARDPATH = "tensorboard-layers-api/basic_" 
RESULTSFILEPATH = "results/basic_"
FIGPATH = "figures/basic_"+ UNIQUEID
TOPK = 5 #Collect top TOPK hyperparameters in separate file

#For representative sampling
MAXSTDREP = 4. #Maximum standard deviations for represntative sampling
NUMPOINTSX0 = 100 #Sampling for x0
NUMPOINTSZ = 100


# In[2]:


############## 
#Helper functions
##############

def cartesian_product(*arrays): 
    return itertools.product(*arrays)


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


def neural_net(param_string="", params={}, m=2, verbose=False):

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
        net = tf.layers.dense(inputs=net, units=num_units, activation=layer_activation,                              use_bias=True, name = 'layer' + str(global_layer_num),kernel_initializer=tf.glorot_normal_initializer())
        #Add to tensor board summary
        with tf.variable_scope('layer' + str(global_layer_num), reuse=True):
            with tf.name_scope('weights'):
                w = tf.get_variable('kernel')
                b = tf.get_variable('bias')
                variable_summaries('w'+str(global_layer_num), w)
                variable_summaries('b'+str(global_layer_num), b)
        global_layer_num += 1    
    u1 =net

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
        net = tf.layers.dense(inputs=net, units=num_units, activation=layer_activation                               ,use_bias=True, name = 'layer' + str(global_layer_num), kernel_initializer=tf.glorot_normal_initializer())
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
    stage1_cost = k_squared*tf.reduce_mean(tf.reduce_mean(tf.norm(u1,'euclidean'))) 
    stage2_cost = tf.reduce_mean(tf.reduce_mean(tf.norm(x2, 'euclidean')))
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
        init_op = tf.global_variables_initializer()
        sess.run(init_op) 

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
        
       ##########################################
        ###Test over representative x0 and z landscape
       #################################################
        if verbose is True:
            print("Evaluating on representative samples...")
        sigma_x0 = params['sigma_x0']
        sigma_z = params['sigma_z']
        
        sigma_x0 = params['sigma_x0']
        sigma_z = params['sigma_z']

        #WARNING:CODE will work only for m = 1
        
        x0_test_rep=np.array(np.meshgrid
                   (np.linspace(-MAXSTDREP*sigma_x0, MAXSTDREP*sigma_x0, num=NUMPOINTSX0),
                    np.linspace(-MAXSTDREP*sigma_x0, MAXSTDREP*sigma_x0, num=NUMPOINTSX0)))
        z_test_rep=np.array(np.meshgrid
                   (np.linspace(-MAXSTDREP*sigma_z, MAXSTDREP*sigma_z, num=NUMPOINTSZ),
                    np.linspace(-MAXSTDREP*sigma_z, MAXSTDREP*sigma_z, num=NUMPOINTSZ)))
        
            
        #x0_test_rep = np.linspace(-MAXSTDREP*sigma_x0, MAXSTDREP*sigma_x0, num=NUMPOINTSX0)
        #z_test_rep = np.linspace(-MAXSTDREP*sigma_z, MAXSTDREP*sigma_z, num=NUMPOINTSZ)

        #x0_distribution = scipy.stats.norm(loc=0., scale=sigma_x0)
        #z_distribution = scipy.stats.norm(loc = 0., scale = sigma_z)

        #x0_pdf= x0_distribution.pdf(x0_test_rep)
        #z_pdf = z_distribution.pdf(z_test_rep)


        #test_costs_rep = np.zeros((NUMPOINTSX0, NUMPOINTSZ))
        #u1_test_rep_2d = np.zeros((NUMPOINTSX0, NUMPOINTSZ))
        #u2_test_rep_2d = np.zeros((NUMPOINTSX0, NUMPOINTSZ))
        #x1_test_rep_2d = np.zeros((NUMPOINTSX0, NUMPOINTSZ))
        #x2_test_rep_2d = np.zeros((NUMPOINTSX0, NUMPOINTSZ))
        
        
        # access the first element in u1 with u1_test[0,i,j,k,l] and the second with u1_test[1,i,j,k,l]
        test_costs_rep = np.array(np.meshgrid(np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSZ), np.zeros(NUMPOINTSZ)))
        u1_test_rep_4d = np.array(np.meshgrid(np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSZ), np.zeros(NUMPOINTSZ)))
        u2_test_rep_4d = np.array(np.meshgrid(np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSZ), np.zeros(NUMPOINTSZ)))
        x1_test_rep_4d = np.array(np.meshgrid(np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSZ), np.zeros(NUMPOINTSZ)))
        x2_test_rep_4d = np.array(np.meshgrid(np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSX0), np.zeros(NUMPOINTSZ), np.zeros(NUMPOINTSZ)))
        
           
        for i in range(NUMPOINTSX0):
            for j in range(NUMPOINTSX0):
                for k in range(NUMPOINTSZ):
                    for l in range(NUMPOINTSZ):
                        x0_current_test = np.array([x0_test_rep[0,i,j], x0_test_rep[1,i,j]])
                        z_current_test = np.array([z_test_rep[0,k,l], z_test_rep[1,k,l]])
                        raw_cost, u1tmp, u2tmp, x1tmp, x2tmp = sess.run([cost, u1, u2, x1, x2], # return these variables
                                    feed_dict={x0: x0_current_test.reshape((1, 2)), z: z_current_test.reshape((1, 2))})

                        test_costs_rep[0, i,j,k,l] = raw_cost
                        u1_test_rep_4d[0, i,j,k,l], u1_test_rep_4d[1,i,j,k,l] = u1tmp[0,0], u1tmp[0,1]
                        u2_test_rep_4d[0, i,j,k,l], u2_test_rep_4d[1,i,j,k,l] = -u2tmp[0,0], -u2tmp[0,1]
                        x1_test_rep_4d[0, i,j,k,l], x1_test_rep_4d[1,i,j,k,l] = x1tmp[0,0], x1tmp[0,1]
                        x2_test_rep_4d[0, i,j,k,l], x2_test_rep_4d[1,i,j,k,l] = x2tmp[0,0], x2tmp[0,1]
        avg_test_cost_rep = np.mean(test_costs_rep[0])
        u1_test_rep = np.mean(u1_test_rep_4d)
        u2_test_rep = np.mean(u2_test_rep_4d)
        x1_test_rep = np.mean(x1_test_rep_4d)
        x2_test_rep = np.mean(x2_test_rep_4d)
        
        
        #total_density = 0.
        #for i in range(NUMPOINTSX0):
        #    for j in range(NUMPOINTSZ):
        #        cur_density = x0_pdf[i]*z_pdf[j]
        #       total_density += cur_density
               
        #        raw_cost,cur_u1,cur_u2,cur_x1,cur_x2 = sess.run([cost,u1,u2,x1,x2], feed_dict = {x0:np.reshape(x0_test_rep[i],[-1,1]),                                                   z:np.reshape(z_test_rep[j] ,[-1,1])})
        #        scaled_cost = raw_cost*cur_density
                
        #        test_costs_rep[i,j] = scaled_cost
               
        #        u1_test_rep_2d[i,j] = cur_u1*z_pdf[j]
        #        u2_test_rep_2d[i,j] = cur_u2*z_pdf[j]
        #        x1_test_rep_2d[i,j] = cur_x1*z_pdf[j]
        #        x2_test_rep_2d[i,j] = cur_x2*z_pdf[j]               
        #avg_test_cost_rep = np.sum(test_costs_rep/total_density)
        #u1_test_rep = np.sum(u1_test_rep_2d, axis = 1)/np.sum(z_pdf)
        #u2_test_rep = np.sum(u2_test_rep_2d, axis = 1)/np.sum(z_pdf)
        #x1_test_rep = np.sum(x1_test_rep_2d, axis = 1)/np.sum(z_pdf)
        #x2_test_rep = np.sum(x2_test_rep_2d, axis = 1)/np.sum(z_pdf)


        
        
        if verbose is True:
            print("Average test cost rep: {}".format(avg_test_cost_rep))     
        

            
        return avg_test_cost, avg_test_cost_rep, x0_test_rep, u1_test_rep, u2_test_rep,x1_test_rep, x2_test_rep


# In[4]:


def main():
    #######
    #Cartesian product to get the layer structures
    ######
    #First find tuples for different types of layers

    hidden_units_type1 = [150]
    hidden_units_type1_str= ['150']
    # activations_type1 = [tf.nn.sigmoid, None]
    # activations_type1_str = ['sigmoid', 'linear']

    hidden_units_type3 = [30]
    hidden_units_type3_str= ['30']


    num_layers_type3 = [2]
    num_layers_type3_str = ['2']

    # activations_type1 = [None]
    # activations_type1_str = ['linear']

    activations_type1 = [tf.nn.sigmoid]
    activations_type1_str = ['sigmoid']

    num_layers_type1 = [2]
    num_layers_type1_str = ['2']


    hidden_units_type2 = [1]
    hidden_units_type2_str= ['1']
    activations_type2 = [tf.identity]
    activations_type2_str = ['identity']
    num_layers_type2= [1]
    num_layers_type2_str = ['1']

    layers_type1 = cartesian_product(hidden_units_type1, activations_type1, num_layers_type1)
    layers_type1_str = ['ls1_' + tup_to_str(tup) for tup in cartesian_product(hidden_units_type1_str, activations_type1_str, num_layers_type1_str)]


    layers_type2 = cartesian_product(hidden_units_type2, activations_type2, num_layers_type2)
    layers_type2_str = ['ls2_'+tup_to_str(tup) for tup in cartesian_product(hidden_units_type2_str, activations_type2_str, num_layers_type2_str)]


    layers_type3 = cartesian_product(hidden_units_type3, activations_type1, num_layers_type3)
    layers_type3_str = ['ls3_' + tup_to_str(tup) for tup in cartesian_product(hidden_units_type3_str,                                                                   activations_type1_str, num_layers_type3_str)]


    layers_type4 = cartesian_product(hidden_units_type2, activations_type2, num_layers_type2)
    layers_type4_str = ['ls4_'+tup_to_str(tup) for tup in cartesian_product(hidden_units_type2_str, activations_type2_str, num_layers_type2_str)]


    layers = cartesian_product(layers_type1, layers_type2, layers_type3, layers_type4)

    layers_str = [tup_to_str(tup) for tup in cartesian_product(layers_type1_str, layers_type2_str, layers_type3_str,                                                   layers_type4_str)]

    #Next find tuples for the whole parameter space
    k_squared_vals = [0.04]
    k_squared_vals_str = ['k_'+str(elem) for elem in k_squared_vals]
    learning_rates = [1e-1]
    learning_rates_str = ['lr_'+str(elem) for elem in learning_rates]
    learning_rate_decays = [1.-1e-9]
    learning_rate_decays_str = ['1-1e-9']
    max_epochs = [10000]
    max_epochs_str = ['me_' + str(elem) for elem in max_epochs]
    batch_sizes = [10000]
    batch_sizes_str = ['bsz_' +str(elem) for elem in batch_sizes]
    sigmas_x0 = [5.]
    sigmas_x0_str = ['stdx0_' +str(elem) for elem in sigmas_x0]
    sigmas_z = [1.]
    sigmas_z_str = ['stdz_' +str(elem) for elem in sigmas_z]


    given_seeds = np.arange(3)
    given_seeds_str = [str(elem) for elem in given_seeds]

    optimizer_functions = [tf.train.AdamOptimizer]
    optimizer_functions_str = ['Adam']


    hyperparam_tuples = list(cartesian_product(layers, k_squared_vals, learning_rates, learning_rate_decays,                                    max_epochs, batch_sizes, sigmas_x0, sigmas_z, optimizer_functions, given_seeds))
    hyperparam_tuples_str = [tup_to_str(tup) for tup in list(cartesian_product(layers_str, k_squared_vals_str,                                         learning_rates_str, learning_rate_decays_str, max_epochs_str,                                 batch_sizes_str, sigmas_x0_str, sigmas_z_str, optimizer_functions_str, given_seeds_str))]

     
    # print(hyperparam_tuples)


    # In[5]:


    # get_ipython().run_line_magic('matplotlib', 'inline')
    ##################
    #Main Loop
    ##################
    hyperparam_results_list = []
    for k,tup in enumerate(tqdm(hyperparam_tuples)):
        
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

        tup_str = hyperparam_tuples_str[k]
        params = {'seed':seed, 'num_batches':NUMBATCHESTESTING, 'batch_size':batch_size, 'sigma_x0':sigma_x0, 'sigma_z':sigma_z,              'optimizer_function':optimizer_function, 'learning_rate':learning_rate, 'max_epochs':max_epoch,              'learning_rate_decay':learning_rate_decay, 'layer_structures':layer_structure , 'k_squared':k_squared}
        
        print(tup_str)
        avg_test_cost, avg_test_cost_rep, x0_test_rep, u1_test_rep, u2_test_rep, x1_test_rep, x2_test_rep =     neural_net(param_string= tup_str, params=params, verbose = True)   
        
       
        hyperparam_results_list.append([tup_str,avg_test_cost, avg_test_cost_rep, x0_test_rep, u1_test_rep,                                   u2_test_rep, x1_test_rep, x2_test_rep])
        results_file = RESULTSFILEPATH + UNIQUEID + '.txt'
        with open(results_file, 'a') as f:
          f.write(tup_str + " " + str(avg_test_cost) + " " + str(avg_test_cost_rep) + '\n')
          f.close()


    # In[ ]:


    ##############
    #Process Results
    ##############
    # get_ipython().run_line_magic('matplotlib', 'inline')

    hyperparam_results_list_sorted = sorted(hyperparam_results_list, key=lambda tup: tup[2])   

    #Write results to file

    if os.path.exists(FIGPATH) is False:
        os.system("mkdir " + FIGPATH)
    top_results_file = RESULTSFILEPATH + UNIQUEID + '_top' + str(TOPK) + '.txt'
    with open(top_results_file, 'w') as g:
        for k in range(min(TOPK,len(hyperparam_results_list))):
            tup_str = hyperparam_results_list_sorted[k][0]
            avg_test_cost = hyperparam_results_list_sorted[k][1]
            avg_test_cost_rep = hyperparam_results_list_sorted[k][2]
            g.write(tup_str + " " + str(avg_test_cost) + " " + str(avg_test_cost_rep) + '\n')
            
            x0_test_rep = hyperparam_results_list_sorted[k][3]
            u1_test_rep = hyperparam_results_list_sorted[k][4]
            u2_test_rep = hyperparam_results_list_sorted[k][5]
            x1_test_rep = hyperparam_results_list_sorted[k][6]        
            x2_test_rep = hyperparam_results_list_sorted[k][7]
            

            plt.close('all')
            plt.figure()
            plt.title('u1 vs x0, cost = ' + str(avg_test_cost_rep) + ', params = ' + tup_str)          
            plt.scatter(x0_test_rep, u1_test_rep)        
            plt.xlabel('x0')
            plt.ylabel('u1')
            fig_path = FIGPATH + "/u1_vs_x0_" + tup_str + '.png'
            plt.savefig(fig_path, format = 'png')
            # plt.show()
            
            plt.close('all')
            plt.figure()
            plt.title('u2 vs x0, cost = ' + str(avg_test_cost_rep) + ', params = ' + tup_str)          
            plt.scatter(x0_test_rep, u2_test_rep)        
            plt.xlabel('x0')
            plt.ylabel('u2')
            fig_path = FIGPATH + "/u2_vs_x0_" + tup_str + '.png'
            plt.savefig(fig_path, format = 'png')
            # plt.show()
            
            plt.close('all')
            plt.figure()
            plt.title('x1 vs x0, cost = ' + str(avg_test_cost_rep) + ', params = ' + tup_str)          
            plt.scatter(x0_test_rep, x1_test_rep)        
            plt.xlabel('x0')
            plt.ylabel('x1')
            fig_path = FIGPATH + "/x1_vs_x0_" + tup_str + '.png'
            plt.savefig(fig_path, format = 'png')
            # plt.show()
            
            plt.close('all')
            plt.figure()
            plt.title('x2 vs x0, cost = ' + str(avg_test_cost_rep) + ', params = ' + tup_str)          
            plt.scatter(x0_test_rep, x2_test_rep)        
            plt.xlabel('x0')
            plt.ylabel('x2')
            fig_path = FIGPATH + "/x2_vs_x0_" + tup_str + '.png'
            plt.savefig(fig_path, format = 'png')
            # plt.show()
                       
        g.close()

if __name__ == "__main__":
    main()