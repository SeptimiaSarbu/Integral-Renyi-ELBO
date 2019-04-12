#Copyright 2016 Septimia Sarbu and Luigi Malago

import os
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.distributions import MultivariateNormalDiag
from tensorflow.contrib.distributions import Normal

#from Image_generator_from_argo import ImagesGenerator

from scipy.stats import gaussian_kde
from scipy.special import dawsn

import matplotlib.pyplot as plt

import pdb
#import pandas as pd

mnist = input_data.read_data_sets('MNIST')

input_dim = 784
n_samples = 1
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
learning_rate = 0.0005
alpha = 1.5
batch_size = 100  # reconstructed images better with 10 than with 100, for n_epochs=10 and n_steps=1000
eps_int_sup = 8e-1
eps_int_inf = 8e-1
tf_type = tf.float64

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#tf.set_random_seed(1234)

##########################################################################3
# Implements custom gradient for the Dawson function, computed in numpy
# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def numpy_dawsn(x):
    return dawsn(x)


# Def custom function:
def mydawson(x, name=None):

    with tf.name_scope(name, "Mydawson", [x]) as name:
        dawson_x = py_func(numpy_dawsn,
                        [x],
                        [tf_type],
                        name=name,
                        grad=_MyDawsonGrad)  # <-- here's the call to the gradient
        return dawson_x[0]

# Actual gradient:
def _MyDawsonGrad(op, grad):
    x = op.inputs[0]
    grad_x = 1 - 2 * tf.multiply(x,tf.py_func(numpy_dawsn, [x], tf_type))

    return grad * grad_x  # add your custom gradient here to use in the backpropagation/chain rule, along with the other gradients given by grad
##########################################################################3

def weight_variable(shape,name):
    #initial = tf.truncated_normal(shape, stddev=0.001, dtype=tf_type)
    #initial = tf.contrib.layers.xavier_initializer()
    initial = tf.get_variable(name,shape=shape, dtype=tf_type, initializer=tf.contrib.layers.xavier_initializer())
    return initial#tf.Variable(initial)


def bias_variable(shape,name):
    #initial = tf.constant(0., shape=shape, dtype=tf_type)
    initial = tf.get_variable(name,shape=shape, dtype=tf_type, initializer=tf.contrib.layers.xavier_initializer())
    return initial#tf.Variable(initial)

def numpy_func(bt,at):
    rez = np.zeros(shape=[batch_size,input_dim])
    for i in range(0,batch_size,1):
        for j in range(0,input_dim,1):
            '''
            if (0.0>bt[i][j]) & (bt[i][j]>at[i][j]):
                rez[i][j] = -at[i][j]
            elif (at[i][j]>0.0) & (bt[i][j]>at[i][j]):
                 rez[i][j] = bt[i][j]
            elif (bt[i][j]>0.0) & (at[i][j]<0.0) & (np.abs(bt[i][j])<np.abs(at[i][j])):
                 rez[i][j] = -at[i][j]
            '''
            if (np.abs(bt[i][j])<np.abs(at[i][j])):
                rez[i][j] = np.abs(at[i][j])
            else:
                rez[i][j] = np.abs(bt[i][j])
    return rez

def gaussian_Renyi_cdf_decoder(hidden_decoder, x_samples):
    W_decoder_hidden_reconstr_mu = weight_variable([hidden_decoder_dim, input_dim],"W_decoder_hidden_reconstr_mu")
    b_decoder_hidden_reconstr_mu = bias_variable([input_dim],"b_decoder_hidden_reconstr_mu")

    W_decoder_hidden_reconstr_logvar = weight_variable([hidden_decoder_dim, input_dim],"W_decoder_hidden_reconstr_logvar")
    b_decoder_hidden_reconstr_logvar = bias_variable([input_dim],"b_decoder_hidden_reconstr_logvar")

    offset = tf.constant(1.0, shape=b_decoder_hidden_reconstr_mu.get_shape(), dtype=tf_type)

    #param_mul = tf.constant(0.5, dtype=tf_type)
    #mu_decoder = tf.divide(offset + tf.nn.softsign(tf.matmul(hidden_decoder, W_decoder_hidden_reconstr_mu) + b_decoder_hidden_reconstr_mu),2.0)

    param_mul = tf.constant(1.0, dtype=tf_type)
    mu_decoder = tf.sigmoid(tf.multiply(param_mul, tf.matmul(hidden_decoder, W_decoder_hidden_reconstr_mu) + b_decoder_hidden_reconstr_mu))

    logvar_decoder = tf.matmul(hidden_decoder, W_decoder_hidden_reconstr_logvar) + b_decoder_hidden_reconstr_logvar

    #param = tf.constant((64.0) ** 1, dtype=tf_type)  # 32 is ok too
    #offset_std = tf.divide(1, tf.sqrt(param))
    #std_decoder = offset_std + tf.divide(logvar_decoder,tf.sqrt(offset + tf.multiply(param, tf.pow(logvar_decoder, 2))))

    std_decoder = tf.exp(0.5 * logvar_decoder) + 1e-5

    ###############################################################
    #How to compute Dawson's function in tensorflow
    # input = tf.placeholder(tf.float32)
    # y = tf.py_func(dawsn, [input], tf.float32)
    # data = 1.0
    # rez_np = sess.run([y], {input: data})
    #END How to compute Dawson's function in tensorflow
    ###############################################################
    term1 = tf.divide(np.sqrt((alpha-1)/((2-alpha)*2)),std_decoder)
    b = x_samples + eps_int_sup #eps_int = 0.5e-1
    a = x_samples - eps_int_inf
    #bt = tf.constant(0.5, dtype=tf_type)
    #at = tf.constant(0.5, dtype=tf_type)
    bt = tf.multiply(term1, b - mu_decoder)
    at = tf.multiply(term1, a - mu_decoder)

    #r,c = mu_decoder.get_shape()
    #rez = tf.py_func(numpy_func, [bt,at], tf_type)
    #condition = tf.less(tf.abs(bt), tf.abs(at))
    #rez = tf.where(condition, tf.abs(at), tf.abs(bt))
    # dawson_sup = tf.py_func(dawsn, [rez], tf_type)

    #rez = tf.maximum(tf.abs(bt), tf.abs(at))
    dawson_sup = mydawson(bt)
    #term2 = tf.multiply(mydawson(bt), tf.exp(tf.pow(b-mu_decoder,2))) - tf.multiply(mydawson(at), tf.exp(tf.pow(a-mu_decoder,2))) + 1e-5
    #term2 = tf.multiply(mydawson(bt), tf.exp(tf.pow(bt, 2))) - tf.multiply(mydawson(at), tf.exp(tf.pow(at, 2))) + 1e-8

    term2 = tf.multiply(mydawson(bt), tf.exp(tf.pow(bt, 2)-tf.pow(at, 2))) - mydawson(at) + 1e-8

    #constant0 = tf.constant(0.0,shape=[batch_size,input_dim],dtype=tf_type)

    #precision = tf.constant(np.exp(eps_int_sup**2)*dawsn(eps_int_sup)+np.exp(eps_int_inf**2)*dawsn(eps_int_inf), shape=[batch_size, input_dim], dtype=tf_type)

    #condition = tf.less(term2, constant0)
    #rez = tf.where(condition, precision , term2)

    #rez = precision + term2
    rez = term2
    log_term2_2 = tf.pow(at,2) + tf.log(rez)

    term1_2 = tf.divide(std_decoder,np.sqrt((alpha - 1) / ((2 - alpha) * 2)))
    #log_Id = tf.log(term1_2) + tf.log(dawson_sup) + tf.pow(bt,2)
    log_Id = tf.log(term1_2) + log_term2_2

    #if tf.greater(at,0.0):
    #    dawson_inf = tf.py_func(dawsn, [at], tf_type)
    #else:
    #    dawson_inf = tf.py_func(dawsn, [-at], tf_type)
    #dawson_inf = tf.cond(tf.less(at,0.0), lambda: tf.py_func(dawsn, [-at], tf_type), lambda: tf.py_func(dawsn, [at], tf_type))

    '''
    dawson_inf = tf.py_func(dawsn, [tf.abs(at)], tf_type)

    term2 = tf.divide(std_decoder,np.sqrt((2-alpha)*2/(alpha-1)))
    C = tf.multiply(tf.pow(np.sqrt(2*np.pi)*std_decoder,(alpha-1)/(2-alpha)),term2)

    #term_sup = tf.multiply(tf.exp(tf.pow(bt, 2)),dawson_sup)
    #term_inf = tf.multiply(tf.exp(tf.pow(at, 2)), dawson_inf)

    term_sup = tf.multiply(tf.exp(tf.pow(bt, 2)-tf.pow(at, 2)),dawson_sup)
    term_inf = dawson_inf

    #log_integral_pdf = tf.log(tf.multiply(C,term_sup-term_inf)+1e-6)
    log_integral_pdf = tf.log(tf.multiply(C, term_sup - term_inf))

    h_z1 = tf.pow(at, 2) + log_integral_pdf
    '''
    elem1 = tf.log(np.sqrt(2*np.pi)*std_decoder)
    #elem2 = log_Id/(alpha-1)
    elem2 = log_Id *( (2-alpha)/(alpha-1) )

    h_z1 = elem1 + elem2

    ###############################################################################################
    # These lines are for the importance sampling estimate of log_int_px
    log_pxz = Normal(mu_decoder, std_decoder).log_prob(x_samples)
    pxz = Normal(mu_decoder, std_decoder).prob(x_samples)

    cdf_poz = Normal(mu_decoder, std_decoder).cdf(x_samples + eps_int_sup)#0.5e-1)  # 2
    cdf_neg = Normal(mu_decoder, std_decoder).cdf(x_samples - eps_int_inf)#0.5e-1)  # 2

    log_cdf_components = tf.log(cdf_poz - cdf_neg + 1e-8)
    log_cdf2_pxz = log_cdf_components
    #END These lines are for the importance sampling estimate of log_int_px
    ###############################################################################################

    #return log_cdf_reconstruct, std_decoder, mu_decoder, bt, at, b, a,term_sup, term_inf, dawson_inf, dawson_sup, h_z1
    return std_decoder, mu_decoder, bt, at, b, a,dawson_sup, h_z1,elem1, elem2, log_Id, term1, term1_2, term2, rez,log_pxz, pxz, log_cdf2_pxz

x = tf.placeholder(tf.float64, shape=[None, input_dim])
x_samples = tf.reshape(tf.tile(x, [1, n_samples]), [-1, input_dim])

#W_encoder_input_hidden = weight_variable([input_dim, hidden_encoder_dim])
#b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
W_encoder_input_hidden = weight_variable([input_dim, hidden_encoder_dim],"W_encoder_input_hidden")
b_encoder_input_hidden = bias_variable([hidden_encoder_dim],"b_encoder_input_hidden")

# Hidden layer encoder
hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

#W_encoder_hidden_mu = weight_variable([hidden_encoder_dim, latent_dim])
#b_encoder_hidden_mu = bias_variable([latent_dim])
W_encoder_hidden_mu = weight_variable([hidden_encoder_dim, latent_dim],"W_encoder_hidden_mu")
b_encoder_hidden_mu = bias_variable([latent_dim],"b_encoder_hidden_mu")

# Mu encoder
mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu
mu_encoder_samples = tf.reshape(tf.tile(mu_encoder, [1, n_samples]), [-1, latent_dim])

W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim, latent_dim],"W_encoder_hidden_logvar")
b_encoder_hidden_logvar = bias_variable([latent_dim],"b_encoder_hidden_logvar")

# Sigma encoder
logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(mu_encoder_samples), name='epsilon', dtype=tf_type)

# Sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)+1e-5
std_encoder_samples = tf.reshape(tf.tile(std_encoder, [1, n_samples]), [-1, latent_dim])

z = mu_encoder_samples + tf.multiply(std_encoder_samples, epsilon)

W_decoder_z_hidden = weight_variable([latent_dim, hidden_decoder_dim],"W_decoder_z_hidden")
b_decoder_z_hidden = bias_variable([hidden_decoder_dim],"b_decoder_z_hidden")

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

#log_pdfs_reconstruct, std_decoder, mu_decoder, bt, at, b, a,term_sup, term_inf, dawson_inf, dawson_sup, h_z1 = gaussian_Renyi_cdf_decoder(hidden_decoder, x_samples)
std_decoder, mu_decoder, bt, at, b, a,dawson_sup, h_z1,elem1, elem2, log_Id, term1, term1_2, term2, rez,log_pxz, pxz, log_cdf2_pxz = gaussian_Renyi_cdf_decoder(hidden_decoder, x_samples)

# evaluate the pdf q(z|x_samples)
log_pdf_qzx = MultivariateNormalDiag(mu_encoder_samples,std_encoder_samples).log_prob(z)
pdf_qzx = MultivariateNormalDiag(mu_encoder_samples,std_encoder_samples).prob(z)

# evaluate the pdf p(z)
mu_z = tf.constant(0.0, shape=[batch_size,latent_dim], dtype=tf_type)
mu_z_samples = tf.reshape(tf.tile(mu_z, [1, n_samples]), [-1, latent_dim])
std_z = tf.constant(1.0, shape=[batch_size,latent_dim], dtype=tf_type)
std_z_samples = tf.reshape(tf.tile(std_z, [1, n_samples]), [-1, latent_dim])

log_pdf_pz = MultivariateNormalDiag(mu_z_samples,std_z_samples).log_prob(z)

pdf_pz = MultivariateNormalDiag(mu_z_samples,std_z_samples).prob(z)

#h_z2 = tf.pow(tf.divide(pdf_qzx,pdf_pz),alpha-1)

h_z2 = log_pdf_qzx-log_pdf_pz
#h_z2 = tf.log(tf.divide(pdf_qzx,pdf_pz)+1e-6)

sum_h_z1 = tf.reduce_sum(h_z1,1)
#h_z = tf.multiply(h_z1,h_z2)

#h_z = (2-alpha)/(alpha-1)*sum_h_z1 + h_z2
h_z = sum_h_z1 + h_z2

#the above computations are valid only for n_samples=1; for more samples, we need to implement the log-sum-exp trick
# mean over number of samples from q(z|x)
#Renyi_cdf = tf.log(tf.reduce_mean(tf.reshape(h_z, [-1, n_samples]), axis=1))
Renyi_cdf = h_z

# average bound over the batch size
log_len_int = tf.constant(np.log(eps_int_sup+eps_int_inf)/(alpha-1), shape=[batch_size,], dtype=tf_type)

#loss = tf.reduce_mean(log_len_int-Renyi_cdf)
loss = tf.reduce_mean(log_len_int-Renyi_cdf)

#loss = log_pdfs_reconstruct
#loss_summ = tf.summary.scalar("lowerbound", loss)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(-loss)

#train_step_gradients_variables = tf.train.AdamOptimizer(learning_rate).compute_gradients(-loss)

######################################################################
#Do gradient clipping to avoid nans in the gradients

#opt_object = tf.train.AdamOptimizer(learning_rate)
#grads_vars = opt_object.compute_gradients(-loss)
#clipped_grads_vars = [(tf.clip_by_value(grads,-150,150),variables) for grads, variables in grads_vars]

#train_step = opt_object.apply_gradients(clipped_grads_vars)
#END Do gradient clipping to avoid nans in the gradients

# These lines are needed to estimate log(p(x))
log_qzx = Normal(mu_encoder, std_encoder).log_prob(z)
qzx = Normal(mu_encoder, std_encoder).prob(z)

mu_prior = tf.constant(0.0,shape=[batch_size,latent_dim],dtype=tf_type)
std_prior = tf.constant(1.0,shape=[batch_size,latent_dim],dtype=tf_type)
#mu_prior = tf.constant(0.0,shape=[1,1],dtype=tf_type)
#std_prior = tf.constant(1.0,shape=[1,1],dtype=tf_type)

#mu_prior_shape=mu_prior.get_shape()

log_pz = Normal(mu_prior, std_prior).log_prob(z)
pz = Normal(mu_prior, std_prior).prob(z)
#END These lines are needed to estimate log(p(x))


# add op for merging summary
summary_op = tf.summary.merge_all()

# add Saver ops
saver = tf.train.Saver(max_to_keep=0)

##############################################################################
NL=0
stepL=0
n_epochs = 1000
n_steps = 550  # int(Nk)
#batch_size = 100  # reconstructed images better with 10 than with 100, for n_epochs=10 and n_steps=1000

sess = tf.InteractiveSession()

#pdb.set_trace()
print("Initializing parameters")
sess.run(tf.global_variables_initializer())

summary_writer = tf.summary.FileWriter('experiment', graph=sess.graph)
test_data = mnist.test.images

filename = "save/model_xavier_epoch=" + str(NL) + "_step=" + str(stepL) + ".ckpt.index"
path=os.getcwd()#get absolute path
full_filename=os.path.join(path, filename)
#pdb.set_trace()
if os.path.isfile(full_filename):
    print("Restoring saved parameters from epoch=",NL," and step=",stepL)
    saver.restore(sess, "save/model_xavier_epoch=" + str(NL) + "_step=" + str(stepL) + ".ckpt")
    #saver.restore(sess, "save/model.ckpt")



'''
##############################################################################
#Create a dataset that has only one type of digit
batch_size_one_digit=1*10**4
Nrec = 1
Nk=0
dataset=[]

#mnist = input_data.read_data_sets('MNIST')

for i in range(0,batch_size_one_digit,1):
    batch = mnist.train.next_batch(Nrec)
    x_batch = batch[0]  # batch[1] are the labels
    label_batch=batch[1]
    #pdb.set_trace()
    if label_batch[0]==3:
        Nk=Nk+1
        dataset.append(x_batch)

dataset2=np.reshape(dataset,[-1,784])
##############################################################################
'''
#n_steps = 500
#pdb.set_trace()
for N in range(1, n_epochs+1, 1):
    for step in range(1, n_steps+1,1):
        batch = mnist.train.next_batch(batch_size)
        x_batch = batch[0]
        #x_batch = np.reshape(dataset2[step],[1,784])
        #pdb.set_trace()
        feed_dict = {x: x_batch}

        #print("\n Before training: \n")
        #bt_np, at_np, term1_np, b_np, mu_decoder_np, dawson_inf_np, dawson_sup_np, h_z_np, mu_z_np = sess.run(
        #    [bt, at, term1, b, mu_decoder, dawson_inf, dawson_sup, h_z, mu_z], feed_dict=feed_dict)

        #z_np, logpdf_qzx_np, logpdf_pz_np, h_z1_np, h_z2_np, h_z_np, sum_h_z1_np, Rc_np = sess.run(
        #    [z, log_pdf_qzx, log_pdf_pz, h_z1, h_z2, h_z, sum_h_z1, Renyi_cdf], feed_dict=feed_dict)

        #pdb.set_trace()
        #_, train_loss= sess.run([train_step, loss], feed_dict=feed_dict)

        #train_step_gradients_np, train_step_variables_np, loss_np = sess.run([train_step_gradients, train_step_variables, loss], feed_dict=feed_dict)
        _, train_loss = sess.run([train_step,loss], feed_dict=feed_dict)

        #_,clipped_grads_vars_np, train_loss = sess.run(
        #    [train_step,clipped_grads_vars, loss], feed_dict=feed_dict)
        #grad = clipped_grads_vars_np[0]
        #clipped_grads_vars_np, train_loss = sess.run(
        #    [clipped_grads_vars, loss], feed_dict=feed_dict)
        #grad = clipped_grads_vars_np[0]

        #bt_np, at_np,term1_np,b_np,mu_decoder_np, dawson_inf_np, dawson_sup_np,h_z_np,mu_z_np = sess.run([bt,at,term1,b,mu_decoder,dawson_inf,dawson_sup,h_z,mu_z], feed_dict=feed_dict)
        #mu_z_samples_np,h_z2_np = sess.run([mu_z_samples,h_z2], feed_dict=feed_dict)

        #summary_writer.add_summary(summary_str, step)
        #print("\n Test np.isnan(np.nan)=",np.isnan(np.nan)) - This is ok
        #print("\n Test pd.isnull(grad)=",pd.isnull(grad)) - This is ok

        #if (pd.isnull(grad)):
        #    print("\n Nan in the gradients")
        #    pdb.set_trace()
        #pdb.set_trace()
        if (N % 100 == 0) & (step % 550 == 0):
            print("\n Epoch N=",N)   
        if ( N % 100 == 0) & (step % 550 == 0):#if (NL+N  == 9) & (step % 1 == 0):#(stepL+step > 899): #| (step % 50 == 0): (NL+N == 9) &
            print("\n After training: \n")
            bt_np, at_np, b_np, a_np, mu_decoder_np, std_decoder_np,dawson_sup_np, h_z_np, mu_z_np,elem1_np, elem2_np, log_Id_np, term1_np, term1_2_np, term2_np, rez_np = sess.run(
                [bt, at, b, a, mu_decoder, std_decoder,dawson_sup, h_z, mu_z,elem1, elem2, log_Id, term1, term1_2, term2, rez], feed_dict = feed_dict)

            z_np, logpdf_qzx_np, logpdf_pz_np, h_z1_np, h_z2_np, h_z_np, sum_h_z1_np, Rc_np,loss_np = sess.run(
                [z, log_pdf_qzx, log_pdf_pz, h_z1, h_z2, h_z, sum_h_z1, Renyi_cdf,loss], feed_dict = feed_dict)

            #print("\n np.min(term_sup_np - term_inf_np)=", np.min(term_sup_np-term_inf_np))
            #print("\n train_loss=",train_loss)
            print("Epoch {0} Step {1} | Loss_train: {2} | h_z2: {3} alpha {4}".format(NL+N, stepL+step, train_loss,h_z2_np,alpha))
            '''
            print("\n np.min(grad[0])=",np.min(grad[0]))
            print("\n np.max(grad[0])=", np.max(grad[0]))

            print("\n np.min(grad[1])=", np.min(grad[1]))
            print("\n np.max(grad[1])=", np.max(grad[1]))
            '''

            #path = os.getcwd()
            #full_filename = os.path.join(path, "save/model.ckpt")
            #save_path = saver.save(sess, full_filename)

            #print("\n Saving model at epoch=",N,"and step=",step)

            #pdb.set_trace()

            path=os.getcwd()
            full_filename=os.path.join(path, "save/model_xavier_epoch=" + str(NL+N) + "_step=" + str(step) + ".ckpt")
            #save_path = saver.save(sess, full_filename)

            #pdb.set_trace()
            '''
            ################################################################################################
            #With the trained model, estimate log(p(x)), to have a comparison accross other implementations
            Nimp=1000
            #draw Nimp samples from q(z|x)
            #std_encoder_samples_np, mu_encoder_samples_np = sess.run([std_encoder_samples, mu_encoder_samples],feed_dict=feed_dict)
            log_int_px = 0
            #z_np,mu_prior_np = sess.run([z,mu_prior], feed_dict=feed_dict)
            term3 = np.zeros(shape=[Nimp,batch_size])
            for k in range(0,Nimp,1):
                log_qzx_np,log_pz_np,log_pxz_np,qzx_np,pz_np,pxz_np,log_cdf2_pxz_np = sess.run([log_qzx,log_pz,log_pxz,qzx,pz,pxz,log_cdf2_pxz],feed_dict=feed_dict)
                term1 = np.sum(log_pz_np-log_qzx_np,1)
                #term2 = np.sum(log_pxz_np,1)

                term2 = np.sum(log_cdf2_pxz_np, 1)

                term3[k] = term1 + term2

            #pdb.set_trace()
            a = np.max(term3,0)
            a_tile = np.tile(a,Nimp)
            a_tile = a_tile.reshape([Nimp,batch_size])
            term4 = np.exp(term3 - a_tile)
            log_int_px += a + np.log(np.sum(term4))-np.log(Nimp)

            log_int_px_batch = np.mean(log_int_px,0)

            print("\n Importance sampling estimation of log_int_px_batch=",log_int_px_batch)
            #pdb.set_trace()
            #END With the trained model, estimate log(p(x)), to have a comparison accross other implementations
            ################################################################################################
            '''
            fig2 = plt.figure(figsize=(40, 40))
            plt.gray()
            N_rec = 7
            for i in range(0, N_rec ** 2, 1):
                # batch = mnist.test.next_batch(batch_size)
                # x_batch = batch[0]
                # data_input = x_batch[0]

                x_batch = test_data[i]
                # x_batch = np.reshape(dataset2[i], [1, 784])

                #pdb.set_trace()
                data_input = x_batch
                data2_input = np.reshape(data_input, [28, 28])

                feed_dict = {x: np.reshape(data_input, [-1, 784])}
                # feed_dict = {x: data_input}

                mu_decoder_rec,std_decoder_rec,mu_encoder_np,std_encoder_np = sess.run([mu_decoder,std_decoder,mu_encoder,std_encoder], feed_dict=feed_dict)

                data = np.reshape(mu_decoder_rec, [1, -1])

                tmp = np.reshape(data, [n_samples, 1, -1])
                mean_tmp = np.mean(tmp, 0)

                data2 = np.reshape(mean_tmp, [28, 28])

                aux=np.abs(data2_input-data2)
                #plt.imshow(aux)
                #plt.show()

                fig2.add_subplot(N_rec, N_rec, i + 1)
                plt.imshow(data2)

                #pdb.set_trace()

            #plt.suptitle("Reconstructed test MNIST images, using the Renyi cdf bound and the mu sigmoid and the std exp", fontsize=40)
            fig2.savefig('Reconstructed_test_MNIST_with_renyi_cdf_mu_sigmoid_std_exp.png')

            #fig2.show()
            print("\n Figure done")
            #pdb.set_trace()

            #Get data ready

            # #Create image generator from class
            # #path = os.getcwd()
            # full_filename = "Argo_Reconstructed_test_MNIST_with_renyi_cdf_mu_sigmoid_std_exp.png"
            #
            # im=ImagesGenerator(dirName=".",fileName=full_filename,n_images_columns=14,n_images_rows=2)
            # im.image_size = (28,28)
            #
            # width = im.number_images_columns
            # height = im.number_images_rows
            # #X = data_set[:width * height]
            # X = test_data[:width * height]
            #
            # data_input = X
            # feed_dict = {x: np.reshape(data_input, [-1, 784])}
            #
            # mu_decoder_rec, std_decoder_rec, mu_encoder_np, std_encoder_np = sess.run(
            #     [mu_decoder, std_decoder, mu_encoder, std_encoder], feed_dict=feed_dict)
            #
            # #X_sampled = np.random.normal(mu_decoder_rec, std_decoder_rec,size=X.shape)
            # X_sampled = mu_decoder_rec
            #
            # X_mu = sess.run(mu_decoder, feed_dict={z: mu_encoder_np})
            #
            # #pdb.set_trace()
            # #means = mu_encoder_np
            # #cov = std_encoder_np
            # #z = z_np
            # #means, cov, z = self.encode(X)
            # images_mu = X_mu
            # images_z = X_sampled
            # #images_z = mu_decoder_rec
            #
            # height = height * 3
            #
            # composite = np.zeros((X.shape[0] * 3, X.shape[1]))
            # for i in range(0, int(height), 3):
            #     composite[int(i * width):int((i + 1) * width)] = X[int(i / 3 * width):int((i / 3 + 1) * width)]
            #     composite[int((i + 1) * width):int((i + 2) * width)] = images_mu[
            #                                                            int(i / 3 * width):int((i / 3 + 1) * width)]
            #     composite[int((i + 2) * width):int((i + 3) * width)] = images_z[
            #                                                            int(i / 3 * width):int((i / 3 + 1) * width)]
            #
            # im.save_images(composite, N, width=width, height=height, title="")
            # print("\n Figure done with Argo")

            # '''
            # generate images
            Ngen = N_rec ** 2
            z_prior_np = np.random.normal(0, 1, size=(Ngen, latent_dim))
            mu_decoder_gen = sess.run(mu_decoder, feed_dict={z: z_prior_np})

            fig3 = plt.figure(figsize=(40, 40))
            plt.gray()
            for i in range(0, N_rec ** 2, 1):
                gen_img = mu_decoder_gen[i]
                gen_img = np.reshape(gen_img, [28, 28])
                fig3.add_subplot(N_rec, N_rec, i + 1)
                plt.imshow(gen_img)

            plt.suptitle("Generated MNIST images, using the Gaussian cdf bound and the constrained encoder",
                         fontsize=40)
            fig3.savefig('Generated_MNIST_with_gaussian_cdf_enc_constrained.png')
            #plt.show()

            '''
            pdb.set_trace()

            dim_test = test_data.shape[0]
            std_reconstr_train = sess.run(std_decoder, feed_dict=feed_dict)
            std_reconstr_train = np.mean(np.reshape(std_reconstr_train, (n_samples, batch_size, -1)), axis=0)
            max_std_reconstr_train = np.max(std_reconstr_train)
            idx_max_train = np.unravel_index(np.argmax(std_reconstr_train), std_reconstr_train.shape)
            min_std_reconstr_train = np.min(std_reconstr_train)
            idx_min_train = np.unravel_index(np.argmin(std_reconstr_train), std_reconstr_train.shape)

            test_loss = 0
            #loss_test_reconstruction_np = 0
            #loss_test_KLD_np = 0

            std_reconstr_test = None
            for i in range(int(dim_test / batch_size) - 1):
                pdb.set_trace()
                test_batch = test_data[i:i + batch_size]
                test_loss += sess.run(loss, feed_dict={x: test_batch})

                #loss_test_reconstruction_np += sess.run(loss_reconstruction, feed_dict={x: test_batch})
                #loss_test_KLD_np += sess.run(loss_KLD, feed_dict={x: test_batch})

                std_reconstr_test_batch = sess.run(std_decoder, feed_dict={x: test_batch})
                std_reconstr_test_batch = np.mean(np.reshape(std_reconstr_test_batch, (n_samples, batch_size, -1)),
                                                  axis=0)
                if i == 0:
                    std_reconstr_test = std_reconstr_test_batch
                else:
                    std_reconstr_test = np.vstack([std_reconstr_test, std_reconstr_test_batch])

                max_std_reconstr_test = np.max(std_reconstr_test)
                idx_max_test = np.unravel_index(np.argmax(std_reconstr_test), std_reconstr_test.shape)
                min_std_reconstr_test = np.min(std_reconstr_test)
                idx_min_test = np.unravel_index(np.argmin(std_reconstr_test), std_reconstr_test.shape)

            test_loss = test_loss * batch_size / dim_test
            #loss_test_reconstruction_np = loss_test_reconstruction_np * batch_size / dim_test
            #loss_test_KLD_np = loss_test_KLD_np * batch_size / dim_test
            '''
            path = os.getcwd()
            full_filename = os.path.join(path, "save/model.ckpt")

            save_path = saver.save(sess, full_filename)

            #pdb.set_trace()
            #print("Epoch {0} Step {1} | Loss_train: {2} Loss_test: {3}".format(N, step, train_loss, test_loss))

            #print("Epoch {0} Step {1} | Loss_train: {2} Loss_test: {3}".format(N, step, train_loss, test_loss))
            #print("Loss_train_reconstruction=", loss_train_reconstruction_np, "Loss_train_KLD=", loss_train_KLD_np)
            #print("Loss_test_reconstruction=", loss_test_reconstruction_np, "Loss_test_KLD=", loss_test_KLD_np)

            # print("max_std_reconstr_train: ", max_std_reconstr_train, " pixel: ", x_batch[idx_max_train])
            # print("min_std_reconstr_train: ", min_std_reconstr_train, " pixel: ", x_batch[idx_min_train])
            # print("max_std_reconstr_test: ", max_std_reconstr_test, " pixel: ", test_data[idx_max_test])
            # print("min_std_reconstr_test: ", min_std_reconstr_test, " pixel: ", test_data[idx_min_test])
            print("========================================================")


'''
######################################################################################
#Test the algorithm
test_data = mnist.test.images
filename="save/model.ckpt.index"

path=os.getcwd()#get absolute path
full_filename=os.path.join(path, filename)
#pdb.set_trace()
if os.path.isfile(full_filename):
    print("Restoring saved parameters")
    #saver.restore(sess, "save/model.ckpt")

    print("\n Test the quality of the reconstruction: \n")

    test_data = mnist.train.images

    for i in range(0, 20, 1):
        # batch = mnist.test.next_batch(batch_size)
        # x_batch = batch[0]
        # data_input = x_batch[0]

        x_batch = test_data[i]
        data_input = x_batch
        data2_input = np.reshape(data_input, [28, 28])

        feed_dict = {x: np.reshape(data_input, [-1, 784])}
        # feed_dict = {x: data_input}

        mu_decoder_rec = sess.run([mu_decoder], feed_dict=feed_dict)

        data = np.reshape(mu_decoder_rec, [1, -1])

        tmp = np.reshape(data, [n_samples, 1, -1])
        mean_tmp = np.mean(tmp, 0)

        data2 = np.reshape(mean_tmp, [28, 28])

        fig2, (ax21, ax22) = plt.subplots(2, 1, figsize=(24, 24))
        plt.gray()
        ax21.imshow(data2)
        ax22.imshow(data2_input)
        fig2.show()
        #fig2.savefig('Ongoing_Img_original_reconstructed_i=' + str(i) + '.png')

# generate images
    Ngen = 20
    z_prior_np = np.random.normal(0, 1, size=(Ngen, latent_dim))
    mu_decoder_gen = sess.run(mu_decoder, feed_dict={z: z_prior_np})
    for i in range(0, Ngen, 1):
        gen_img = mu_decoder_gen[i]
        gen_img = np.reshape(gen_img, [28, 28])
        fig2, ax21 = plt.subplots(1, 1, figsize=(24, 24))
        plt.gray()
        ax21.imshow(gen_img)
        fig2.show()
        pdb.set_trace()

else:
    print("\n No model saved: needs training \n")
'''
