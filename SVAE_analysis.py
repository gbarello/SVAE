import tensorflow as tf
import numpy as np
import utilities as utils
import sys
from run_sparse_vae import prepare_network as prep
import svae.annealed_importance_sampling.tf_hastings as AIS
import time
import svae.distributions.distributions as dist
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

EPS = 0#.00005

def main(directory):

    params = utils.fetch_file(directory + "model_params")

    for k in params.keys():
        print("{}:\t{}".format(k,params[k]))
    exit()
    netparams = prep(params)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    CP = read_ckpt_file(directory + "saved_params/checkpoint")
    
    for c in CP:
        saver.restore(sess,directory + "saved_params/" + c)

        W = sess.run(netparams["weights"])

        if params["pca_truncation"] == "cut":
            np.savetxt(directory + "{}_weights.csv".format(c),netparams["PCA"].inverse_transform(W.transpose()))
        elif params["pca_truncation"] == "smooth":
            np.savetxt(directory + "{}_weights.csv".format(c),W.transpose())

    sess.close()

    AISout = np.array([calc_log_likelihood(netparams["testdat"][:10],W,params,netparams,n_ais_step = nstep,eps = .2,n_hast_step = 10,n_ham_step = 3) for nstep in range(1000,20000,1000)])

    np.savetxt(directory + "/AIS_loglikelihood.csv",AISout)
    
def read_ckpt_file(f):
    F = open(f,"r")
    CP = []
    for l in F:
        CP.append(l.split(":")[-1].split('"')[1])
    return CP

def calc_log_likelihood(DATA,W,MP,NP,n_ais_step = 10000,n_prior_samp = 50,n_hast_step = 5,eps = .01,n_ham_step = 10,use_prior = False):

    A = W
    p_var = np.float32(EPS * np.eye(NP["n_lat"])) + np.dot(np.transpose(A),A)/(MP["sigma"]**2)
    
#    a = np.random.multivariate_normal(np.zeros(len(p_var)),p_var,100)
    e,v = np.linalg.eig(p_var)
    
    def p_mean(x):
        temp = np.dot(np.transpose(A),x)
        return np.float32(np.dot(np.linalg.pinv(np.dot(np.transpose(A),A)),temp))
        
    #now I can go through each test images and calculate log probs
    output = []

    TPT = 0
    tfrac = .9
    for k in range(len(DATA)):
        t1 = time.time()

        d = DATA[k]

        D1_f,D1_g,D1_samp = dist.get_distribution(MP["loss_type"],params = {})
        #D1_f,D1_g,D1_samp = dist.get_distribution("gauss",params = {})
        prior_f,prior_g,prior_samp = dist.get_distribution(MP["loss_type"],params = MP)        
        
        post_g_f,post_g_g,post_g_samp = dist.get_distribution("special_corgauss",params = {"cov":p_var,"mean":p_mean(d),"special_cov":np.float32(np.eye(len(p_var))*MP["sigma"]**2)})

        latvals = tf.placeholder(tf.float32,[n_prior_samp,NP["n_lat"]])
        #latvals2 = tf.expand_dims(latvals,1)
        
        #init = tf.global_variables_initializer()
        #sess = tf.Session()
        #sess.run(init)

        D1_f_expression = lambda x:-D1_f(x)#[:,0]
        D1_g_expression = lambda x:-D1_g(x)#[:,0]

        print(D1_f_expression(latvals).shape)
        #init_D = lambda x: sess.run(-D1_f_expression,{latvals:x})
        #init_DG = lambda x: sess.run(-D1_g_expression,{latvals:x})

        post_f_expression = lambda x:-(post_g_f(x) + prior_f(x))#[:,0]
        post_g_expression = lambda x:-(post_g_g(x) + prior_g(x))#[:,0]
        
        #post_D = lambda x: sess.run(-post_f_expression,{latvals:x})
        #post_DG = lambda x: sess.run(-post_g_expression,{latvals:x})

        grad = [D1_g_expression,post_g_expression]

        x,w = AIS.AIS(D1_f_expression,post_f_expression,D1_samp,NP["n_lat"],n_samp = n_prior_samp,n_AIS_step = n_ais_step,nhstep = n_hast_step,eps = eps,grad = grad,L = n_ham_step)
        
        output.append(w.mean())
        
        t2 = time.time()
        
        if TPT == 0:
            TPT = t2 - t1
        else:
            TPT = tfrac * TPT + (1. - tfrac)*(t2 - t1)
            
        #        LOG.log("Log Lik:\t{}\tHours Left:\t{}".format(w.mean(),np.round((len(DATA) - k - 1)*TPT/(60*60),3)))
        print("Log Lik:\t{}\tHours Left:\t{}".format(w.mean(),np.round((len(DATA) - k - 1)*TPT/(60*60),3)))

    return np.array(output)

    
if __name__ == "__main__":
    main(sys.argv[1])
