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
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

EPS = 0#.00005

def main(directory,
         compute_loglikelihood,
         n_test_data,
         n_ais_step,
         eps,
         n_hast_step,
         n_ham_step,
         n_prior_samp):

    params = utils.fetch_file(directory + "model_params")

    for k in params.keys():
        print("{}:\t{}".format(k,params[k]))
    netparams = prep(params)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    CP = read_ckpt_file(directory + "saved_params/checkpoint")
    Wcp = []
    for c in CP:
        print("Saving weights for cp-file {}".format(c))
        saver.restore(sess,directory + "saved_params/" + c)

        W = sess.run(netparams["weights"])
        Wcp.append(W)
        if params["pca_truncation"] == "cut":
            np.savetxt(directory + "{}_weights.csv".format(c),W.transpose())#netparams["PCA"].inverse_transform(W.transpose()))
        elif params["pca_truncation"] == "smooth":
            np.savetxt(directory + "{}_weights.csv".format(c),W.transpose())

    sess.close()
    if compute_loglikelihood:
        print("Computing loglikelihood.")
        AISout = np.array([calc_log_likelihood(netparams["testdat"][:n_test_data],w,params,netparams,n_prior_samp = n_prior_samp,n_ais_step = n_ais_step,eps = eps,n_hast_step=n_hast_step,n_ham_step = n_ham_step) for w in Wcp])

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
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("directory",help = "Relative path to directory containing the trained model you'd like to analyze.")
    
    parser.add_argument("--compute_loglikelihood",action="store_true",help = "Run Annealed Importance Sampling to compute the loglikelihood.")
    parser.add_argument("--n_test_data",default = 50,help = "number of test data samples to use for loglikelihood.")
    parser.add_argument("--n_ais_step",default = 10000,help = "number of AIS steps to take.")
    parser.add_argument("--eps",default = .1,help = "epsilon parameter for AIS.")
    parser.add_argument("--n_hast_step",default = 10,help = "number of MCMC steps to take for AIS.")
    parser.add_argument("--n_ham_step",default = 3,help = "number of steps to take for hamiltonian monte-carlo.")
    parser.add_argument("--n_prior_samp",default = 50,help = "number of samples to use for AIS.")

    args = vars(parser.parse_args())

    #arg: loss, dataset, decoder (deep or shallow)

    main(**args)
