import time
import sys
import numpy as np
import lasagne as L
import theano
import theano.tensor as T
import pickle
import logclass.log as log

import svae.decoder.decoder as dec
import svae.encoder.encoder as enc
import svae.variance.variance as variance
import svae.hastings.hasting as AIS

import gaussian_comp as gcomp

import svae.data.get_data as dat
import svae.data.image_processing.test_gratings as grat

import svae.losses.make_loss as make_loss
import svae.distributions.distributions as distributions

from skimage.filters import gaussian as GFILT

import os

import shutil
import utilities as utils
def run(dirname,save_weights,test_gratings,RF_comp,test_loglik,train_loglik,plot_loglik,plot_train_loglik,save_test_latents,n_data_samp,n_ais_step,n_prior_samp,n_hast_step,eps,n_ham_step,use_prior,full,fast,seed,AIS_test):

    np.random.seed(seed)

    LOG = log.log(dirname + "/analysis_log.log")

    MP = utils.load_obj(dirname +"model_params")

    n_pca = int((MP["patch_size"] ** 2)*MP["pca_frac"])

    #this is to handle legacy data files that didn't have the CNN keyword

    if "CNN" not in MP.keys():
        MP["CNN"] = False

    if MP["CNN"]:
        datsize = MP["patch_size"]**2
    else:
        datsize = n_pca
        
    n_lat = int(n_pca * MP["overcomplete"])

    MP["n_lat"] = n_lat
    MP["n_pca"] = n_pca
    MP["datsize"] = datsize
    MP["dirname"] = dirname

    for x in MP.keys():
        print("{}\t{}".format(x,MP[x]))

    train,test,var,PCA = dat.get_data(MP["patch_size"],n_pca,MP["dataset"],MP["whiten"],MP["CNN"])

    LOG.log("Train Shape:\t{}".format(train.shape))
    LOG.log("Test Shape:\t{}".format(test.shape))
    LOG.log("Var Shape:\t{}".format(var.shape))

    W = get_weights(MP)
    
    try:
        Wf = get_weights(MP,"decoder_params_final")
        FINAL = True
    except:
        LOG.log("Final params not available")
        FINAL = False
        
    if save_weights or full or fast:

        #W[0] is [144,NLAT]. teh PCA var is size n_pca. I want to take the PCA var, inverse transform it, and them normalize by it.
        if MP["CNN"]:
            w_norm = np.sqrt(np.reshape(PCA.explained_variance_,[1,-1]))
            Wout = PCA.inverse_transform(PCA.transform(np.transpose(W[0]))*w_norm)
        else:
            Wout = PCA.inverse_transform(np.transpose(W[0]))
        
        LOG.log("Saving Weights")        
        np.savetxt(MP["dirname"] + "weights.csv",Wout)
        
        if FINAL:
            if MP["CNN"]:
                w_norm = np.sqrt(np.reshape(PCA.explained_variance_,[1,-1]))
                Wout = PCA.inverse_transform(PCA.transform(np.transpose(Wf[0]))*w_norm)
                #w_norm = PCA.inverse_transform(np.sqrt(np.reshape(PCA.explained_variance_,[1,-1])))
                #Wout = np.transpose(Wf[0])*w_norm
            else:
                Wout = PCA.inverse_transform(np.transpose(W[0]))
                
            np.savetxt(MP["dirname"] + "weights_final.csv",Wout)
            
    if save_test_latents or full or fast:
        LOG.log("Saving Latents")
        mean,var,trans = get_latents(test[:np.min([10*n_data_samp,len(test)])],MP,W,PCA,SAVE = True)
        trans1 = np.array([np.diag(x) for x in trans])
        trans2 = trans[:,0,:]
        
        np.savetxt(MP['dirname'] + "test_means_best.csv",mean)
        np.savetxt(MP['dirname'] + "test_sample_best.csv",np.array([np.random.multivariate_normal(mean[v],var[v]) for v in range(len(var))]))
        np.savetxt(MP['dirname'] + "test_trans_diag_best.csv",trans1)
        np.savetxt(MP['dirname'] + "test_trans_trans_best.csv",trans2)

        if FINAL:
            mean,var,trans = get_latents(test[:np.min([10*n_data_samp,len(test)])],MP,Wf,PCA,SAVE = True)
            trans1 = np.array([np.diag(x) for x in trans])
            trans2 = trans[:,0,:]

            np.savetxt(MP['dirname'] + "test_means_final.csv",mean)
            np.savetxt(MP['dirname'] + "test_trans_diag_final.csv",trans1)
            np.savetxt(MP['dirname'] + "test_trans_trans_final.csv",trans2)

        if MP["CNN"]:

            norm = np.sqrt(np.reshape(PCA.explained_variance_,[1,-1]))
            out = PCA.inverse_transform(PCA.transform(test[:np.min([10*n_data_samp,len(test)])])*w_norm)
        else:
            out = PCA.inverse_transform(test[:np.min([10*n_data_samp,len(test)])])
            
        np.savetxt(MP["dirname"] + "test_images.csv",out)#PCA.inverse_transform(test[:np.min([n_data_samp,len(test)])]))

    if test_gratings or full or fast:
        LOG.log("Processing Gratings")
        mean,lab,grats = grating_test(MP,PCA)
   
        np.savetxt(MP["dirname"] + "test_grating.csv",mean)
        np.savetxt(MP["dirname"] + "test_grating_labels.csv",lab)
        np.savetxt(MP["dirname"] + "test_grating_images.csv",grats)

    if RF_comp or full or fast:
        LOG.log("Calculating RFs")
        for scale in [.4,.5,.6]:
            RFs = RF_test(MP,PCA,scale)
            for k in range(len(RFs)):
                np.savetxt(MP["dirname"] + "receptive_fields_{}_{}.csv".format(k,scale),RFs[k])

    if test_loglik or full:
        LOG.log("Calculating Likelihoods")
        plot_loglikelihood(test[:np.min([n_data_samp,len(test)])],MP,"test_final_loglik.csv",indices = ["best"],n_ais_step = n_ais_step,n_prior_samp = n_prior_samp,n_hast_step = n_hast_step,eps = eps,n_ham_step = n_ham_step,use_prior = use_prior,LOG = LOG)

    if train_loglik or full:
        LOG.log("Calculating Likelihoods")
        plot_loglikelihood(train[:np.min([n_data_samp,len(train)])],MP,"train_final_loglik.csv",indices = ["best"],n_ais_step = n_ais_step,n_prior_samp = n_prior_samp,n_hast_step = n_hast_step,eps = eps,n_ham_step = n_ham_step,use_prior = use_prior,LOG = LOG)

    if plot_loglik or full:
        LOG.log("Plotting Likelihoods")
        plot_loglikelihood(test[:np.min([n_data_samp,len(test)])],MP,"test_plot_loglik.csv",n_ais_step = n_ais_step,n_prior_samp = n_prior_samp,n_hast_step = n_hast_step,eps = eps,n_ham_step = n_ham_step,use_prior = use_prior,LOG = LOG)
        
    if plot_train_loglik or full:
        LOG.log("Plotting Likelihoods")
        plot_loglikelihood(train[:np.min([n_data_samp,len(test)])],MP,"train_plot_loglik.csv",n_ais_step = n_ais_step,n_prior_samp = n_prior_samp,n_hast_step = n_hast_step,eps = eps,n_ham_step = n_ham_step,use_prior = use_prior,LOG = LOG)

    if AIS_test:
        test_loglikelihood(test[:2],MP,"best",n_ais_step,n_prior_samp,n_hast_step,eps,n_ham_step,use_prior,LOG)

def get_weights(MP,param_name = "decoder_params_best"):
    W=utils.load_obj(MP["dirname"] + param_name)
    return W
                    
def get_decoder(params,MP,var = -1):

    if var == -1:
        lat_mean = T.matrix("latmean","float32")
    else:
        lat_mean = var
    
    rec, weights= dec.make_decoder(lat_mean,0.,0.,MP["n_lat"],MP["datsize"],MP["n_batch"],MP["decoder"],MP["whiten"])
    
    for k in range(len(weights)):
        weights[k].set_value(params[k])
        
    return rec

        
def plot_loglikelihood(data,MP,name,n_ais_step,n_prior_samp,n_hast_step,eps,n_ham_step,use_prior,LOG,indices = -1):
    LOG.log("getting all likelihoods:\t{}".format(name))
    
    if indices == -1:
        
        ind = MP["param_save_freq"] * (2**np.arange(0,np.floor(np.log2(MP["n_grad_step"]/MP["param_save_freq"])) + 1))
        ii = np.int32(np.concatenate([np.array([0]),ind,np.array([MP["n_grad_step"]])]))
        LOG.log("Loglikelihood plot points: {}".format(ii))
        
        '''
        res = MP["n_grad_step"] / 10.
        
        nstep = MP["param_save_freq"] * int(res / MP["param_save_freq"])
        LOG.log("Params sampled every {} steps.".format(nstep))

        i1 = range(0,nstep,MP["param_save_freq"])
        i2 = range(nstep,MP["n_grad_step"],nstep)

        ii = i1 + i2
        
        #ii = range(0,MP["n_grad_step"],nstep)
        '''
        
    else:
        ii = indices
    final = []

    print(ii[:5])

    for k in ii[:5]:
        LOG.log("Using "+"decoder_params_{}".format(k))
        #try:
        W = get_weights(MP,"decoder_params_{}".format(k))
        lik = calc_log_likelihood(data,W,MP,LOG,n_ais_step,n_prior_samp,n_hast_step,eps,n_ham_step,use_prior)
        final.append(lik)
        #except:
        #    print("something bad happened,skipping to next file")
        #    continue
        
    np.savetxt(MP["dirname"] + name,np.array(final))

def get_latents(data,MP,W,PCA,param_name = "encoder_params_best",SAVE = True):
    #I need to load the encoder
    images = T.matrix("images","float32")

    mean = []
    var = []
    trans = []
    
    enc_func = load_encoder(images,MP,100,param_name = param_name)

    for k in range(0,len(data),100):
        m,v,t = enc_func(data[k:k+100])
        mean.append(m.copy())
        var.append(v.copy())
        trans.append(t.copy())

    mean = np.concatenate(mean)
    var = np.concatenate(var)
    trans = np.concatenate(trans)
    
    return mean,var,trans

def grating_test(MP,PCA,param_name = "encoder_params_best"):
    gratings = [[grat.GRATS(c,a,k,s,12),grat.GRATC(c,a,k,s,12)]
                for a in np.linspace(0,np.pi,10)
                for c in np.linspace(.05,1,5)
                for k in np.linspace(2,12,10)
                for s in np.linspace(0,12,10)]

    labels = [[[0,a,c,k,s],[1,a,c,k,s]]
                for a in np.linspace(0,np.pi,10)
                for c in np.linspace(.05,1,5)
                for k in np.linspace(2,12,10)
                for s in np.linspace(0,12,10)]

    G = np.reshape(np.array(gratings),[-1,12*12])
    L = np.reshape(np.array(labels),[-1,4])


    if MP["CNN"]:
	Gtrans = dat.get_CNN_dat(PCA.transform(G),PCA,MP["whiten"])
    else:
	Gtrans = PCA.transform(G)
    images = T.matrix("images","float32")
    
    enc_func = load_encoder(images,MP,100,param_name = param_name)
    out = []
    
    for k in range(0,len(Gtrans),100):
        out.append(enc_func(Gtrans[k:k+100])[0])

    out = np.concatenate(out,axis = 0)

    return out,L,G

def RF_test(MP,PCA,scale,param_name = "encoder_params_best",nsam = 100):

    images = T.matrix("images","float32")
    
    enc_func,n = encoder_response_function(images,MP,nsam,param_name = param_name)
    #we want to take the weighted sum of the inputs

    l_out = [[] for l in range(n)]
    
    for index in range(100):
        noise = np.random.uniform(0,1,[nsam,MP["patch_size"],MP["patch_size"]])
        noise = np.array([GFILT(x,scale,mode = 'wrap') for x in noise])
        noise = np.reshape(noise,[nsam,MP["patch_size"]*MP["patch_size"]])

        if MP["CNN"]:
            INP = dat.get_CNN_dat(PCA.transform(noise),PCA,MP["whiten"])
        else:
            INP = PCA.transform(noise)            
    
        out = enc_func(INP)#size [n_layer,n_samp,n_neuron]
            
        n_RS = np.reshape(noise,[-1,1,12,12])
        o_RS = [np.reshape(out[k],[out[k].shape[0],out[k].shape[1],1,1]) for k in range(len(out))]

        for k in range(len(l_out)):
            l_out[k].append((n_RS*o_RS[k]).mean(axis = 0))

    n_RS = [np.reshape(np.array(l_out[k]).mean(axis = 0),[-1,MP["patch_size"]**2]) for k in range(len(l_out))]

    return n_RS

def load_encoder(images,MP,nbatch,param_name = "encoder_params_best"):

    lat_mean_layer,lat_var_layer,out = enc.make_encoder([nbatch,MP["datsize"]],MP["n_lat"],MP["MVG"],MP["n_gauss_dim"],MP["CNN"])

    #load parameters
    enc_parameters = utils.load_obj(MP["dirname"] + param_name)

    L.layers.set_all_param_values(out,enc_parameters)

    if MP["MVG"]:
        lat_var = [L.layers.get_output(k,inputs = images) for k in lat_var_layer]
    else:
        lat_var = L.layers.get_output(lat_var_layer,inputs = images)

    var,trans = variance.get_var_mat(lat_var,MP["MVG"])
    mean = L.layers.get_output(lat_mean_layer,inputs = images)
    #so I need to get the variance and mean

    lfunc = theano.function([images],[mean,var,trans],allow_input_downcast = True)

    return lfunc

def get_encoder_parameters(MP,param_name = "encoder_params_best"):

    enc_parameters = utils.load_obj(MP["dirname"] + param_name)

    return [param.get_value() for param in enc_parameters]

def encoder_response_function(images,MP,nbatch,param_name = "encoder_params_best"):

    lat_mean_layer,lat_var_layer,out = enc.make_encoder([nbatch,MP["datsize"]],MP["n_lat"],MP["MVG"],MP["n_gauss_dim"],MP["CNN"])

    #load parameters
    enc_parameters = utils.load_obj(MP["dirname"] + param_name)

    L.layers.set_all_param_values(out,enc_parameters)
    
    intermediate = L.layers.get_all_layers(lat_mean_layer)
    int_resp = [L.layers.get_output(l,inputs = images) for l in intermediate]
    
    lfunc = theano.function([images],int_resp,allow_input_downcast = True,on_unused_input = 'ignore')
    
    return lfunc,len(int_resp)

def test_loglikelihood(data,MP,name,n_ais_step,n_prior_samp,n_hast_step,eps,n_ham_step,use_prior,LOG):
    LOG.log("getting all likelihoods:\t{}".format(name))

    for k in [100,200,400]:
        for seed in [1,2]:
            np.random.seed(seed)
            LOG.log("Using "+"decoder_params_{}".format("best"))
            W = get_weights(MP,"decoder_params_{}".format("best"))
            LOG.log("nham: {}\tseed: {}".format(k,seed))
            lik = calc_log_likelihood(data,W,MP,LOG,n_ais_step = 200,n_prior_samp = k,n_hast_step = 2,eps = eps,n_ham_step = 20,use_prior = use_prior)            

def calc_log_likelihood(DATA,W,MP,LOG,n_ais_step = 200,n_prior_samp = 200,n_hast_step = 2,eps = .1,n_ham_step = 20,use_prior = False):

    A = W[0]
    p_var = np.dot(np.transpose(A),A)/(MP["sigma"]**2)

    def p_mean(x):
        temp = np.dot(np.transpose(A),x)
        return np.dot(np.linalg.pinv(np.dot(np.transpose(A),A)),temp)

    
    #now we get the variational posterior
    images = T.matrix("images","float32")

    lfunc = load_encoder(images,MP,1,param_name = "encoder_params_best")

    #I am going to use this to keep track of my latents in AIS
    latent = T.matrix("latents","float32")
        
    #now I can go through each test images and calculate log probs
    output = []

    TPT = 0
    tfrac = .9
    for k in range(len(DATA)):
        t1 = time.time()
        data = np.array([DATA[k]])
        if MP["loss_type"] == "gauss" and False:            
            w = gcomp.get_LL(W[0],data[0],MP["sigma"])
            
            print(w)
            output.append(w)
            
        else:
            
            m,v,t = lfunc(data)
            
            m = m[0]
            v = v[0]
            t = t[0]
 
            if use_prior:
                L_prior,g_L_prior,samp = distributions.get_distribution(MP["loss_type"],MP)
                nl_var = -L_prior(latent.dimshuffle([0,'x',1])).mean(axis = 1)# we need to dimshuffle becuase L_prior takes a 3-D tensor.
                g_nl_var = -g_L_prior(latent.dimshuffle([0,'x',1])).mean(axis = 1)
                
            else:
                nl_var = neglog_var_post(latent,p_mean(data[0]),np.linalg.inv(v),MP["n_lat"])
                g_nl_var = g_neglog_var_post(latent,p_mean(data[0]),np.linalg.inv(v),MP["n_lat"])

            if True:
                nl_var = neglog_var_post(latent,p_mean(data[0]),np.linalg.inv(p_var + np.identity(len(p_var))),MP["n_lat"])
                g_nl_var = g_neglog_var_post(latent,p_mean(data[0]),np.linalg.inv(p_var + np.identity(len(p_var))),MP["n_lat"])

            nl_pos = neglog_tru_post(latent,p_mean(data[0]),p_var,MP["sigma"],MP["n_lat"],MP)
            g_nl_pos = g_neglog_tru_post(latent,p_mean(data[0]),p_var,MP["n_lat"],MP)
            
            nl_var_func = theano.function([latent],nl_var,allow_input_downcast = True)
            nl_pos_func = theano.function([latent],nl_pos,allow_input_downcast = True)
            
            g_nl_var_func = theano.function([latent],g_nl_var,allow_input_downcast = True)
            g_nl_pos_func = theano.function([latent],g_nl_pos,allow_input_downcast = True)
            
            grad = [g_nl_var_func,g_nl_pos_func]

            if use_prior:
                def var_post_samp(x):
                    num = x[0]
                    
                    return samp(x[0],MP["n_lat"])
            else:
                def var_post_samp(x):
                    n_sample = np.random.multivariate_normal(p_mean(data[0]),v,[x[0]])
                    return n_sample

            if True:
                def var_post_samp(x):
                    n_sample = np.random.multivariate_normal(p_mean(data[0]),p_var + np.identity(len(p_var)),[x[0]])
                    return n_sample
                
            x,w = AIS.AIS(nl_var_func,nl_pos_func,var_post_samp,MP["n_lat"],n_samp = n_prior_samp,n_AIS_step = n_ais_step,nhstep = n_hast_step,eps = eps,grad = grad,L = n_ham_step)
        
            output.append(w.mean())
            
        t2 = time.time()
        
        if TPT == 0:
            TPT = t2 - t1
        else:
            TPT = tfrac * TPT + (1. - tfrac)*(t2 - t1)
            
        LOG.log("Log Lik:\t{}\tHours Left:\t{}".format(w.mean(),np.round((len(DATA) - k - 1)*TPT/(60*60),3)))
    return np.array(output)

def neglog_post_gauss(x,mean,ivar,sig,n):
        
    #note that ivar is the INVERSE variance
    xp = x - T.reshape(mean,[1,-1])

    d = T.tensordot(xp,ivar,axes = [1,1])

    exp = - (xp*d).sum(axis = 1)/2

    N = - (n/2)*T.log(2*np.pi*sig*sig)
    
    return - exp - N

def neglog_var_post(x,mean,ivar,n):
        
    #note that ivar is the INVERSE variance
    xp = x - T.reshape(mean,[1,-1])

    d = T.tensordot(xp,ivar,axes = [1,1])

    exp = - (xp*d).sum(axis = 1)/2

    s,ldet = np.linalg.slogdet(ivar)

    N = - (n/2)*(T.log(2*np.pi)) - (-ldet)/2#- here because it is the INVERSE variance
    
    return - exp - N

def neglog_tru_post(x,mean,ivar,sig,n,MP):

    f,g,d = distributions.get_distribution(MP["loss_type"],MP)
    
    pexp = f(x.dimshuffle([0,'x',1])).mean(axis = 1)
    
    return neglog_post_gauss(x,mean,ivar,sig,n) - pexp

def g_neglog_var_post(x,mean,ivar,n):
    return g_neglog_post_gauss(x,mean,ivar,n)
    
def g_neglog_post_gauss(x,mean,ivar,n):
    xp = x - T.reshape(mean,[1,-1])

    d = - T.tensordot(xp,ivar,axes = [1,1])
    
    return - d

def g_neglog_tru_post(x,mean,ivar,n,MP):
    
    f,g,d = distributions.get_distribution(MP["loss_type"],MP)

    pexp = g(x.dimshuffle([0,'x',1])).mean(axis = 1)

    return g_neglog_post_gauss(x,mean,ivar,n) - pexp
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dirname",help = "Relative path to directory containing the trained model you'd like to analyze.")
    parser.add_argument("--save_weights",action="store_true",help = "Save the weights.")
    parser.add_argument("--test_gratings",action="store_true",help = "Save the latent activities in response to gratings.")
    parser.add_argument("--RF_comp",action="store_true",help = "Compute receptive fields via weighted response to random stimuli.")

    parser.add_argument("--test_loglik",action="store_true",help = "Compute the loglikelihood of the test set.")
    parser.add_argument("--AIS_test",action="store_true",help = "Run an AIS test.")
    parser.add_argument("--train_loglik",action="store_true",help = "Compute the loglikelihood of the training set.")
    parser.add_argument("--plot_loglik",action="store_true",help = "Compute the loglikelihood of the test set at intermediate points in training.")
    parser.add_argument("--plot_train_loglik",action="store_true",help = "Compute the loglikelihood of the training set at intermediate points in training.")
    parser.add_argument("--save_test_latents",action="store_true",help = "Compute the variational latent distributions of the test set.")

    parser.add_argument("--n_data_samp",type = int,help = "Number of data samples to perform analysis on.",default = 100)
    
    parser.add_argument("--n_ais_step",type = int,help = "Number of intermediate distributions to use in annealed importance sampling.",default = 200)
    parser.add_argument("--n_prior_samp",type = int,help = "Number of sampled points to use in annealed importance sampling.",default = 200)
    parser.add_argument("--n_hast_step",type = int,help = "Number of hastings steps to use at each iteration in annealed importance sampling.",default = 2)
    parser.add_argument("--n_ham_step",type = int,help = "Number of hamiltonian dynamical steps to use in each hastings step for annealed importance sampling.",default = 20)
    parser.add_argument("--eps",type = float,help = "Scale of noise to use in hastings for annealed importance sampling.",default = .1)
    parser.add_argument("--seed",type = int,help = "Random seed to initialize the RNGods.",default = 1)

    parser.add_argument("--full",action="store_true",help = "Flag to run all analysis",default = False)
    parser.add_argument("--fast",action="store_true",help = "Flag to run fast analysis (no AIS)",default = False)
    parser.add_argument("--use_prior",action="store_true",help = "Flag to use the prior as the initial distribution in AIS. Otherwise use the variational posterior.",default = False)

    args = vars(parser.parse_args())

    #arg: loss, dataset, decoder (deep or shallow)
    run(**args)
