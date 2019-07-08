####################################################################
## import relevant packages

import time                                                         # 
import sys                                                          # 
import numpy as np                                                  # Import basic numerical operations
import pickle                                                       # 
import logclass.log as log                                          # Import INTERNAL library: 
import tensorflow as tf                                             # Import tensorflow for the heavy lifting
from tensorflow.keras import backend as K			    # Import keras backend for flexible memory allocation
import svae.decoder.decoder as dec                                  # Import INTERNAL SVAE library files: the decoder library
import svae.encoder.encoder as enc                                  # Import INTERNAL SVAE library files: the encoder library
import svae.variance.variance as make_var                           # Import INTERNAL SVAE library files: variance calculation code
import svae.data.get_data as dat                                    # Import INTERNAL SVAE library files: data loading library
import svae.losses.make_loss as make_loss                           # Import INTERNAL SVAE library files: loss generation code
import os							    # 
os.environ["CUDA_VISIBLE_DEVICES"]="0"                              # Select GPU:0 on scotty
import shutil                                                       # Import a general set of utilities
import utilities as util                                            # Import additional utilities (Gabriel's personal utils)
 
####################################################################
# GPU options

config = tf.ConfigProto();                                          # Generate a configuration for tensorflow
config.gpu_options.allow_growth = True 				    # Allow for memory growth

####################################################################
## Function = This function appears to shuffle a bunch of datapoints

def split_by_batches(data,batch_size,shuffle = True):
    # data is a Nx??? array
    if shuffle:                                                     # If asked for...
        D = data[np.random.permutation(range(data.shape[0]))]       #  ...shuffle the data 
    else:                                                           # Otherwise
        D = data                                                    #  ...do nothing to the data

    D = np.array([D[k:k + batch_size] for k in range(0,len(data)-batch_size,batch_size)])
    return D

####################################################################
## Function = ?

def run_training_loop(data,vdata,input_tensor,pos_mean,batch_size,train_op,loss_op,recerr_op,log,dirname,log_freq,n_grad_step,save_freq,update_LR):
    # 
    def var_loss(session,vdat,nbatch = 10):
        # SUB-FUNCTION TO CALCULATE THE LOSS OF THE VAE
        D     = split_by_batches(vdat,batch_size,shuffle = False)   # Shuffel and partition data into batches
        loss  = 0                                                   # Initialize the loss to zero
        rerr  = 0                                                   # Initialize the reconstruction loss to zero
        nb    = 0                                                   # Initialize a counter over the number of batches
        means = []                                                  # Initialize mean storage array to an empty array
        for d in D:                                                 # Loop through the different batches
            nb   += 1                                               # Update counter
            l,r,m = session.run([loss_op,recerr_op,pos_mean],{input_tensor:d}) # TENSORFLOW: RUN A SESSION??
            loss += l                                               # Update the loss function
            rerr += r                                               # Update the reconstruction error 
            means.append(m)                                         # Append the mean to the mean storage array
            if nb == nbatch:                                        # Check if passed the number of batches
                break                                               #   ... if so, BREAK
        loss /= nbatch                                              # Normalize the loss to the number of batches
        rerr /= nbatch                                              # Normalize the reconstruction error to the number of batches
        return loss,rerr,np.concatenate(means,axis = 0)             # Return the loss the reconstruction error
    
    init   = tf.global_variables_initializer()                      #
    config = tf.ConfigProto()		                            # Initialize the tensorflow session configuration
    config.gpu_options.allow_growth = True 			    # Allow for memory growth
    sess   = tf.Session(config=config)                              # Start a tensorflow session
    K.setsession(sess)
    sess.run(init)                                                  # run the session
    

    nloss   = 0                                                     # Initailize a loss   
    t1      = time.time()                                           # Record start time
    av_time = -1                                                    #
    efrac   = .9                                                    #

    log.log(["grad_step","loss","recloss","var_loss","var_rec","learning_rate","time_rem"],PRINT = True)

    t_loss_temp = []                                                #
    t_rec_temp  = []                                                # 
    lrflag      = True                                              #
    saver       = tf.train.Saver(max_to_keep = 1000)                #
    
    for grad_step in range(n_grad_step + 1):                        #
        batch          = data[np.random.choice(np.arange(len(data)),batch_size)]     # Get a batch of data
        _,loss,recloss,newLR = sess.run([train_op,loss_op,recerr_op,update_LR],{input_tensor:batch}) # Run a session to get the loss/reconstruction error
        t_loss_temp.append(loss)                                    # Append loss to the ????
        t_rec_temp.append(recloss)                                  # Append reconstruction error to the ?????
        
        if grad_step % log_freq  == 0:
            if grad_step == 0:
                av_time = -1
            elif grad_step != 0 and  av_time < 0:
                av_time = (time.time() - t1)
            else:
                av_time = efrac*av_time + (1. - efrac)*(time.time() - t1)
                
            t1               = time.time()                          # 
            trem             = av_time*((n_grad_step)+1-grad_step)  # 
            trem             = trem / log_freq / 60. / 60.          # 
            loss             = np.mean(t_loss_temp)                 # 
            recloss          = np.mean(t_rec_temp)                  # 
            vloss,vrec,means = var_loss(sess,vdata)                 # 
            log.log([grad_step,loss,recloss,vloss,vrec,newLR,trem],PRINT = True) #
            t_loss_temp = []                                        # 
            t_rec_temp  = []                                        # 

        if grad_step % save_freq == 0:                              #
            util.dump_file(dirname + "/training_means_{}.pkl".format(grad_step),means)
            saver.save(sess,dirname + "/saved_params/saved_model_{}".format(str(grad_step))) #

    saver.save(sess,dirname + "/saved_params/saved_model_{}".format("final")) #
    sess.close()                                                    # 

####################################################################
## Function = ?   

def run(patch_size,n_batch,pca_frac,overcomplete,learning_rate,final_learning_rate,n_grad_step,loss_type,n_gauss_dim,n_lat_samp,seed,param_save_freq,log_freq,sigma,s1,s2,S,device,PCA_truncation,dataset):

    os.environ["CUDA_VISIBLE_DEVICES"]=str(device)                  # 
    np.random.seed(seed)                                            # Set the RNG seed to ensure randomness 
            
    dirname = util.get_directory(direc="./model_output/",tag = loss_type + "_{}".format(n_gauss_dim))
    params  = {
        "dataset":dataset,
        "patch_size":patch_size,
        "n_batch":n_batch,
        "pca_frac":pca_frac,
        "overcomplete":overcomplete,
        "learning_rate":np.float32(learning_rate),
        "final_learning_rate":np.float32(final_learning_rate),
        "pca_truncation":PCA_truncation,
        "n_grad_step":n_grad_step,
        "loss_type":loss_type,
        "n_gauss_dim":n_gauss_dim,
        "n_lat_samp":n_lat_samp,
        "sigma":np.float32(sigma),
        "param_save_freq":param_save_freq,
        "log_freq":log_freq,
        "s1":np.float32(s1),
        "s2":np.float32(s2),
        "S":np.float32(S)
    }
    
    util.dump_file(dirname +"/model_params",params)
    
    LOG = log.log(dirname + "/logfile.csv")

    netpar    = prepare_network(params)                             # 
    var       = netpar["variance"]                                  # 
    loss_exp  = netpar["loss_exp"]                                  # 
    recon_err = netpar["recon_err"]                                 # 
    images    = netpar["images"]                                    # 
    data      = netpar["data"]                                      # 
    varif     = netpar["vardat"]                                    #
    
    #get factor to multiply LR by:
    if final_learning_rate < learning_rate:
        LR_factor = np.float32(np.exp(-np.log(learning_rate/final_learning_rate)/n_grad_step))
    else:
        print("Final LR must be lower than initial LR! Overriding with LR_factor = 1")
        LR_factor = np.float32(1)
        
    LR    = tf.Variable(np.float32(learning_rate),trainable = False)# 
    adam  = tf.train.AdamOptimizer(learning_rate = LR)              # Set up the Adam optimization 
    train = adam.minimize(loss_exp)                                 # Run training
    update_LR = tf.assign(LR,LR*LR_factor)
    run_training_loop(data,varif,images,netpar["mean"],n_batch,train,loss_exp,recon_err,LOG,dirname,log_freq,n_grad_step,param_save_freq,update_LR)
    # OLD: run_training_loop(data,varif,images,n_batch,train,loss_exp,recon_err,LOG,dirname,log_freq,n_grad_step,param_save_freq)

####################################################################
## Function = ?   

def prepare_network(params,old = False):
    
    patch_size      = params["patch_size"]                          # Size of patches to run (size of one side of the square patch)
    n_batch         = params["n_batch"]                             # Number of batches to divide data into
    pca_frac        = params["pca_frac"]                            # 
    overcomplete    = params["overcomplete"]                        # 
    learning_rate   = params["learning_rate"]                       # 
    final_learning_rate   = params["final_learning_rate"]                       # 
    n_grad_step     = params["n_grad_step"]                         #
    loss_type       = params["loss_type"]                           # 
    n_gauss_dim     = params["n_gauss_dim"]                         # 
    n_lat_samp      = params["n_lat_samp"]                          # 
    sigma           = params["sigma"]                               # 
    param_save_freq = params["param_save_freq"]                     # 
    log_freq        = params["log_freq"]                            # 
    s1              = params["s1"]                                  # 
    s2              = params["s2"]                                  #
    S               = params["S"]                                   #
   #  patch_size      = params["patch_size"]                        # 
        
    print("getting data")
    if params["pca_truncation"] == "cut":                           # Option for hard cut-offs in the dimensionality-reducing PCA step
        n_pca = int((params["patch_size"] ** 2)*params["pca_frac"]) # Calculate the number of PCA components to keep

        if old:
            data,varif,test,PCA = dat.get_data(patch_size,n_pca,"BSDS",True) # 
        else:
            data,varif,test,PCA = dat.get_data(patch_size,n_pca,params["dataset"],True)

        n_lat = int(n_pca * params["overcomplete"])
        
    elif params["pca_truncation"] == "smooth":                      # Option for smooth truncation in the PCA dimensionality reduction step
        n_pca = int((params["patch_size"] ** 2))

        if old:
            data,varif,test,PCA     = dat.get_data(patch_size,n_pca,"BSDS",False)
            dataw,varifw,testw,PCAw = dat.get_data(patch_size,n_pca,params["dataset"],True)
        else:
            data,varif,test,PCA     = dat.get_data(patch_size,n_pca,params["dataset"],False)
            dataw,varifw,testw,PCAw = dat.get_data(patch_size,n_pca,params["dataset"],True)
        
        freq  = np.linspace(0,1,data.shape[1])                      #
        fc    = 1./params["overcomplete"]                           #
        n_lat = int(n_pca)                                          # 
        mask  = np.array([np.exp(-((freq/fc)**4))])                 #
        ev    = np.sqrt(np.array([PCA.explained_variance_]))        #
        
        data  = dataw*mask                                          # Apply mask to ???
        varif = varifw*mask                                         # Apply mask to ???
        test  = testw*mask                                          # Apply mask to ??? 
        
        data  = PCA.inverse_transform(data)                         # 
        varif = PCA.inverse_transform(test)                         # 
        test  = PCA.inverse_transform(varif)                        # 

        s     = np.std(data)                                        #

        data  /= s                                                  # Normalize the data
        varif /= s                                                  # Normalize the ??
        test  /= s                                                  # Normalize the ??
        
    print("constructing network")
    
    images = tf.placeholder(tf.float32,[n_batch,n_pca])
    
    lat_mean, lat_trans, lat_cor = enc.make_encoder(images,n_lat,n_gauss_dim,old = old)

    noise1 = tf.random_normal(shape = [int(lat_mean.shape[0]),n_lat_samp,1,int(lat_mean.shape[1])])
    noise2 = tf.random_normal(shape = [int(lat_mean.shape[0]),n_lat_samp,1,int(lat_mean.shape[1])])

    lv1 =  tf.reduce_sum(tf.expand_dims(lat_trans,1)*noise1,-1)
    lv2 =  tf.reduce_sum(tf.expand_dims(lat_cor,1)*noise2,-1)

    noise = lv1 + lv2
    
    latents = tf.expand_dims(lat_mean,1) + noise

    var,DC,CC = make_var.get_var_mat(lat_trans,lat_cor)

    reconstruction, weights = dec.make_decoder(latents,n_pca)
    
    prior_params = {"loss_type":loss_type,
              "s1":s1,
              "s2":s2,
              "S":S}
    
    loss_exp, recon_err = make_loss.make_loss(loss_type,images,reconstruction,lat_mean,var,latents,sigma,prior_params)

    return {
        "n_lat":n_lat,
        "images":images,
        "variance":var,
        "diagonal_var":DC,
        "factor_var":CC,
        "reconstruction":reconstruction,
        "weights":weights,
        "loss_exp":loss_exp,
        "recon_err":recon_err,
        "latents":latents,
        "noise":noise,
        "mean":lat_mean,
        "diag_trans":lat_trans,
        "diag_cor":lat_cor,
        "data":data,
        "vardat":varif,
        "testdat":test,
        "PCA":PCA
    }   

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("loss_type",                       help = "Latent distribution. Choose from: 'exp', 'gauss', 'cauch', and 'spikenslab'")
    parser.add_argument("overcomplete",      type = float, help = "Degree of overcomleteness to use. To match with the original SC results, use a value greater than 1.")
    
    parser.add_argument("--s1",              type = float, help = "scale of spike in spikenslab models", default = .1)
    parser.add_argument("--s2",              type = float, help = "Scale of slab in spikenslab models",  default = 1.)
    parser.add_argument("--S",               type = float, help = "sparsity (between 0 and 1)",          default = .1)

    parser.add_argument("--patch_size",      type = int,   help = "Size of image patches to fit.",      default = 12)
    parser.add_argument("--n_batch",         type = int,   help = "Size of batches to use in sgd.",     default = 32)
    parser.add_argument("--pca_frac",        type = float, help = "Fraction of PCA components to use.", default = 1)
    parser.add_argument("--learning_rate",   type = float, help = "learning rate to use",               default = .001)
    parser.add_argument("--final_learning_rate",   type = float, help = "learning rate to use",               default = .00001)
    parser.add_argument("--sigma",           type = float, help = "Std. of noise.",                     default = np.float32(np.exp(-1.)))

    parser.add_argument("--n_gauss_dim",     type = int,   help = "Rank of the non-orthogonal transformation matrix of an MVG posterior.",     default = 0)
    parser.add_argument("--n_lat_samp",      type = int,   help = "number of samples to draw from the variational posterior during training.", default = 1)
    parser.add_argument("--seed",            type = int,   help = "random seed.",                                                              default=1)

    parser.add_argument("--n_grad_step",     type = int,   help = "number of gradient descent steps to take..",                  default = 1000000)
    parser.add_argument("--param_save_freq", type = int,   help = "number of gradient descent steps between saving parameters.", default = 50000)
    parser.add_argument("--log_freq",        type = int,   help = "number of gradient descent steps between log entries.",       default = 100)
    parser.add_argument("--device",          type = int,   help = "Which GPU to use.",                                           default = 0)
    parser.add_argument("--dataset",         type = str,   help = "Which dataset to use (default: BSDS)",                        default = "BSDS")
    parser.add_argument("--PCA_truncation",  type = str,   help = "How to truncate fourier space: 'cut' - hard cutoff in PCA eigenvalue. 'smooth' - smooth decrease in eigenvalues (default)",default = "cut")

                        
    args = vars(parser.parse_args())

    #Arg: Loss, dataset, decoder (deep or shallow)
    run(**args)#patch_size = 12,n_batch = 32,pca_frac = np.pi/4,overcomplete = float(sys.argv[4]),lr = .0001,n_iter = 10000,n_samp = 500,loss_type = sys.argv[1],dataset = sys.argv[2],whiten = True,decoder = sys.argv[3],mvg = False,n_gauss_dim = 1,n_lat_samp = 3,seed = 1,CNN = True)
