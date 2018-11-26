import numpy as np
import tensorflow as tf
from .. distributions import distributions as dist

PI = np.float32(np.pi)
two = np.float32(2)

def make_loss(loss_type,image,recon,mean,var,latvals,sig,params):

#    var = tf.reduce_sum(tf.expand_dims(tf.transpose(trans,[0,2,1]),-1)*tf.expand_dims(trans,1),2)

    recon_err = get_log_recon_err(image,recon,sig)
        
    post = get_log_gauss_D(mean,var,latvals)
    
    f,g,d = dist.get_distribution(loss_type,params)
    prior = f(latvals)

    rerr = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(tf.pow(tf.expand_dims(image,1) - recon,2),axis = 1),axis = 1))

    KL = KL_loss(post,prior)

    return tf.reduce_mean(recon_err + KL), rerr

def get_log_gauss_D(mean,var,latent):

    vinv = tf.map_fn(tf.linalg.inv,var)

    #    lmd = tf.expand_dims(latent - tf.expand_dims(mean,1),2)#[nb,ng,1,nl]
    lmd = latent - tf.expand_dims(mean,1)#[nb,ns,nl]

    lmdV = tf.reduce_sum(tf.expand_dims(lmd,axis = 2)*tf.expand_dims(vinv,axis = 1),axis = -1)
    
    #    ip = tf.reduce_sum(lmd*lmd.transpose([0,1,3,2])*tf.expand_dims(tf.expand_dims(vinv,0),0),axis = [2,3])
    ip = tf.reduce_sum(lmd * lmdV, axis = -1)

    logdet = tf.map_fn(tf.linalg.logdet,var)

    lognorm = tf.expand_dims(- (logdet/two) - int(latent.shape[-1])*tf.log(two*PI)/two,-1)
    return -(ip/two) + lognorm

def get_log_recon_err(image,recon,sig):
    #pretty sure the coef sign is right, because we actually want to minimize the NEGATIVE log-likelihood (see sign above in make_loss)
    
    coef = tf.log(np.float32(2)*PI*(sig*sig))/np.float32(2)
    
    lat = tf.pow((tf.expand_dims(image,1) - recon)/sig,2)
    
    mean = tf.reduce_mean(coef + lat,axis = 1)
                          
    return tf.reduce_sum(mean,axis = 1)

def get_latent(mean,trans,noise):
    return tf.expand_dims(mean,1) + tf.reduce_sum(tf.expand_dims(trans,1)*tf.expand_dims(noise,2),axis = 3)

def KL_loss(a,b):
    #a and b are [nbatch,nsamp]    
    #KL divergence is 
    return tf.reduce_mean(a - b,1)
