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

import svae.data.get_data as dat
import svae.data.image_processing.test_gratings as grat

import svae.losses.make_loss as make_loss
import svae.distributions.distributions as distributions

from skimage.filters import gaussian as GFILT

import os

import shutil
import utilities as utils

def run(dirname):

    LOG = log.log(dirname + "/weight_log.log")

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

    LOG.log(np.std(test))

    sp1 = np.random.randn(test.shape[0],n_lat) * MP["s1"]

    sp2 = np.random.randn(test.shape[0],n_lat) * MP["s2"]
   
    S = MP["S"]

    LOG.log("sp1 {}".format(np.std(sp1)))
    LOG.log("sp2 {}".format(np.std(sp2)))

    LOG.log("Wsp1 {}".format(np.std(np.tensordot(sp1,W[0],axes = [1,1]))))
    LOG.log("Wsp2 {}".format(np.std(np.tensordot(sp2,W[0],axes = [1,1]))))

    LOG.log("SW {}".format(S*np.std(np.tensordot(sp2,W[0],axes = [1,1])) + (1. - S)*np.std(np.tensordot(sp1,W[0],axes = [1,1]))))

    A = get_file(MP["dirname"] + "/test_means_best.csv")

    LOG.log("RV {}".format(np.std(np.tensordot(W[0],A,axes = [1,1]))))
    LOG.log("DV {}".format(np.std(var)))
    
        
def get_weights(MP,param_name = "decoder_params_best"):
    W=utils.load_obj(MP["dirname"] + param_name)
    return W                    

def get_file(fname):
    return np.loadtxt(fname)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dirname",help = "Relative path to directory containing the trained model you'd like to analyze.")

    args = vars(parser.parse_args())

    #arg: loss, dataset, decoder (deep or shallow)
    run(**args)
