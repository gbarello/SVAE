import pickle
import numpy as np
from sklearn.decomposition import PCA as PCA

from PIL import Image
import math
from . import image_processing as IM
import glob
import sys
from .. distributions import distributions

BSDSloc='/srv/data/data0/gbarello/data/BSDS300/'

def G(x,y,s,r,f):
    a = np.cos(r)*x + np.sin(r)*y
    b = np.cos(r)*y - np.sin(r)*x
    
    return np.exp(-((a/s)**2 + (b/(f*s))**2)/2)

def gauss_RF(size,c,r,f = 2):
    out = np.zeros([size,size])
    for i in range(size):
        for j in range(size):
            out[i,j] = G(i-c[0],j-c[1],1,r,f)

    return out
            
def rand_dict(patch_size,nvar):
    cen = np.random.uniform(0,patch_size,[nvar,2])
    rot = np.random.uniform(0,2*np.pi,[nvar])
    
    D = np.array([gauss_RF(patch_size,cen[k],rot[k]) for k in range(nvar)])/10

    return D

def get_CNN_dat(data,pca,whiten):
    cov = np.reshape(pca.explained_variance_,[1,-1])
    if whiten:
        data = pca.inverse_transform(data/np.sqrt(cov))
    else:
        data = pca.inverse_transform(data)

    return data

def make_synthetic_data(dist,patch_size,nvar,ndat = 100000):

    try:
        F = open("./datasets/syn_dict_{}_{}".format(patch_size,nvar),"rb")
        D = pickle.load(F)
        F.close()

    except:
        D = rand_dict(patch_size,nvar)
        D = np.reshape(D,[nvar,-1])
        
        F = open("./datasets/syn_dict_{}_{}".format(patch_size,nvar),"wb")
        pickle.dump(D,F)
        F.close()

    R = dist(ndat,nvar)
    
    return np.dot(R,D)#[ndat,patch_size**2]

def get_data(patch_size,nvar,dataset = "bruno",whiten = False,CNN = False):
    from scipy.ndimage.filters import gaussian_filter as gfilt

    try:
        F = open("./datasets/{}_{}_{}_{}".format(patch_size,nvar,dataset,whiten),"rb")
        dataset = pickle.load(F)
        F.close()

        white,fit_data,fit_var,fit_test = dataset

    except:
        if dataset == "bruno":

            from scipy.io import loadmat

            data = loadmat("./datasets/IMAGES.mat")["IMAGES"]
            
            #data = np.reshape(read_dat("./datasets/bruno_dat.csv"),[512,512,10])
            data = np.transpose(data,[2,1,0])
                        
            if False:
                #this whitens the data
                ff0 = [np.expand_dims(np.fft.fftfreq(len(d)),1) for d in data]
                ff1 = [np.expand_dims(np.fft.fftfreq(len(d[0])),0) for d in data]
                ff = [np.sqrt(ff0[k]**2 + ff1[k]**2) for k in range(len(ff0))]
                f0 = 1./(2.56)#200 cycles per picture = 2.56
                
                mask = [np.abs(f) * np.exp(-(np.abs(f)/f0)**4) for f in ff]
                
                data = [np.real(np.fft.ifft2(mask[k]*np.fft.fft2(data[k]))) for k in range(len(data))]
                data = np.array(data)
                
                #data = (data + data.min())/(data.max() - data.min())
                
                data = (data - np.mean(data))/np.std(data)
            
            data = np.reshape(np.concatenate([IM.split_by_size(d,patch_size) for d in data]),[-1,patch_size*patch_size])
            
            VARS = [np.std(d)**2 for d in data]
            p_10 = np.mean(VARS)
            
            data = np.array([d for d in data if np.std(d)**2 > p_10/10])

            data = (data - np.mean(data))/np.std(data)
            ######################

            
        elif dataset == "MNIST":
            data = read_dat("./../../data/MNIST/mnist_train.csv")
            
            lab = data[:,0]
            data = data[:,1:]
            data = np.reshape(data,[-1,28*28])
            
            data = (data + data.min())/(data.max() - data.min())
        
        elif dataset == "BSDS":
            imlist = np.squeeze(IM.get_array_data(BSDSloc + "iids_train.txt"))
            data = imlist
            
            #this whitens the data
            ff0 = [np.expand_dims(np.fft.fftfreq(len(d)),1) for d in data]
            ff1 = [np.expand_dims(np.fft.fftfreq(len(d[0])),0) for d in data]
            ff = [np.sqrt(ff0[k]**2 + ff1[k]**2) for k in range(len(ff0))]
            f0 = 1./10
            
            mask = np.abs(ff) * np.exp(-(np.abs(ff)/f0)**4)
            
            data = [np.real(np.fft.ifft2(mask[k]*np.fft.fft2(data[k]))) for k in range(len(data))]
            ######################

            data = [IM.get_filter_samples(BSDSloc + "images/train/" + i + ".jpg",size = patch_size) for i in imlist]
            data = np.concatenate(data)

            data = np.reshape(data,[-1,patch_size*patch_size])

            print("BSDS data size: {}".format(data.shape))
            
        else:
            f,g,dist = distributions.get_distribution(dataset)            
            data = make_synthetic_data(dist,patch_size,nvar)

        LL = len(data)
        var = data[:int(LL/10)]
        test = data[int(LL/10):int(2*LL/10)]
        data = data[int(2*LL/10):]
        
        white = PCA(nvar,copy = True,whiten = False)###THIS USED TO BE =whiten
        
        fit_data = data#white.fit_transform(data)
        fit_var = var#white.transform(var)
        fit_test = test#white.transform(test)

        fit_data = np.random.permutation(fit_data)
        fit_var = np.random.permutation(fit_var)
        fit_test = np.random.permutation(fit_test)
                
        F = open("./datasets/{}_{}_{}_{}".format(patch_size,nvar,dataset,whiten),"wb")
        pickle.dump([white,fit_data,fit_var,fit_test],F)
        F.close()

    if CNN:
        fit_data = get_CNN_dat(fit_data,white,whiten)
        fit_var = get_CNN_dat(fit_var,white,whiten)
        fit_test = get_CNN_dat(fit_test,white,whiten)
        
    return np.float32(fit_data),np.float32(fit_var),np.float32(fit_test),white

def read_dat(f):
    F = open(f,"r")

    out = []
    for l in F:
        temp = l.split(",")

        temp[-1] = temp[-1][:-1]

        out.append([float(x) for x in temp])
    F.close()
    return np.array(out)
        
