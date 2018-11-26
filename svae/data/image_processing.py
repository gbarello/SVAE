import numpy as np
from PIL import Image
import math
from scipy import signal

def get_array_data(f,convert_to_float = False):
    F = open(f,"r")
    
    out = []

    for l in F:
        temp = l.split(",")
        temp[-1] = temp[-1][:-1]
        if convert_to_float:
            out.append([float(x) for x in temp])
        else:
            out.append([str(x) for x in temp])
            
    return np.array(out)

def split_by_size(A,s):
    SH = A.shape

    return np.array([A[i:i+s,j:j+s] for i in range(0,SH[0]-s,s) for j in range(0,SH[1]-s,s)])

def get_filter_samples(iname,size = 15,imsize = "def"):

    '''
    parameters:
      iname : the path to the image file to process
      nf    : number of "surround" filter patches
      na    : number of angles
      k     : frequency of gabor
      s     : scale (std.) of gabor
      t     : total size of gabor filter to compute (should be >> s)
      d     : distance between filters (also changes sampling rate)
      imsize: a standard size to reshape the 2nd axis to (while preserving the aspect ratio). 
   
    '''

    img = Image.open(iname)
    PX = img.size

    if imsize == "def":
        imsize = PX[1]

    newsize = (PX[0]*imsize/PX[1],PX[1]*imsize/PX[1])
    img = img.resize(newsize)
    
    img = np.array(img).mean(axis = 2)/255.#conv. to grayscale
#    img = (img - img.mean())/img.std()

    OT = split_by_size(img,size)

    return np.reshape(OT,(OT.shape[0],-1))

def get_audio_samples(data,fmin,fmax,tau = .1,chunk = .5,nfreq = 50,scale="LOG",srate = 16000):

    #first find the frequencies to sample:
    if scale == "LOG":
        #if it is log scaling we need 
        hz = [float(fmin*np.exp(f)) for f in [k*np.log(float(fmax)/fmin)/(nfreq-1) for k in range(nfreq)]]
    if scale == "LINEAR":
        #if it is log scaling we need 
        hz = [fmin + f*(fmax - fmin)/(nfreq - 1) for f in range(nfreq)]

    #first make the filters, with envelope to get rid of high frequency artifacts
    env = np.array([t * (srate * tau - 1 - t) for t in range(int(srate*tau))])

    F = np.array([env * np.array([np.sin(2. * math.pi * f * t / srate) for t in range(int(srate*tau))]) for f in hz])

    print(F.shape)

    conv = np.array([np.convolve(f,data) for f in F]).transpose()

    chunks = np.array(np.split(conv,np.arange(len(conv))[::int(int(chunk*srate))])[1:-1])

    return chunks

