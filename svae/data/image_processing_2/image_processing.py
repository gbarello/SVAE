from PIL import Image
import numpy as np
import math
from scipy import signal
import test_gratings as test

norm = 100

#these functions make DOG filters

def load_grayscale_image(imfile):
    img = np.array(Image.open(imfile))

    if len(img.shape) == 3:
        img = np.array(img).mean(axis = 2)
        
    img = img/255

    return img

def LAPs(a,f,t,r = 2):

    s = f/(2*math.pi)

    #the optimal 'k' for a given LAP is .7 * 2pi * s

    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2

    def x(xp,yp):
        return v1*(float(xp)-t2) + (v2*(float(yp)-t2))
    def y(xp,yp):
        return v2*(float(xp)-t2) - (v1*(float(yp)-t2))

    EE = np.exp(np.array([[-((x(xp,yp)**2) + ((y(xp,yp)/r)**2))/(2*s*s) for xp in range(t)] for yp in range(t)]))
    scale = -np.array([[x(xp,yp) for xp in range(t)] for yp in range(t)])/(s*s*s*s)

    #gotta divide by wavelength which is prop. to s
    #empirically the normalization needs two additional powers of the scale
    FF =  EE*scale

    x =  norm*FF/(np.sqrt(np.sum(np.abs(FF)**2)))

    return x - np.mean(x)

def LAPc(a,f,t,r = 2):

    s = f/(math.pi*np.sqrt(2))

    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2

    def x(xp,yp):
        return v1*(float(xp)-t2) + v2*(float(yp)-t2)
    def y(xp,yp):
        return v2*(float(xp)-t2) - v1*(float(yp)-t2)

    EE = np.exp(-np.array([[((x(xp,yp)**2) + ((y(xp,yp)/r)**2))/(2*s*s) for xp in range(t)] for yp in range(t)]))
    scale = (np.array([[(x(xp,yp)**2) for xp in range(t)] for yp in range(t)]) - s*s)/(s*s*s*s*s*s*np.sqrt(2*np.pi))#the sqrt(3) is to give it the same normalization as the odd-phase one.

    #gotta divide by wavelength for normalization of freq. bands, and f ~ s
    FF =  -EE*scale

    x = norm*FF/(np.sqrt(np.sum(np.abs(FF)**2))*f)

    return x - np.mean(x)
    
##these functions make a gabor filter (s and c)

def gaborS(a,k,s,t):
    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2
    sin = np.array([[np.sin(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k) for x in range(t)] for y in range(t)])
    exp = np.exp(-np.array([[(float(x) - t2)**2 + (float(y) - t2)**2 for x in range(t)] for y in range(t)])/(2.*(s**2)))

    out =  norm*sin*exp/np.sqrt(k*k*np.sum((sin*exp)**2))

    base = (test.GRATC(1,0,1,0,t)*out).sum() # this is here because really these things should integrate a constant stimulus to zero but the gabors are messed up in that they don't! I should be using DOG.

    return out - base

def gabor(a,k,s,t,P):
    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2
    sin = np.array([[np.sin(P + 2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k) for x in range(t)] for y in range(t)])
    exp = np.exp(-np.array([[(float(x) - t2)**2 + (float(y) - t2)**2 for x in range(t)] for y in range(t)])/(2.*(s**2)))

    out =  norm*sin*exp/np.sqrt(k*k*np.sum((sin*exp)**2))
    
    base = (test.GRATC(1,0,1,0,t)*out).sum()

    return out - base

def gaborC(a,k,s,t):
    v1 = np.sin(a)
    v2 = np.cos(a)
    t2 = float(t-1)/2
    sin = np.array([[np.cos(2*math.pi*((v1*(float(x)-t2)) + (v2*(float(y)-t2)))/k) for x in range(t)] for y in range(t)])
    exp = np.exp(-np.array([[(float(x) - t2)**2 + (float(y) - t2)**2 for x in range(t)] for y in range(t)])/(2.*(s**2)))

    out = norm*sin*exp/np.sqrt(k*k*np.sum((sin*exp)**2))
    
    base = (test.GRATC(1,0,1,0,t)*out).sum()

    return out - base

##
                    
def get_filter_coefficients(pic, na, k, s, t, meansub = False,lap = True):
    kernels = np.array([gaborC(aa,k,s,t) for aa in [i * math.pi / na for i in range(na)]])
    P = pic

    if meansub:
        P = P - signal.convolve(P,(1./(t**2))*np.ones((t,t)),mode = "same")

    fil = np.array([signal.convolve(P,K,mode = "valid") for K in kernels])

    return np.transpose(fil,(1,2,0))

def get_phased_filter_coefficients(pic, na, npha, k, s, t, r, meansub = False,lap = True):
    if lap:
        if npha != 2:
            print("warning! when using DOG ffilters npha = 2, not {}".format(npha))

        kernels = np.array([[LAPc(aa,k,t,r),LAPs(aa,k,t,r)] for aa in [i * math.pi / na for i in range(na)]])
    else:
        kernels = np.array([[gabor(aa,k,s,t,p) for p in [j*math.pi/npha for j in range(npha)]] for aa in [i * math.pi / na for i in range(na)]])

    P = pic

    if meansub:
        P = P - signal.convolve(P,(1./(t**2))*np.ones((t,t)),mode = "same")


    fil = np.array([[signal.convolve(P,p,mode = "valid") for p in K] for K in kernels])


    return np.transpose(fil,(2,3,0,1))

def get_filter_maps(nf, na, npha, k, s, t, d, r,lap = True):
    if lap:
        if npha != 2:
            print("warning! when using DOG ffilters npha = 2, not {}".format(npha))

        kernels = np.array([[LAPc(aa,k,t,r),LAPs(aa,k,t,r)] for aa in [i * math.pi / na for i in range(na)]])
    else:
        kernels = np.array([[gabor(aa,k,s,t,p) for p in [j*math.pi/npha for j in range(npha)]] for aa in [i * math.pi / na for i in range(na)]])


    filters = np.reshape(np.array([kernels for k in range(1 + nf)]),[1 + nf,na*npha,t,t])

    fpos = np.array([(0,0)] + [(int(np.cos(aa)*d),int(np.sin(aa)*d)) for aa in [i*2*math.pi/nf for i in range(nf)]])

    fullW = 2*d + t 

    def get_full(ker,pos,full):
        out = np.zeros([full,full])

        cen = float(full)/2
        kcen = float(len(ker))/2

        for i in range(len(ker)):
            for j in range(len(ker[i])):
                out[int(cen - kcen + i + pos[0]),int(cen - kcen + j + pos[1])] = ker[i,j]
        return out

    out = np.array([[get_full(filters[i,j],fpos[i],fullW) for j in range(na*npha)] for i in range(1 + nf)])

    print(out.shape)

    return out

def sample_coef(coef,sam,nf,d):
    fpos = np.array([(0,0)] + [(int(np.cos(aa)*d),int(np.sin(aa)*d)) for aa in [i*2*math.pi/nf for i in range(nf)]])

    return np.array([[coef[(s + f)[0],(s + f)[1]] for f in fpos] for s in sam])

def get_filter_samples(iname,nf,na,k,s,t,d,samd,imsize = 500,mode = "file",MS = False):

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

    if mode == "file":
        img = Image.open(iname)
        print(img.size)
#        newsize = (PX[0]*imsize/PX[1],PX[1]*imsize/PX[1])
#        img = img.resize(newsize)
        
        img = np.array(img).mean(axis = 2)
        img = img/255
    else:
        img = iname


    fil = get_filter_coefficients(img,na,k,s,t,meansub = MS)
     
    rr = np.array([np.random.random_integers(-samd,samd,2) for x in range(samd + d + 1,len(fil[0,0]) - (samd + d) - 1,samd) for y in range(samd + d + 1,len(fil[0]) - (samd + d) - 1,samd)])

    xx = np.array([[y,x] for x in range(samd + d + 1,len(fil[0,0]) - (samd + d) - 1,samd) for y in range(samd + d + 1,len(fil[0]) - (samd + d) - 1,samd)])

    fsam = rr + xx

    out = sample_coef(fil,fsam,nf,d)
    
    return out
    
    
def get_phased_filter_samples(iname,nf,na,npha,k,s,t,d,samd,r,imsize = 500,mode = "file",MS = False):

    '''
    parameters:
      iname : the path to the image file to process
      nf    : number of "surround" filter patches
      na    : number of angles
      npha  : number of phases
      k     : frequency of gabor
      s     : scale (std.) of gabor
      t     : total size of gabor filter to compute (should be >> s)
      d     : distance between filters (also changes sampling rate)
      imsize: a standard size to reshape the 2nd axis to (while preserving the aspect ratio). 
   
    '''

    if mode == "file":
        img = Image.open(iname)

        print(img.size)
        #newsize = (PX[0]*imsize/PX[1],PX[1]*imsize/PX[1])
        #img = img.resize(newsize)
        
        img = np.array(img).mean(axis = 2)
        img = img/255
    else:
        img = iname


    fil = get_phased_filter_coefficients(img,na,npha,k,s,t,meansub = MS,r = r)
     
    rr = np.array([np.random.random_integers(-samd,samd,2) for x in range(samd + d + 1,len(fil) - (samd + d) - 1,samd) for y in range(samd + d + 1,len(fil[x]) - (samd + d) - 1,samd)])

    xx = np.array([[x,y] for x in range(samd + d + 1,len(fil) - (samd + d) - 1,samd) for y in range(samd + d + 1,len(fil[x]) - (samd + d) - 1,samd)])

    fsam = rr + xx


    out = sample_coef(fil,fsam,nf,d)
    
    
    return out
    
if __name__ == "__main__":

    NA = 10
    k = 4
    t = 12
    fd = 5

    FF = np.reshape(get_filter_maps(8, NA, 2, k,k, t, fd),[9,NA,2,2*fd + t,2*fd+t])

    print(FF.shape)


    for k in range(9):
        print((FF[0,0,0]*FF[k,0,0]).sum())
