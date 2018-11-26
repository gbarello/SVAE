import scipy.signal as signal
import numpy as np
from PIL import Image
import randomwalk.randomwalk as walks

norm = 1

def load_grayscale_image(imfile):
    print(imfile)
    img = np.array(Image.open(imfile))
    print(img.shape)

    if len(img.shape) == 3:
        img = np.array(img).mean(axis = 2)
        
    img = img/255

    return img

def LAPs(a,f,t,r = 2):

    '''
    Description: build a derivative-of-gaussian filter kernel at angle a with wavelength f, and total size t, and elogation r.


    '''

    s = f/(2*np.pi)

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

def LAPc(a,f,t,r):

    s = f/(np.pi*np.sqrt(2))

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
    
def get_filter_coefficients(pic, na, k, t, r):

    kernels = np.array([[[LAPc(aa,f,t,r),LAPs(aa,f,t,r)] for f in k] for aa in [i * np.pi / na for i in range(na)]])
    P = pic

    fil = get_filter_maps(P,kernels)

    #fil has shape [nang, nfreq, 2, nx,ny]

    return fil,kernels

def get_filter_maps(pic,kernels):
    return np.array([[[signal.convolve(pic,p,mode = "valid") for p in f] for f in a] for a in kernels])

def get_filter_bank(coeff,indices,pos,cenpos):
    
    '''
    Description: take an array of filter coefficients, indices for the filters, and an array of positions, and a central position, and return a set of filters fo the image.

    args:
    coeff - array of coefficients - [nang,nfreq,npha,nx,ny]
    indices - array of indices - [nfilt,3]
    pos - array of positions relative to center - [nfilt,2]
    cenpos - position of center - [2]
    
    '''
    pxvals = np.int32(pos + np.expand_dims(cenpos,0))

    nx,ny = coeff.shape[-2:]

    if np.min(pxvals) < 0:
        print("out of range, min")
        return False
    if np.max(pxvals[:,0]) >= nx:
        print("out of range, maxx")
        return False
    if np.max(pxvals[:,1]) >= ny:
        print("out of range, maxy")
        return False

    index = np.concatenate([indices,pxvals],axis = 1)    

    return coeff[np.transpose(index).tolist()]

def sample_path(coeffs,path,indices,pos):

    out = []

    for p in path:
        res = get_filter_bank(coeffs,indices,pos,p)

        if res is False:
            print("Path out of bounds")
            exit()
            return False

        out.append(res)

    return np.array(out)

def sample_images(img_names,na, k, t, r, f_pos, npaths, dx_var, F, L, load = True):
    #for each image, I need to compute the filter coefficients

    if load:
        filts = [get_filter_coefficients(load_grayscale_image(i),na,k,t,r) for i in img_names]
    else:
        filts = [get_filter_coefficients(i,na,k,t,r) for i in img_names]

    #at each position, we get each index

    indices = np.concatenate([[[a,b,c] for a in range(na) for b in range(len(k)) for c in range(2)] for p in f_pos])
    positions = np.concatenate([[p for a in range(na) for b in range(len(k)) for c in range(2)] for p in f_pos])

    XV = np.array([[dx_var,0],[0,dx_var]])
    FV = np.array([[F,0],[0,F]])

    out = []
    path_out = []
    
    for fcoef,ker in filts:
        out.append([])
        path_out.append([])
        
        xmax = fcoef.shape[-2]
        ymax = fcoef.shape[-1]
        
        for n in range(npaths):
            PATH = walks.gauss_random_walk(XV,FV,L - 1)
            pl = np.min(PATH)
            pm = np.max(PATH)

            pfl = np.min(positions)
            pfm = np.max(positions)

            START = np.array([[np.random.randint(- pl - pfl,xmax - pm - pfm),np.random.randint(- pl - pfl,ymax - pm - pfm)]])

            out[-1].append(sample_path(fcoef,PATH+START,indices,positions))
            path_out[-1].append(START+PATH)

    return np.array(out),np.array(path_out),filts[0][1]


if __name__ == "__main__":
    im = np.random.randn(10,300,500)

    na = 4
    k = [10,15]
    t = np.max(k)*5
    r = 2
    f_pos = np.array([[0,0],[0,10],[10,0],[-10,0],[0,-10]])
    npaths = 100
    dx_var = 1
    F = .8
    L = 10

    res,path,filt = sample_images(im,na, k, t, r, f_pos, npaths, dx_var, F, L,load = False)

    print(np.array(res).shape)

    np.savetxt("./kernels.csv",np.reshape(filt,[-1,np.prod(filt.shape[-2:])]))
            
