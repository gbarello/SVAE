import numpy as np
import theano
import theano.tensor as T
import hastings.hasting as hast
import distributions.distributions as dist

NORM = True
CONSISTENCY = False


def prior(x,N):
    return - np.abs(x).sum(axis = 1) - N * np.log(2)

def g_prior(x,N):
    return - np.sign(x)

prior_func = lambda x: -prior(x,N)
g_prior_func = lambda x: -g_prior(x,N)


#XO,W = hast.AIS(fa,fb,lambda n:np.random.laplace(0,1,[n[0],n[1]]),N,n_samp = 500,n_AIS_step = 100,nhstep = 3,eps = .1,grad = [ga,gb],L = 100,PRINT = True)

#print("AIS norm",np.mean(W))
#print("true norm",norm(1))

N = 100

for l in ["c_spikenslab"]:#,"gauss","exp","cauch","spikenslab"]:
    print(l)
    f,g,d = dist.get_distribution(l)
    
    XX = T.matrix("x","float32")
    
    ffunc = theano.function([XX],-f(XX.dimshuffle([0,'x',1])).mean(axis = 1),allow_input_downcast = True)
    gfunc = theano.function([XX],-g(XX.dimshuffle([0,'x',1])).mean(axis = 1),allow_input_downcast = True)

    grad = [g_prior_func,gfunc]

    if NORM:
        XO,W = hast.AIS(prior_func,ffunc,lambda n:np.random.laplace(0,1,[n[0],n[1]]),N,n_samp = 500,n_AIS_step = 1000,nhstep = 20,eps = .01,grad = grad,L = 10,PRINT = True)
        print("AIS norm",np.mean(np.exp(W)))

    if CONSISTENCY:
        m = []
        for k in [100]:
            x,w = hast.hastings(ffunc,xi,k,eps = .01,grad = gfunc,L = 10)
            m.append(w.mean())
        
        print(m)
        xi = d(1000,N)
        true = ffunc(xi).mean()

        print(true)
