import numpy as np

def hastings_step(f,x,step,p):

    xp = x + step
    
    start = -f(x)
    end = -f(xp)

    alpha = end-start

    out = np.copy(x)

    i = (p < alpha)

    out[i] = xp[i]

    return [out,f(out)]

def ham_hastings_step(f,g,x,mom,r,eps,L):

    Q = np.copy(x)
    P = np.copy(mom)

    P = P - eps * g(Q)/2
    for i in range(L):

        Q = Q + eps*P
        if i != L-1:
            P = P - eps*g(Q)
    P = P - eps*g(Q)/2

    P = -P

    fU = f(Q)
    fK = (P**2).sum(axis = 1)/2
    iU = f(x)
    iK = (mom**2).sum(axis = 1)/2

    out = np.copy(x)

    i = (r < (fU - iU +fK - iK))

    out[i] = Q[i]

    return out,f(out)

def hastings(f,init,nstep,eps = .1,grad = -1,L = 10):

    #HASTINGS TAKES NEG LG LIKELIHOOD
    
    u = np.log(np.random.uniform(0,1,[nstep,init.shape[0]]))
    s = np.random.randn(nstep,init.shape[0],init.shape[1])

    xt = np.copy(init)
    ft = f(xt)
    for k in range(nstep):
        if grad == -1:
            xt,ft = hastings_step(f,xt,s[k]*eps,u[k])
        else:
            xt,ft = ham_hastings_step(f,grad,xt,s[k],u[k],eps,L)

    return xt,ft

def AIS(f1,f2,f1sam,shape,n_samp,n_AIS_step,nhstep,eps = .1,grad = -1,L = 10,PRINT = False):
    #THESE ARE NEG LOG LIKELIHOODS
    beta = np.linspace(0,1,n_AIS_step + 1)
    
    X = f1sam([n_samp,shape])
    F = []

    for k in range(1,len(beta)):
        print(k)
        if PRINT:
            print(k)

        fa = lambda y:(1.-beta[k-1])*f1(y) + beta[k-1]*f2(y)
        fb = lambda y:(1.-beta[k])*f1(y) + beta[k]*f2(y)

        if grad != -1:
            g = lambda y:(1.-beta[k])*grad[0](y) + beta[k]*grad[1](y)
        else:
            g = -1
                        
        X,f = hastings(fb,X,nhstep,eps,g,L)
        F.append([-fa(X),-fb(X)])

        if PRINT:
            G = np.array(F)
            print((G[:,1] - G[:,0]).sum(axis = 0).mean())


    F = np.array(F)
    lW = (F[:,1] - F[:,0]).sum(axis = 0)
    
    return X,lW

if __name__ == "__main__":

    def prior(x,N):
        return - np.abs(x).sum(axis = 1) - N * np.log(2)

    def poste(x,N):
        #UNNORMALIZED PSOTERIOR!
        return - ((x-5)*(x-5)/(.25**2)).sum(axis = 1)/2

    def true_norm(x,N):
        #UNNORMALIZED PSOTERIOR!
        return - (N/2)*np.log(2*np.pi*.25*.25)

    def g_prior(x,N):
        return - np.sign(x)

    def g_poste(x,N):
        return  - (x-5)/(.25**2)

    N = 100
    nsamp = 500
    n_AIS_step = 50
    nhstep = 3
    eps = .1
    L = 100

    fa = lambda x: -prior(x,N)
    fb = lambda x: -poste(x,N)# + prior(x,N)

    ga = lambda x: -g_prior(x,N)
    gb = lambda x: -g_poste(x,N)# + prior(x,N)

    norm = lambda x: -true_norm(x,N)# + prior(x,N)

    #f1,f2,f1sam,shape,n_samp,n_AIS_step,nhstep,eps = .1,grad = -1,L = 10)
    import time
    t1 = time.time()
    XO,W = AIS(fa,fb,lambda n:np.random.laplace(0,1,[n[0],n[1]]),N,n_samp = nsamp,n_AIS_step = n_AIS_step,nhstep = nhstep,eps = eps,grad = [ga,gb],L = L,PRINT = True)
    t2 = time.time()
    print("time: {}".format(t2 - t1))
    print("AIS norm",np.mean(W))
    print("true norm",norm(1))
        
    from .. distributions import distributions as dist

    for l in ["gauss","exp","cauch","spikenslab"]:
        f,g,d = dist.get_distribution(l)

        XX = T.matrix("x","float32")

        ffunc = theano.function([XX],f(XX.dimshuffle([0,'x',1])).mean(axis = 1),allow_input_downcast = True)
        gfunc = theano.function([XX],g(XX.dimshuffle([0,'x',1])).mean(axis = 1),allow_input_downcast = True)

        xi = np.random.multivariate_normal(np.zeros(N),np.identity(N),[1000,N]) 
        
        x,w = hastings(ffunc,xi,100,eps = .1,grad = gfunc,L = 10)

        print(w.mean())
