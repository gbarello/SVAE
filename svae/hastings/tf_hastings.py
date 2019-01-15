import tensorflow as tf
import numpy as np

def hastings_step(f,x,step,p):

    xp = x + step
    
    start = -f(x)
    end = -f(xp)

    alpha = end-start

    out = x

    i = tf.greater(alpha,p)

    out = tf.where(i,xp,out)

    return out#,f(out)]

def ham_hastings_step(f,g,x,mom,r,eps,L):

    '''
    This funciton takes the functions f and g (which take x as an argument) and returns a tensor representing x and mom after a single hamiltonian hastings step.

    '''

    Qinit = x
    Q = x
    P = mom
    P = P - eps * g(Q)/2

    
    for i in range(L):
        Q = Q + eps*P
        if i != L-1:
            P = P - eps*g(Q)
            
    P = P - eps*g(Q)/2

    P = -P

    fU = f(Q)
    fK = tf.reduce_sum(P**2,axis = 1)/2
    iU = f(x)
    iK = tf.reduce_sum(mom**2,axis = 1)/2

    out = x

    i = tf.greater(fU - iU + fK - iK,r)

    out = tf.where(i,Q,out)

    return out#,f(out)

def hastings(f,init,nstep,eps = .1,grad = -1,L = 10):

    #HASTINGS TAKES NEG LG LIKELIHOOD
    
    u = tf.log(tf.random_uniform([nstep,int(init.shape[0])],0,1))
    s = tf.random_normal([nstep,int(init.shape[0]),int(init.shape[1])])

    xt = init
    ft = f(xt)
    
    if grad == -1:
        out = tf.scan(lambda x,II:hastings_step(f,x,II[0]*eps,II[1]),[s,u],xt)
    else:
        out = tf.scan(lambda x,II:ham_hastings_step(f,grad,x,II[0],II[1],eps,L),[s,u],xt)

    #out should be shape [nstep,x_dim]
    return out[-1]#,f(out[-1])

def AIS(f1,f2,f1sam,shape,n_samp,n_AIS_step,nhstep,eps = .1,grad = -1,L = 10,PRINT = False):
    #THESE ARE NEG LOG LIKELIHOODS
    beta = np.linspace(0,1,n_AIS_step + 1,dtype = np.float32)
    
    X = tf.Variable(np.float32(f1sam(n_samp,shape)))
    F = []
    
    f = lambda y,b:(1.-b)*f1(y) + b*f2(y)

    if grad == -1:
        g = -1
        hastings_step = lambda x,b:hastings(lambda xx:f(xx,b),x,nhstep,eps = eps,grad = g,L = L)
    else:
        g = lambda y,b:(1.-b)*grad[0](y) + b*grad[1](y)
        hastings_step = lambda x,b:hastings(lambda xx:f(xx,b),x,nhstep,eps = eps,grad = lambda xx:g(xx,b),L = L)

    

    #result = [X]
    #for i in range(n_AIS_step):
        #result.append(hastings_step(result[-1],beta[i+1]))
    #result = tf.stack(result,axis = 0)
    print(beta.shape)
    print(X.shape)
    result = tf.scan(hastings_step,beta[1:],X)
    #These are the "x" values, now we need to compute fa, fb

    print(result.shape)
    #result = tf.concat([[X],result],axis = 0)

    F = tf.map_fn(lambda b:(-f(b[0],b[1]),-f(b[0],b[2])),[result,beta[:-1],beta[1:]],dtype = (tf.float32,tf.float32))

    for i in range(2):
        print(F[i].shape)
    
    lW = tf.reduce_sum(F[1] - F[0],axis = 0)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)        
        x,lw, = sess.run([result,lW])
    
    return x,lw
    '''
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
    '''
if __name__ == "__main__":

    def prior(x,N):
        return - tf.reduce_sum(tf.abs(x),axis = 1) - N * np.float32(np.log(2))

    def poste(x,N):
        #UNNORMALIZED PSOTERIOR!
        return - tf.reduce_sum(((x-5)*(x-5)/(.25**2)),axis = 1)/2

    def true_norm(x,N):
        #UNNORMALIZED PSOTERIOR!
        return - (N/2)*np.log(2*np.float32(np.pi)*.25*.25)

    def g_prior(x,N):
        return - tf.sign(x)

    def g_poste(x,N):
        return  - (x-5)/(.25**2)

    N = 100
    nsamp = 500
    n_AIS_step = 100
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
    for k in range(5):
        t1 = time.time()
        XO,W = AIS(fa,fb,lambda a,b:np.float32(np.random.laplace(0,1,[a,b])),N,n_samp = nsamp,n_AIS_step = n_AIS_step,nhstep = nhstep,eps = eps,grad = [ga,gb],L = L,PRINT = True)
        t2 = time.time()
        print("time: {}".format(t2 - t1))
        print("AIS norm",np.mean(W))
        print("true norm",norm(1))

    '''
    from .. distributions import distributions as dist

    f1,g1,f1sam = dist.get_distribution("exp",{})

    f2 = lambda x:f1(x - 1.) - 1.
    g2 = lambda x:g1(x - 1.)
    
    n_samp = 500
    n_AIS_step = 1000
    nhstep = 20
    shape = 100
    
    grad = [g1,g2]
#    grad = -1
    X,W = AIS(f1,f2,f1sam,shape,n_samp,n_AIS_step,nhstep,eps = .01,grad = grad,L = 10,PRINT = False)

    print(np.mean(W))
    print(np.exp(1.))
    '''
