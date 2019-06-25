import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

PI = np.float32(np.pi)

def get_distribution(loss_type,params = {}):
    if loss_type == "gauss":
        return make_G(params)
    if loss_type == "corgauss":
        return make_CG(params)
    if loss_type == "special_corgauss":
        return make_special_CG(params)
    elif loss_type == "exp":
        return make_E(params)
    elif loss_type == "cauch":
        return make_C(params)
    elif loss_type == "spikenslab":
        return make_SS(params)
    elif loss_type == "e_spikenslab":
        return make_E_SS(params)
    elif loss_type == "c_spikenslab":
        return make_C_SS(params)
    else:
        print("error! Loss function not recognized")
        exit()

def make_G(params):

    if "mean" not in params.keys():
        params["mean"] = np.float32(0)
    if "std" not in params.keys():
        params["std"] = np.float32(1)
    
    def f(latvals):
        return -tf.reduce_sum(((latvals - tf.expand_dims(params["mean"],0))/tf.expand_dims(params["std"],0))**2,axis = -1)/2 - (int(latvals.shape[-1])/2)*np.log(2*PI)

    def g(latvals):
        return -(latvals - tf.expand_dims(params["mean"],0))/tf.expand_dims(params["std"],0)

    def dist(x,y):
        return np.random.randn(x,y)*np.expand_dims(params["std"],0) + np.expand_dims(params["mean"],0)

    return f,g,dist

def make_CG(params):

    def f(latvals):
        lv_meansub = latvals - expand_tensor(params["mean"],len(latvals.shape) - len(params["mean"].shape))
        return -tf.reduce_sum(lv_meansub*tf.tensordot(lv_meansub,tf.linalg.inv(params["cov"]),axes = [[-1],[1]]),axis = -1)/2 - (int(latvals.shape[-1])*np.log(2*PI) + tf.linalg.logdet(params["cov"]))/ 2

    def g(latvals):
        lv_meansub = latvals - expand_tensor(params["mean"],len(latvals.shape) - len(params["mean"].shape))
        return -tf.tensordot(lv_meansub,tf.linalg.inv(params["cov"]),axes = [[-1],[1]])

    def dist(x,y):
        #assert y == params["mean"].shape[0]
        return np.random.multivariate_normal(params["mean"],params["cov"],x)#np.tensordot(np.random.randn(x,y),np.linalg.cholesky(params["cov"]),axes = [[1],[1]]) + params["mean"]

    return f,g,dist

def make_special_CG(params):

    def f(latvals):
        lv_meansub = latvals - expand_tensor(params["mean"],len(latvals.shape) - len(params["mean"].shape))
        return -tf.reduce_sum(lv_meansub*tf.tensordot(lv_meansub,tfp.math.pinv(params["cov"]),axes = [[-1],[1]]),axis = -1)/2 - (int(latvals.shape[-1])*np.log(2*PI) + tf.linalg.logdet(params["special_cov"]))/ 2

    def g(latvals):
        lv_meansub = latvals - expand_tensor(params["mean"],len(latvals.shape) - len(params["mean"].shape))
        return -tf.tensordot(lv_meansub,tfp.math.pinv(params["cov"]),axes = [[-1],[1]])


    return f,g,None

def make_SS(params):

    s1 = params["s1"]
    s2 = params["s2"]
    S = params["S"]
    
    #Log[P] = Log[S g1 + (1-S)g2]
    
    def f(latvals):
        D1 = -tf.reduce_sum((latvals/s1)**2,axis = -1)/2 - (int(latvals.shape[-1])/2)*np.log(2*PI*(s1**2))
        D2 = -tf.reduce_sum((latvals/s2)**2,axis = -1)/2 - (int(latvals.shape[-1])/2)*np.log(2*PI*(s2**2))

        return tf.log(S) + D2 + tf.log(1 + (1. - S)*tf.exp(D1 - D2)/S)#T.log(S*T.exp(D2) + (1.-S)*T.exp(D1))

    def g(latvals):
        D1 = -tf.reduce_sum((latvals/s1)**2,2)/2 - (int(latvals.shape[-1])/2)*np.log(2*PI*(s1**2))
        D2 = -tf.reduce_sum((latvals/s2)**2,2)/2 - (int(latvals.shape[-1])/2)*np.log(2*PI*(s2**2))

        D1 = tf.expand_dims(D1,-1)
        D2 = tf.expand_dims(D2,-1)
        
        DD1 = -latvals/(s1**2)
        DD2 = -latvals/(s2**2)

        return DD2 + ((DD1-DD2)*(1. - S)*tf.exp(D1 - D2)/S)/(1 + (1. - S)*tf.exp(D1 - D2)/S)#(S*T.exp(D2)*DD2 + (1.-S)*T.exp(D1)*DD1)/(S*T.exp(D2) + (1.-S)*T.exp(D1))
    #D1 + (T.exp(DD2 - DD1)*(D2 - D1)*(S/(1-S)))/(1  + T.exp(DD2 - DD1)*(S/(1-S)))

    def dist(x,y):
        a = np.random.uniform(0,1,[x])

        b = np.zeros_like(a)
        
        b[a < S] = 1
        b[b < 1] = 0

        b = np.reshape(b,[-1,1])
        
        small = np.random.randn(x,y)*s1
        big = np.random.randn(x,y)*s2
        
        return big*b + small*(1-b)

    return f,g,dist

def make_E_SS(params):
    #Log[P] = Log[S g1 + (1-S)g2]
    
    s1 = params["s1"]
    s2 = params["s2"]
    S = params["S"]
    
    def f(latvals):
        D1 = -tf.reduce_sum((latvals/s1)**2,axis = -1)/2 - (int(latvals.shape[-1])/2)*np.log(2*PI*(s1**2))
        D2 = -tf.reduce_sum(tf.abs(latvals/s2),axis = -1) - (int(latvals.shape[-1]))*np.log(2*s2)

        #return T.log(S*T.exp(D2) + (1.-S)*T.exp(D1))
        return tf.log(S) + D2 + tf.log(1 + (1. - S)*tf.exp(D1 - D2)/S)

    def g(latvals):
        D1 = -tf.reduce_sum((latvals/s1)**2,axis = -1)/2 - (int(latvals.shape[-1])/2)*np.log(2*PI*(s1**2))
        D2 = -tf.reduce_sum(tf.abs(latvals/s2),axis = -1) - (int(latvals.shape[-1]))*np.log(2*s2)

        D1 = tf.expand_dims(D1,-1)
        D2 = tf.expand_dims(D2,-1)
        
        DD1 = -latvals/(s1**2)
        DD2 = -tf.sign(latvals/s2)

        return DD2 + ((DD1-DD2)*(1. - S)*tf.exp(D1 - D2)/S)/(1 + (1. - S)*tf.exp(D1 - D2)/S)#(S*T.exp(D2)*DD2 + (1.-S)*T.exp(D1)*DD1)/(S*T.exp(D2) + (1.-S)*T.exp(D1))
    #D1 + (T.exp(DD2 - DD1)*(D2 - D1)*(S/(1-S)))/(1  + T.exp(DD2 - DD1)*(S/(1-S)))

    def dist(x,y):
        a = np.random.uniform(0,1,[x])

        b = np.zeros_like(a)
        
        b[a < S] = 1
        b[b < 1] = 0

        b = np.reshape(b,[-1,1])
        
        small = np.random.randn(x,y)*s1
        big = np.random.laplace(0,1,[x,y])*s2
        
        return big*b + small*(1-b)

    return f,g,dist

def make_C_SS(params):
    #Log[P] = Log[S g1 + (1-S)g2]
    
    s1 = params["s1"]
    s2 = params["s2"]
    S = params["S"]

    def f(latvals):
        D1 = -tf.reduce_sum((latvals/s1)**2,axis = -1)/2 - (int(latvals.shape[-1])/2)*np.log(2*PI*(s1**2))
        D2 = -tf.reduce_sum(tf.log(1. + (latvals/s2)**2),axis = -1) - int(latvals.shape[-1])*np.log(PI*s2)

        return tf.log(S) + D2 + tf.log(1 + (1. - S)*tf.exp(D1 - D2)/S)#T.log(S*T.exp(D2) + (1.-S)*T.exp(D1))

    def g(latvals):
        D1 = -tf.reduce_sum((latvals/s1)**2,axis = -1)/2 - (int(latvals.shape[-1])/2)*np.log(2*PI*(s1**2))
        D2 = -tf.reduce_sum(tf.log(1. + (latvals/s2)**2),axis = -1) - int(latvals.shape[-1])*np.log(PI*s2)

        D1 = tf.expand_dims(D1,-1)
        D2 = tf.expand_dims(D2,-1)
        
        DD1 = -latvals/(s1**2)
        DD2 = -2*(latvals/(s2*s2))/(1. + (latvals/s2)**2)

        return DD2 + ((DD1-DD2)*(1. - S)*tf.exp(D1 - D2)/S)/(1 + (1. - S)*tf.exp(D1 - D2)/S)

    def dist(x,y):
        a = np.random.uniform(0,1,[x])

        b = np.zeros_like(a)
        
        b[a < S] = 1
        b[b < 1] = 0

        b = np.reshape(b,[-1,1])
        
        small = np.random.randn(x,y)*s1
        big = np.random.standard_cauchy([x,y])*s2
        
        return big*b + small*(1-b)

    return f,g,dist

def make_E(parms):
    def f(latvals):
        exp = tf.reduce_sum(-tf.abs(latvals),axis = -1) - int(latvals.shape[-1])*np.log(2)
        return exp
    def g(latvals):
        return - tf.sign(latvals)
    def dist(x,y):
        return np.random.laplace(0,1,[x,y])

    return f,g,dist
    
def make_C(parms):
    def f(latvals):
        return -tf.reduce_sum(tf.log(1. + latvals**2),axis = -1) - int(latvals.shape[-1])*np.log(PI)

    def g(latvals):
        return -2*latvals/(1. + latvals**2)

    def dist(x,y):
        return np.random.standard_cauchy([x,y])
    
    return f,g,dist

def expand_tensor(x,n):
    assert n >= 0
    if n == 0:
        return x
    else:
        return expand_tensor(tf.expand_dims(x,0),n-1)
