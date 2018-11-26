import numpy as np

def IP(v,M):
    return np.tensordot(v,np.tensordot(M,v,axes = [1,0]),axes = [0,0])

def get_LL(W,x,sig):
    n = W.shape[1]
    D = x.shape[0]

    A = np.linalg.inv(np.dot(W,np.transpose(W)) + sig*sig*np.identity(D))

    _,logdet = np.linalg.slogdet(A)

    return (- IP(x,A) + logdet - D * np.log(2*np.pi*sig*sig))/2
if __name__== "__main__":

    W = np.array([[1,.1]])
    a = np.array([1])
    sig = 1

    
    print(get_LL(W,a,sig))
