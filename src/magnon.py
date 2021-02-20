import numpy as np

J  = 1
d  = 1
mu = 1
gamma = 1
alpha = 0

global B

def H(S):

    n = np.shape(S)[1] # number of spins
    
    #ss = -1/2 * J * np.einsum('ij,ik->',S,S) # sum over all spins
    ss = - 1/2 * J * np.sum([ S[:,i] @ ( S[:,i-1] + S[:,(i + 1) % n] ) for i in range(n)])
    s2 = - d       * np.sum(S[:,2]*S[:,2])
    #sb = - mu     * np.einsum('ij,i->',S,B)
    
    return ss + s2 

def djH(S):

    delta = 0.00001
    
    dS = np.full(np.shape(S),delta)

    return (H(S + dS) - H(S))/dS


C = gamma/(mu*(1+alpha**2))

def f_llg(t,S):
    dH = djH(S)
    return C * ( np.cross(S,dH) + alpha * np.cross(S, np.cross(S, dH) ) )

    
    
    
