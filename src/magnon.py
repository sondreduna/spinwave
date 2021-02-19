import numpy as np


J  = 1
d  = 1
mu = 1

def H(S,B):

    ss = -1/2 * J * np.einsum('ij,ik->',S,S)
    s2 = - d      * np.sum(S[1]*S[1])
    sb = - mu     * np.einsum('ij,i->'S,B)
    
    return ss + s2 + sb

def 
