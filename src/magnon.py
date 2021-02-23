from ode import *

global J
global d
global mu
global B
global alpha

J  = 0
d  = 0
mu = 1
B  = np.array([0,0,1])
alpha = 0

gamma = 1

C = -gamma/(mu*(1+alpha**2))


def H(S):

    n = np.shape(S)[0] # number of spins
    
    #ss = -1/2 * J * np.einsum('ij,ik->',S,S) # sum over all spins
    ss = - J * np.sum([ S[:,i] @ ( S[:,i-1] + S[:,(i + 1) % n] ) for i in range(n)])
    s2 = - d * np.sum(S[:,2]*S[:,2])
    sb = - np.sum([s @ B for s in S])
    
    return ss + s2 + sb

def djH(S,j):

    delta = 0.0001
    
    dSx   = np.zeros(np.shape(S))
    dSy   = np.zeros(np.shape(S))
    dSz   = np.zeros(np.shape(S))
    
    dSx[j] = np.array([delta,0,0])
    dSy[j] = np.array([0,delta,0])
    dSz[j] = np.array([0,0,delta])
    
    dx    = (H(S + dSx) - H(S - dSx))/(2*delta)
    dy    = (H(S + dSy) - H(S - dSy))/(2*delta)
    dz    = (H(S + dSz) - H(S - dSz))/(2*delta)
    
    return np.array([dx,dy,dz])

def gradH(S):
    return np.array([-djH(S,j) for j in range(S.shape[0])])


def f_llg(t,S,**kwargs):

   
    
    dH = gradH(S)
    return C * ( np.cross(S,dH) + alpha * np.cross(S, np.cross(S, dH) ) )


def initial_cond(theta,phi):

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x,y,z])


class MagnonSolver(ODESolver):

    def __init__(self,t0,y0,tN,h,method = "Heun",**kwargs):
        super().__init__(f_llg,t0,y0,tN,h,method)

        J = kwargs["J"]
        d = kwargs["d"]
        mu = kwargs["mu"]
        B = kwargs["B"]
        alpha = kwargs["alpha"]
    
