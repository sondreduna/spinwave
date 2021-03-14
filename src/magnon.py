from ode import *
import numba as nb

J  = 0
d  = 0
mu = 1
B  = np.array([0,0,1])
alpha = 0    

gamma = 1

C = -gamma/(mu*(1+alpha**2))

e_z = np.array([0,0,1])

def H(S):

    n = np.shape(S)[0] # number of spins
    ss = - J * np.sum([ S[i,:] @ ( S[i-1,:] + S[(i + 1) % n,:] ) for i in range(n)])
    s2 = - d * np.einsum('i,i',S[:,2],S[:,2])
    sb = - mu* np.einsum('ji,i->',S,B)

    return ss + s2 + sb

@nb.jit(nopython = True)
def djH(S,j,n):
    ss = J * (S[j-1,:] + S[(j+1)%n,:] ) # sum over nearest neighbours
    return ss + 2* d*S[j,2] * e_z + mu * B

@nb.jit(nopython = True)
def gradH(S,n):
    dH = np.zeros((n,3))
    for j in range(n):
        dH[j,:] = djH(S,j,n)
    return dH

@nb.jit(nopython = True)
def f_llg(t,S):
    n = np.shape(S)[0] # number of spins
    dH = gradH(S,n)
    return C * (np.cross(S,dH) + alpha * np.cross(S, np.cross(S, dH) ) )


def initial_cond(theta,phi):

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.array([x,y,z])


class MagnonSolver(ODESolver):

    def __init__(self,t0,y0,tN,h,method = "Heun",**kwargs):
        super().__init__(f_llg,t0,y0,tN,h,method)

        self.shape = np.shape(y0)

        # each spin array is two dimensional
        
        self.Y = np.zeros((self.N + 2,self.shape[0], self.shape[1]))

        # setting parameters for the problem

        global J
        global d
        global mu
        global B
        global alpha
        
        J = kwargs["J"]
        d = kwargs["d"]
        mu = kwargs["mu"]
        B = kwargs["B"]
        alpha = kwargs["alpha"]


    
