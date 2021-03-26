from ode import *
import numba as nb

# --------------------
# Physical constants :

J  = 0
d  = 0
mu = 1
B  = np.array([0,0,1])
alpha = 0    
gamma = 1

C = -gamma/(mu*(1+alpha**2))

e_z = np.array([0,0,1])
# --------------------

# Levi-civita tensor:
eijk = np.zeros((3,3,3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

def cross(A,B):
    """
    Cross product of A and B. Last dimension of B and A must be either 3 or 2.
    Some testing indicates that this is actually more efficient than np.cross. Se remarks in report.
    """
    return np.einsum('ijk,...j,...k',eijk,B,A)

def H(S):
    """
    Hamiltonian for array of spins S 
    
    Parameters
    ----------

    S : array
        Array of spins. 

    Returns
    -------

    H(S) : float
        Hamiltonian of system.
        
    """

    n = np.shape(S)[0] # number of spins
    ss = - J * np.sum([ S[i,:] @ ( S[i-1,:] + S[(i + 1) % n,:] ) for i in range(n)])
    s2 = - d * np.einsum('i,i',S[:,2],S[:,2])
    sb = - mu* np.einsum('ji,i->',S,B)

    return ss + s2 + sb

@nb.jit(nopython = True)
def djH(S,j,n):
    """
    Effective field for spin j
    
    Parameters
    ----------
    S : array
        Array of spins.
    j : int 
        Index for spin in question.
    n : int 
        Number of spins in total.

    Returns
    -------
    _ : array
        Effective field felt by spin number j.

    """
    ss = J * (S[j-1,:] + S[(j+1)%n,:] ) # sum over nearest neighbours
    return ss + 2* d*S[j,2] * e_z + mu * B

@nb.jit(nopython = True)
def gradH(S,n):
    """
    Effective field for all spins in S:
    
    Parameters
    ----------
    S : array
        Array of spins.
    n : int 
        Number of spins in total.

    Returns
    -------
    _ : array 
        Array of the effective fields for each particle.

    """
    dH = np.zeros((n,3))
    for j in range(n):
        dH[j,:] = djH(S,j,n)
    return dH

@nb.jit(nopython = True)
def f_llg(t,S):
    """
    Function defining the right hand side of the Landau Lifshitz Gilbert Equation:
                                dS
                                -- = f(t, S). 
                                dt
    
    Parameters
    ----------
    t : float 
        Time.
    S : array
        Array of spins of system. 

    Returns
    -------
    _ : array
        f(t,S)
  
    """
    
    n = np.shape(S)[0] # number of spins
    dH = gradH(S,n)
    
    return C * (np.cross(S,dH) + alpha * np.cross(S, np.cross(S, dH) ) )


def initial_cond(theta,phi):
    """
    Get a vector on the 2-sphere by specifying the azimuthal and polar angle.

    Parameters
    ----------
    theta : float
        Azimuthal angle.
    phih  : float
        Polar angle.

    Returns
    -------
    _ : array
        Position of point on sphere.  
    """

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.array([x,y,z])


class MagnonSolver(ODESolver):
    """
    
    Simple class for solving the LLG, derived from the normal ODESolver.
    The only important difference is that it allows for an extra dimension of 
    the variable y in the ODE. This is done in order to avoid flattening the array 
    of the spins. 
    
    Moreover, it unpacks the parameters to define the specific hamiltonian to use 
    in the system under consideration. This is done using global variables in order to 
    avoid having to pass the parameters through the function at each call.

    Note that if one version of the system is simulated one time, the functions 
    compiled with numba treats the global variables as constants. To update parameters in
    a single run, one has to use f_llg.recompile()


    """

    def __init__(self,t0,y0,tN,h,method = "Heun",**kwargs):
        super().__init__(f_llg,t0,y0,tN,h,method)

        self.shape = np.shape(y0)

        # each spin array is two dimensional
        
        self.Y = np.zeros((self.N + 2,self.shape[0], self.shape[1]))


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

