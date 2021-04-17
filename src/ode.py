import numpy as np
from tqdm import trange

class ODEStep:

    """
    Abstract class for doing an explicit step with an ODE
    solver.

    Attributes
    ----------
    f : function
        Function defining ODE.
    h : float
        Step length.
    y : array
        Current y value.
    t : float
        Current time value.

    Methods
    -------
    
    __call__()
        step forward one unit of h in time.
    """

    def __init__(self, f, h, y0, t0):
        self.f, self.h = f, h
        self.t = t0
        self.y = y0

    def __call__(self):
        """
        Function for stepping forward
        one unit of h in time

        """
        raise NotImplementedError
    
class HeunStep(ODEStep):
    """
    Derived class of ODEStep. Heun's method.

    """

    def __call__(self):
        t,y,h = self.t, self.y, self.h

        y_e = self.f(t, y)
        y_p = y + h * y_e

        self.y = y + h/2 * (y_e + self.f(t + h, y_p))
        self.t += h
        return self.y


class RK4Step(ODEStep):
    """
    Derived class of ODEStep. RK4 method.
    """

    def __call__(self):
        t,y,h = self.t, self.y, self.h

        k1  = self.f( t      , y           )
        k2  = self.f( t + h/2, y + k1 * h/2)
        k3  = self.f( t + h/2, y + k2 * h/2)
        k4  = self.f( t + h  , y + k3 * h  )
    
        self.y =  y + h * (k1 + 2*k2 + 2*k3 + k4)/6
        self.t += h
        return self.y

class EulerStep(ODEStep):
    """
    Derived class of ODEStep. Euler's method.
    """

    def __call__(self):
        t,y,h = self.t, self.y, self.h

        self.y = y + h*self.f(t,y)
        self.t += h
        return self.y

class ODESolver:
    """
    Class for solving ODE. 

    Attributes
    ----------
    step : ODEStep
        Object used to store current y and time value, and advance in time.
    tN : float
        Time to integrate up to.
    N : int 
        Number of steps.
    shape : tuple(int)
        Shape of y
    Y : array
        Array of y-values.
    T : array
        Array of t-values.

    Methods
    -------
    __call__(verbose)
        Solve the ODE.
     
    """

    def __init__(self,f,t0,y0,tN,h,method = "Heun"):

        if method == "Heun":
            self.step = HeunStep(f,h,y0,t0)
        elif method == "RK4":
            self.step = RK4Step(f,h,y0,t0)
        elif method == "Euler":
            self.step = EulerStep(f,h,y0,t0)
        else:
            raise NotImplementedError

        self.tN = tN
        self.N  = int((tN - t0)/h)
        
        self.shape = np.size(y0) 
        
        self.Y = np.zeros((self.N + 2, self.shape))
        self.T = np.zeros(self.N + 2)

    def __call__(self,verbose = False):
        """
        Parameters
        ----------
        verbose : boolean
            True  : show a progressbar of the integration
            False : no progressbar. 
        """
        self.Y[0] = self.step.y
        if verbose:
            for i in trange(1,self.N+2):
                self.step.h = min(self.step.h, self.tN - self.T[i-1])
                self.Y[i] = self.step()
                self.T[i] = self.T[i-1] + self.step.h
        else:
            for i in range(1,self.N+2):
                self.step.h = min(self.step.h, self.tN - self.T[i-1])
                self.Y[i] = self.step()
                self.T[i] = self.T[i-1] + self.step.h 

        return self.T, self.Y
        
