import numpy as np

class ODEStep:

    def __init__(self, f, h, y0, t0 = 0):
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

    def __call__(self):
        t,y,h = self.t, self.y, self.h
        
        y_p = y + h   * self.f(t, y)
        y_  = y + h/2 * (self.f(t,y) + self.f(t + h, y_p))

        self.y = y_
        self.t += h
        return self.y

class RK4Step(ODEStep):

    def __call__(self):
        t,y,h = self.t, self.y, self.h

        k1  = self.f( t      , y           )
        k2  = self.f( t + h/2, y + k1 * h/2)
        k3  = self.f( t + h/2, y + k2 * h/2)
        k4  = self.f( t + h  , y + k3 * h  )
    
        self.y =  y + h * (k1 + 2*k2 + 2*k3 + k4)/6
        self.t += h
        return self.y

class ODESolver:

    def __init__(self,f,t0,y0,tN,h,method = "Heun"):

        if method == "Heun":
            self.step = HeunStep(f,h,y0,t0)
        elif method == "RK4":
            self.step = RK4Step(f,h,y0,t0)
            raise NotImplementedError

        self.tN = tN 
        self.N  = int((tN - t0)/h)

        self.shape = np.shape(y0)
        self.Y = np.zeros((self.N + 1, self.shape[0], self.shape[1]))
        self.T = np.arange(t0, t0 + (self.N + 1) * h, h)

    def __call__(self):
        self.Y[0] = self.step.y

        for i in range(1,self.N + 1):
            self.step.h = min(self.step.h, self.tN - self.T[i-1])
            self.Y[i] = self.step()

        return self.T, self.Y
        
