import matplotlib.pyplot as plt
from matplotlib import rc
from time import time
from mpl_toolkits.mplot3d import Axes3D

rc("text",usetex = True)
rc("font",family = "sans-serif")

fontsize = 25
newparams = {'axes.titlesize': fontsize,
             'axes.labelsize': fontsize,
             'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 
             'legend.fontsize': fontsize,
             'figure.titlesize': fontsize,
             'legend.handlelength': 1.5, 
             'lines.linewidth': 2,
             'lines.markersize': 7,
             'figure.figsize': (11, 7), 
             'figure.dpi':200,
             'text.usetex' : True,
             'font.family' : "sans-serif"
            }

plt.rcParams.update(newparams)

from ode import *
from magnon import *

def fa(t, X):
    # Example taken from beginning of Chapter II in
    # Hairer, Nørsett, Wanner
    # Solving Ordinary Differential Equations, vol I
    # X[0] = y1
    # X[1] = y1dot
    # X[2] = y2
    # X[3] = y2dot
    # Constants
    mu  = 0.012277471
    mu_ = 1 - mu
    # Variable parameters
    D1 = ((X[0] + mu )**2 + X[2]**2)**(3/2)
    D2 = ((X[0] - mu_)**2 + X[2]**2)**(3/2)
    # Derivatives
    dX0 = X[1]
    dX1 = X[0] + 2*X[3] - mu_*(X[0] + mu)/D1 - mu*(X[0] - mu_)/D2
    dX2 = X[3]
    dX3 = X[2] - 2*X[1] - mu_*X[2]/D1        - mu*X[2]/D2
    return np.array([dX0, dX1, dX2, dX3])

def ode_solver_test():

    t  = 0
    T  = 17.06521656015796
    X0 = np.array([
         0.994, # y1(0)
         0.0,   # y1'(0)
         0.0,   # y2(0)
        -2.001585106379082522405 # y2'(0)
        ])


    # Intergrate for a duration equal to 2T.
    # If the error after the first orbit is too large,
    # the second orbit will be completely off.
    # Try experimenting with the timestep.
    h  = 5e-4

    fasolver = ODESolver(fa,0,X0,2*T,h,"RK4")
    # Also, measure the time of the calculation
    tic = time()
    Ts, Xs = fasolver()
    toc = time()
    print('Number of steps: ', len(Ts))
    print('Simulation took ', toc - tic, ' seconds')

    fig = plt.figure(figsize = (12, 6))
    plt.plot(Xs[:,0], Xs[:,2])
    plt.tight_layout()
    fig.savefig("../fig/odetest.pdf")
    

def llg_test():
    
    h = 0.01

    S_0 = np.array([initial_cond(0.1,0.5)])

    params = {'d':0,'J':0,'mu':1,'B':np.array([0,0,1.]),'alpha':0}

    spinsolver = MagnonSolver(0,S_0,2*np.pi,h,"RK4",**params)

    Ts, Xs = spinsolver()
    
    plt.plot(Xs[:,0,0],Xs[:,0,1])
    plt.tight_layout()
    plt.axis("square")
    plt.savefig("../fig/llgtest.pdf")

    np.save("../data/T.npy",Ts)
    np.save("../data/S.npy",Xs)
    

    

    
