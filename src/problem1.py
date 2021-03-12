from magnon import *
from time import time
from tqdm import tqdm

def S_xy(wt,a,b):
    x = a*np.cos(wt) - b*np.sin(wt)
    y = b*np.cos(wt) + a*np.sin(wt)

    return np.array([x,y])


def error_analysis(N):

    tN = 2 * np.pi    # simulate 1 period
    hs = np.logspace(-5,-1,N)
    
    S_0 = np.array([initial_cond(0.1,0.5)])

    S_a = S_xy(tN,S_0[0,0],S_0[0,1]) # analytical solution at endpoint
    
    params = {'d':0,'J':0,'mu':1,'B':np.array([0,0,1.]),'alpha':0}

    errs = np.zeros((2,2,N)) # global errors
    times = np.zeros((2,N))  # runtimes 
    
    for i, h_i in tqdm(enumerate(hs)):
        spinsolver_heun = MagnonSolver(0,S_0,tN,h_i,"Heun",**params)
        spinsolver_euler = MagnonSolver(0,S_0,tN,h_i,"Euler",**params)

        tic = time()
        _, X_heun = spinsolver_heun()
        toc = time()

        times[0,i] = toc - tic

        tic = time()
        _, X_euler = spinsolver_euler()
        toc = time()

        times[1,i] = toc - tic
        
        # using as a measure of the global error the
        # accumulated error at the endpoint.
        
        errs[0,0,i] = np.abs(S_a[0] - X_heun[-1,0,0])
        errs[0,1,i] = np.abs(S_a[1] - X_heun[-1,0,1])

        errs[1,0,i] = np.abs(S_a[0] - X_euler[-1,0,0])
        errs[1,1,i] = np.abs(S_a[1] - X_euler[-1,0,1])

    np.save("../data/hs.npy",hs)
    np.save("../data/errs.npy",errs)
    np.save("../data/times.npy",times)
    
def damping():

    h = 0.01
    S_0 = np.array([initial_cond(0.1,0.5)])

    alphas = np.array([0.1, 0.2, 0.5])
    for a in alphas:
        params = {'d':0,'J':0,'mu':1,'B':np.array([0,0,1.]),'alpha':a}
        spinsolver = MagnonSolver(0,S_0,10*np.pi,h,"Heun",**params)
        Ts, Xs = spinsolver()

        np.save(f"../data/X_a={a}.npy",Xs)
        np.save(f"../data/T_damp.npy",Ts) # only need the times one of the runs


if __name__ == "__main__":
    
    error_analysis(10)
    #damping()
    
