from magnon import *


def ground_states():

    num_spins = 10
    params = {'d':0.1,'J':1,'mu':1,'B':np.array([0,0,0]),'alpha':0.05}

    # choosing random initial angles
    thetas = np.random.random(num_spins) * np.pi * 2
    phis   = np.random.random(num_spins) * np.pi 
    
    S_0 = np.array([initial_cond(thetas[i],phis[i]) for i in range(num_spins)])

    spinsolver = MagnonSolver(0,S_0,20*np.pi,0.01,"Heun",**params)
    Ts,Xs = spinsolver()

    np.save(f"../data/S_gs_ferro.npy",Xs)
    np.save(f"../data/T_gs.npy",Ts)

    # Repeat for J < 0

    params = {'d':0.1,'J':-1,'mu':1,'B':np.array([0,0,0]),'alpha':0.05}

    spinsolver = MagnonSolver(0,S_0,20*np.pi,0.01,"Heun",**params)
    Ts,Xs = spinsolver()

    np.save(f"../data/S_gs_antiferro.npy",Xs)

def precession_uncoupled():

    num_spins = 10
    params = {'d':1,'J':0,'mu':1,'B':np.array([0,0,0]),'alpha':0}

    # choosing random initial angles
    thetas = np.random.random(num_spins) * np.pi * 2
    phis   = np.random.random(num_spins) * np.pi 
    
    S_0 = np.array([initial_cond(thetas[i],phis[i]) for i in range(num_spins)])

    spinsolver = MagnonSolver(0,S_0,20*np.pi,0.01,"Heun",**params)
    Ts,Xs = spinsolver()

    np.save("../data/X_precession.npy",Xs)

    # choosing only one tilted spin
    S_  = initial_cond(0.1,0.5) 
    S_0 = np.zeros((10,3))
    S_0[:,2] = 1 # initialise all others to point in the z direction
    S_0[0] = S_

    spinsolver = MagnonSolver(0,S_0,2*np.pi,0.01,"Heun",**params)
    Ts,Xs = spinsolver()

    np.save("../data/X_precession_tiltone.npy",Xs)

def precession_coupled():

    num_spins = 10
    params = {'d':1,'J':1,'mu':1,'B':np.array([0,0,0]),'alpha':0}

    # choosing only one tilted spin
    S_  = initial_cond(0.1,0.5) 
    S_0 = np.zeros((10,3))
    S_0[:,2] = 1 # initialise all others to point in the z direction
    S_0[0] = S_

    spinsolver = MagnonSolver(0,S_0,20*np.pi,0.01,"Heun",**params)
    Ts,Xs = spinsolver()

    np.save("../data/X_coupled.npy",Xs)

if __name__ == "__main__":

    #ground_states()
    #precession_uncoupled()
    #precession_coupled()
