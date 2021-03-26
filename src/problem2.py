from magnon import *


def ground_states():

    num_spins = 10
    params = {'d':0.1,'J':1.0,'mu':1,'B':np.array([0,0,0]),'alpha':0.05}

    # choosing random initial angles
    thetas = np.random.random(num_spins) * np.pi * 2
    phis   = np.random.random(num_spins) * np.pi 
    
    S_0 = np.array([initial_cond(t,f) for (t,f) in zip(thetas,phis)])

    spinsolver = MagnonSolver(0,S_0,20*np.pi,0.001,"Heun",**params)
    Ts,Xs = spinsolver(True)

    np.save(f"../data/S_gs_ferro.npy",Xs)
    np.save(f"../data/T_gs.npy",Ts)
    
    # Repeat for J < 0
    
    params["J"] = -1.0
    spinsolver = MagnonSolver(0,S_0,20*np.pi,0.001,"Heun",**params)

    # Recompile the compiled functions to update the parameter correctly !
    
    djH.recompile()
    gradH.recompile()
    f_llg.recompile()
    
    Ts,Xs = spinsolver(True)

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
    Ts,Xs = spinsolver(True)

    np.save("../data/X_precession_tiltone.npy",Xs)

def precession_coupled():

    num_spins = 10
    params = {'d':1,'J':1,'mu':1,'B':np.array([0,0,0]),'alpha':0}

    # choosing only one tilted spin
    S_  = initial_cond(np.pi/6,0.5) 
    S_0 = np.zeros((num_spins,3))
    S_0[:,2] = 1 # initialise all others to point in the z direction
    S_0[0] = S_

    spinsolver = MagnonSolver(0,S_0,10*np.pi,0.01,"Heun",**params)
    Ts,Xs = spinsolver(True)

    np.save("../data/X_coupled.npy",Xs)


def precession_coupled_damped():
    
    num_spins = 10
    params = {'d':1,'J':1,'mu':1,'B':np.array([0,0,0]),'alpha':0.1}

    # choosing only one tilted spin
    S_  = initial_cond(0.1,0.5) 
    S_0 = np.zeros((num_spins,3))
    S_0[:,2] = 1 # initialise all others to point in the z direction
    S_0[0] = S_

    spinsolver = MagnonSolver(0,S_0,10*np.pi,0.01,"Heun",**params)
    Ts,Xs = spinsolver(True)

    np.save("../data/X_coupled_alpha=0.1.npy",Xs)


def precession_coupled_anti():
    
    num_spins = 10
    params = {'d':1,'J':-1,'mu':1,'B':np.array([0,0,0]),'alpha':0}

    # choosing only one tilted spin
    S_  = initial_cond(0.1,0.5) 
    S_0 = np.zeros((10,3))
    S_0[:,2] = 1 # initialise all others to point in the z direction
    S_0[0] = S_

    spinsolver = MagnonSolver(0,S_0,10*np.pi,0.01,"Heun",**params)
    Ts,Xs = spinsolver(True)

    np.save("../data/X_coupled_anti.npy",Xs)


def many_spins_coupled():
    num_spins = 201
    params = {'d':1,'J':1,'mu':1,'B':np.array([0,0,0]),'alpha':0}

    # choosing only one tilted spin
    #S_  = initial_cond(0.4,0.5) 
    #S_0 = np.zeros((num_spins,3))
    #S_0[:,2] = 1 # initialise all others to point in the z direction
    #S_0[num_spins//2] = S_ # set the perturbation on the middle point

    # random inital perturbation:
    angs = np.random.random((2,num_spins))*2*np.pi
    S_0  = np.array([initial_cond(t,p) for (t,p) in angs.T])
    
    spinsolver = MagnonSolver(0,S_0,10*np.pi,0.01,"Heun",**params)
    Ts,Xs = spinsolver(True)

    np.save("../data/X_coupled200.npy",Xs)
    
if __name__ == "__main__":

    ground_states()
    #precession_uncoupled()
    #precession_coupled()
    #precession_coupled_damped()
    #precession_coupled_anti()
    #many_spins_coupled()
