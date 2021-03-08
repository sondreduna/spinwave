from magnon import *

import matplotlib.pyplot as plt
from matplotlib import rc
from time import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import matplotlib as mpl

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



def plot_ground_state():

    S_ferro = np.load("../data/S_gs_ferro.npy")
    S_anti  = np.load("../data/S_gs_antiferro.npy")
    Ts      = np.load("../data/T_gs.npy")

    # plotting for J > 0 
    fig = plt.figure()

    plt.title("$J > 0$")

    plt.plot(Ts, S_ferro[:,:,2])
    plt.xlabel("$t$")

    plt.ylabel("$S_z(t)$")
    plt.grid(ls = "--")

    plt.tight_layout()

    fig.savefig("../fig/gs_ferro.pdf")

    fig = plt.figure()

    plt.title("$J < 0$")

    plt.plot(Ts, S_anti[:,:,2])
    plt.xlabel("$t$")

    plt.ylabel("$S_z(t)$")
    plt.grid(ls = "--")

    plt.tight_layout()

    fig.savefig("../fig/gs_antiferro.pdf")
    

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

def plot_precessions():

    S = np.load("../data/X_precession.npy")
    
    fig = plt.figure()

    plt.plot(S[:,:,0],S[:,:,1])

    plt.xlabel(r"$S_x$")
    plt.ylabel(r"$S_y$")

    plt.axis("scaled")
    plt.grid(ls = "--")
    plt.tight_layout()

    fig.savefig("../fig/precession_xy.pdf")

    # loading the data with one spin tilted
    S = np.load("../data/X_precession_tiltone.npy")
    
    n = 30 # steps in time
    fig = plt.figure(figsize = (20,11))

    gs = gridspec.GridSpec(8, 2)

    ax = plt.subplot(gs[:7, :])

    # get 10 positions used for plotting
    y = np.zeros(np.shape(S[::n,0,0]))
    x = np.linspace(0,2,10) 

    u = S[::n,:,0]
    v = S[::n,:,1]
    w = S[::n,:,2]

    cm = mpl.cm.get_cmap('viridis')

    N = len(y)
    # this is a bit hacky
    for i in range(N):
        for j in range(10):
            ax.quiver(x[j],y[i],u[i,j],w[i,j],
                      color = cm(i/N),
                      scale_units='xy',
                      scale = 1,
                      angles="xy",
                      headwidth = 2,
                      width=  0.005)
    #ax.set_xlabel(r"$S_x$")
    ax.set_ylabel(r"$S_z$")
    ax.grid(ls ="--")
    ax.set_ylim(0,1)
    ax.set_xlim(-0.25,2.25)

    plt.tight_layout()
    #ax.set_xticks([])
    #ax.set_yticks([])
    cb_ax = plt.subplot(gs[7,:])    
    norm= mpl.colors.Normalize(vmin=0,vmax=2 * np.pi)

    cb1 = mpl.colorbar.ColorbarBase(cb_ax, cmap=cm,norm = norm,
                                orientation='horizontal')
    cb1.set_label(r'$t$')
    plt.tight_layout()

    fig.savefig("../fig/10_precessions.pdf")

def plot_coupled():
    # loading the data with one spin tilted, with coupling
    S = np.load("../data/X_coupled.npy")
    
    n = 50 # steps in time
    fig = plt.figure(figsize = (20,11))

    gs = gridspec.GridSpec(8, 2)

    ax = plt.subplot(gs[:7, :])

    # get 10 positions used for plotting
    y = np.zeros(np.shape(S[::n,0,0]))
    x = np.linspace(0,2,10) 

    u = S[::n,:,0]
    v = S[::n,:,1]
    w = S[::n,:,2]

    cm = mpl.cm.get_cmap('viridis')

    N = len(y)
    # this is a bit hacky
    for i in range(N):
        for j in range(10):
            ax.quiver(x[j],y[i],u[i,j],w[i,j],
                      color = cm(i/N),
                      scale_units='xy',
                      scale = 1,
                      angles="xy",
                      headwidth = 2,
                      width=  0.002)
    #ax.set_xlabel(r"$S_x$")
    ax.set_ylabel(r"$S_z$")
    ax.grid(ls ="--")
    ax.set_ylim(0,1)
    ax.set_xlim(-0.25,2.25)

    plt.tight_layout()
    #ax.set_xticks([])
    #ax.set_yticks([])
    cb_ax = plt.subplot(gs[7,:])    
    norm= mpl.colors.Normalize(vmin=0,vmax=2 * np.pi)

    cb1 = mpl.colorbar.ColorbarBase(cb_ax, cmap=cm,norm = norm,
                                orientation='horizontal')
    cb1.set_label(r'$t$')
    plt.tight_layout()

    fig.savefig("../fig/10_precessions_coupled.pdf")    

if __name__ == "__main__":

    #ground_states()
    #plot_ground_state()
    #precession_uncoupled()
    plot_precessions()
    #precession_coupled()
    plot_coupled()
