import matplotlib.pyplot as plt
from matplotlib import rc
from time import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np

rc("text",usetex = True)
rc("font",family = "sans-serif")

fontsize = 24
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

## Problem 1 

def S_xy(wt,a,b):
    x = a*np.cos(wt) - b*np.sin(wt)
    y = b*np.cos(wt) + a*np.sin(wt)

    return np.array([x,y])

def compare_analytical():

    T = np.load("../data/T.npy")
    S = np.load("../data/S.npy")
    
    S_a = S_xy(T,S[0,0,0],S[0,0,1])

    fig = plt.figure()
    
    plt.plot(T[::10],S[::10,0,0],".",label = r"$S_{x,\mathrm{heun}}$",color ="red")
    plt.plot(T,S_a[0], label  = r"$S_{x,\mathrm{exact}}$",color ="red", ls ="--")

    plt.plot(T[::10],S[::10,0,1],".",label = r"$S_{y,\mathrm{heun}}$",color ="blue")
    plt.plot(T,S_a[1], label  = r"$S_{y,\mathrm{exact}}$",color ="blue", ls ="--")

    plt.xlabel(r"$\omega t$")
    plt.ylabel(r"$S$")
    plt.grid(ls ="--")
    
    plt.legend()
    plt.tight_layout()

    fig.savefig("../fig/comparison.pdf")

    fig2 = plt.figure()
    
    plt.plot(T,S[:,0,0] - S_a[0],label = r"$S_{x,\mathrm{heun}} -S_{x,\mathrm{exact}} $",color ="red")

    plt.plot(T,S[:,0,1] - S_a[1],label = r"$S_{y,\mathrm{heun}} -S_{y,\mathrm{exact}} $",color ="blue")

    plt.xlabel(r"$\omega t$")
    plt.ylabel(r"$\Delta S$")
    plt.grid(ls ="--")
    
    plt.legend()
    plt.tight_layout()

    fig2.savefig("../fig/comparison_diff.pdf")

def plot_error():

    hs = np.load("../data/hs_j.npy")
    errs = np.load("../data/errs_j.npy")
    times = np.load("../data/times_j.npy")
    
    fig1, ax = plt.subplots()

    plt.grid(ls = "--")
    ax.set_yscale("log")
    ax.set_xscale("log")
    
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"Global error")

    lns1 = ax.plot(hs,
            errs[0,0,:],
            label = "Heun $x$",
            ls = "--",
            color="red",
            marker="o")
    lns2 = ax.plot(hs,
            errs[0,1,:],
            label = "Heun $y$",
            ls = "--",
            color="blue",
            marker = "o")

    lns3 = ax.plot(hs,hs**2, label = r"$\sim h^2$",ls = "-.",color = "green")

    ax2 = ax.twinx()

    ax2.set_ylabel(r"Runtime [s]")
    ax2.set_yscale("log")
    lns4 = ax2.plot(hs,times[0,:],".",label = "Heun runtime",color = "black")

    # Gathering all the labels
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    
    # Put a legend to the right of the current axis
    ax2.legend(lns,labs, loc='upper center', bbox_to_anchor=(0.5, 1.125),
          ncol=2, fancybox=True, shadow=True)

    fig1.tight_layout()
    fig1.savefig("../fig/err_heun.pdf")

    fig2, ax = plt.subplots()

    plt.grid(ls ="--")

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"Global error")

    lns1 = ax.plot(hs,
            errs[1,0,:],
            label = "Euler $x$",
            ls = "--",
            color = "red",
            marker = "o")
    lns2 = ax.plot(hs,
            errs[1,1,:],
            label = "Euler $y$",
            ls = "--",
            color = "blue",
            marker = "o")

    ax.set_yscale("log")
    ax.set_xscale("log")
    lns3 = ax.plot(hs,hs, label = r"$\sim h$",ls = "-.",color = "green")

    ax2 = ax.twinx()

    ax2.set_ylabel(r"Runtime [s]")
    ax2.set_yscale("log")
    lns4 = ax2.plot(hs,times[0,:],".",label = "Euler runtime",color = "black")

    # gathering all the labels
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]

    # Put a legend to the right of the current axis
    ax2.legend(lns,labs, loc='upper center', bbox_to_anchor=(0.5, 1.125),
          ncol=2, fancybox=True, shadow=True)

    fig2.tight_layout()
    fig2.savefig("../fig/err_euler.pdf")
def exp_fit(T,alpha,S_0):

    return np.linalg.norm(S_0[:2]) * np.exp(-alpha*T)
        
def plot_damping():

    X1 = np.load("../data/X_a=0.1.npy")
    X2 = np.load("../data/X_a=0.2.npy")
    X3 = np.load("../data/X_a=0.5.npy")
    T  = np.load("../data/T_damp.npy")

    fig = plt.figure(figsize = (15,10))
    gs = gridspec.GridSpec(2,2)

    ax1 = plt.subplot(gs[0,:])

    ax1.set_title(r"$\alpha = 0.1$")
    ax1.plot(T,X1[:,0,0], label =r"$S_x$", color = "red")
    ax1.plot(T,X1[:,0,1], label =r"$S_y$", color = "blue")

    
    fit_1 = exp_fit(T,0.1,X1[0,0,:])
    fit_2 = exp_fit(T,0.2,X2[0,0,:])
    fit_3 = exp_fit(T,0.5,X3[0,0,:])
    
    ax1.plot(T,fit_1,
             label= r"$\sqrt{S_x^2 + S_y^2}\exp{\left(- \alpha \omega t \right)}$",
             color ="blue",
             ls = "--",lw = 1)
    ax1.plot(T,-fit_1,
             label= r"$-\sqrt{S_x^2 + S_y^2}\exp{\left(- \alpha \omega t\right)}$",
             color ="red",
             ls = "--",lw = 1)
    ax1.set_xlabel(r"$\omega t$")
    ax1.set_ylabel(r"$S$")
    ax1.grid(ls = "--")

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.50),
          ncol=4, fancybox=True, shadow=True)

    ax2 = plt.subplot(gs[1,0])

    ax2.set_title(r"$\alpha = 0.2$")
    ax2.plot(T,X2[:,0,0], label =r"$S_x$", color = "red")
    ax2.plot(T,X2[:,0,1], label =r"$S_y$", color = "blue")
    ax2.plot(T,fit_2,
             color ="blue",
             ls = "--",lw = 1)
    ax2.plot(T,-fit_2,
             color ="red",
             ls = "--",lw = 1)
    ax2.set_xlabel(r"$\omega t$")
    ax2.set_ylabel(r"$S$")
    ax2.grid(ls = "--")

    ax3 = plt.subplot(gs[1,1])

    ax3.set_title(r"$\alpha = 0.5$")
    ax3.plot(T,X3[:,0,0], label =r"$S_x$", color = "blue")
    ax3.plot(T,X3[:,0,1], label =r"$S_y$", color = "red")
    ax3.plot(T,fit_3,
             color ="blue",
             ls = "--",lw = 1)
    ax3.plot(T,-fit_3,
             color ="red",
             ls = "--",lw = 1)
    ax3.set_xlabel(r"$\omega t$")
    ax3.set_ylabel(r"$S$")
    ax3.grid(ls = "--")
    plt.tight_layout()

    plt.savefig("../fig/damped_precession.pdf")    


## Problem 2


def plot_ground_state():

    S_ferro = np.load("../data/S_gs_ferro.npy")
    S_anti  = np.load("../data/S_gs_antiferro.npy")
    Ts      = np.load("../data/T_gs.npy")

    # plotting for J > 0 
    fig, ax = plt.subplots(ncols = 2, figsize = (20,8),sharey=True)

    ax[0].set_title("$J > 0$")

    ax[0].plot(Ts, S_ferro[:,:,2])
    ax[0].set_xlabel("$t$")
    ax[0].set_ylabel("$S_z(t)$")
    ax[0].grid(ls = "--")

    ax[1].set_title("$J < 0$")
    ax[1].plot(Ts, S_anti[:,:,2])
    ax[1].set_xlabel("$t$")
    ax[1].grid(ls = "--")

    plt.tight_layout()
    fig.savefig("../fig/gs.pdf")


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


def plot_heat_timeevo():

    S1 =  np.load("../data/X_coupled.npy")
    S2 =  np.load("../data/X_coupled_alpha=0.05.npy")

    x1 = S1[0:4000:,:,0]
    x2 = S2[0:4000:,:,0]
    
    fig = plt.figure(figsize = (20,12))

    gs = gridspec.GridSpec(8, 2)

    ax = plt.subplot(gs[:7, 0])

    ax.set_title(r"$\alpha = 0$")
    ax.imshow(x1)
    ax.set_xlabel(r"Particle index")
    ax.set_ylabel(r"Timestep")

    ax.set_aspect(0.003)

    ax = plt.subplot(gs[:7, 1])

    ax.set_title(r"$\alpha = 0.05$")
    im = ax.imshow(x2)
    ax.set_xlabel(r"Particle index")
    ax.set_ylabel(r"Timestep")

    ax.set_aspect(0.003)
    
    cm = mpl.cm.get_cmap('viridis')
    
    cb_ax = plt.subplot(gs[7,:])    
    norm= mpl.colors.Normalize(vmin=min(x1[:,0]),vmax= max(x1[:,0]))
    cb1 = mpl.colorbar.ColorbarBase(cb_ax, cmap=cm,norm = norm,
                                orientation='horizontal')
    cb1.set_label(r'$S_x$')

    plt.tight_layout()

    fig.savefig("../fig/damped_vs_undamped.pdf")
