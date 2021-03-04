from magnon import *

import matplotlib.pyplot as plt
from matplotlib import rc
from time import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib.gridspec as gridspec

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


def plot_error():

    hs = np.load("../data/hs.npy")
    errs = np.load("../data/errs.npy")
    times = np.load("../data/times.npy")
    
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
    
def damping():

    h = 0.01
    S_0 = np.array([initial_cond(0.1,0.5)])

    alphas = np.array([0.1, 0.2, 0.5])
    for a in alphas:
        params = {'d':0,'J':0,'mu':1,'B':np.array([0,0,1.]),'alpha':a}
        spinsolver = MagnonSolver(0,S_0,10*np.pi,h,"RK4",**params)
        Ts, Xs = spinsolver()

        np.save(f"../data/X_a={a}.npy",Xs)
        np.save(f"../data/T_damp.npy",Ts) # only need the times one of the runs

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


if __name__ == "__main__":
    
    #compare_analyical()
    #error_analysis(10)
    #plot_error()
    #damping()
    plot_damping()
    