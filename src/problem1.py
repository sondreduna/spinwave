from magnon import *

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
