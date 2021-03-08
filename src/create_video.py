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

import os 

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

def spin_snapshot(S):

    fig, ax = plt.subplots(figsize = (13,8))
    
    x = np.linspace(0,2,10)
    
    u = S[:,0]
    v = S[:,1]
    w = S[:,2]

    # this is a bit hacky

    for j in range(10):
        ax.quiver(x[j],0,u[j],w[j],
                  color = "black",
                  scale_units='xy',
                  scale = 1,
                  angles="xy",
                  headwidth = 2,
                  width = 0.002)
    #ax.set_xlabel(r"$S_x$")
    ax.set_ylabel(r"$S_z$")
    ax.grid(ls ="--")
    ax.set_ylim(0,1)
    ax.set_xlim(-0.25,2.25)

    plt.tight_layout()
    plt.close()
    return fig

def spin_video(S,title,savepath):

    n = 30 # use every 30th snapshot

    N = S[::n,0,0].size
    
    for i in range(N):
        fig = spin_snapshot(S[i,:,:])
        fig.savefig(savepath+"img{0:0=3d}.png".format(i))

    img_name = savepath + "img%03d.png"
    os.system(f"ffmpeg -framerate 20 -i {img_name} {title}.mp4")

    # remove all the pictures !

    for i in range(N):
        os.remove("/home/sondre/Pictures/figs_simulation/img{0:0=3d}.png".format(i))
 
if __name__ == "__main__":

    S = np.load("../data/X_coupled.npy")
    spin_video(S,"../fig/coupled_spins","/home/sondre/Pictures/figs_simulation/")
