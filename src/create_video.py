from magnon import *
import matplotlib.pyplot as plt
from matplotlib import rc
from time import time
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from multiprocessing import Process, Pool

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

# making this global for testing
S = np.load("../data/X_coupled.npy")
S = S[::2,:,:]

def spin_snapshot(S):

    fig, ax = plt.subplots(figsize = (13,8))
    
    x = np.linspace(0,2,S[:,0].size) # stack all spins inside (0, 2)
    
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

def spin_snapshot_3d(S):

    u = S[:,0]
    v = S[:,1]
    w = S[:,2]
    
    fig = plt.figure(figsize = (15,10))
    ax = fig.gca(projection='3d')

    ax.set_xlim3d(-0.3, 0.3)
    ax.set_ylim3d(-1.5, 4.0)
    ax.set_zlim3d(-0.1,1.1)

    spins = np.size(u[:])

    x = np.zeros_like(u)
    y = np.linspace(0,3,spins)
    z = np.zeros_like(u)

    for i in range(spins):
        ax.quiver(x[i],y[i],z[i],u[i],v[i],w[i],arrow_length_ratio = 0.01, color = "black") 

    ax.azim = 20
    ax.dist = 10
    ax.elev = 25

    plt.xticks(rotation=-45)
    plt.yticks(rotation=45)

    ax.set_xlabel("$S_x$",labelpad = 40)
    ax.set_ylabel(r"$S_y + \mathrm{offset}$",labelpad = 40)
    ax.set_zlabel("$S_z$")

    plt.tight_layout()
    plt.close()
    return fig


def save_fig(i, savepath = "/home/sondre/Pictures/figs_simulation/"):
    fig = spin_snapshot(S[i])
    fig.savefig(savepath+"img{0:0=3d}.png".format(i))

def spin_video(S,title,savepath,dim3 = False):

    N = 400        # use 400 images per video
    S = S[::2,:,:] # use only every second image
    
    for i in tqdm(range(N)):
        if dim3:
            fig = spin_snapshot_3d(S[i,:,:])
        else:
            fig = spin_snapshot(S[i,:,:])
        fig.savefig(savepath+"img{0:0=3d}.png".format(i))

    img_name = savepath + "img%03d.png"
    os.system(f"ffmpeg -framerate 60 -i {img_name} {title}.mp4")

    # remove all the pictures !

    for i in range(N):
        os.remove("/home/sondre/Pictures/figs_simulation/img{0:0=3d}.png".format(i))

def spin_video_parallel(title,savepath):
    """
    Function for (trying) to save pictures in a parallel manner
    to save some time. Very hacky solution, which albeit is not 
    particularly general as makes it hard to give more arguments to the function being called 
    in paralllel.
    """
    N = 400            # use 400 images per video
    idx = np.arange(N) # indexes for images to save
    
    with Pool(8) as pool:
        pool.map(save_fig,idx)

    img_name = savepath + "img%03d.png"
    os.system(f"ffmpeg -framerate 60 -i {img_name} {title}.mp4")

    # remove all the pictures !

    for i in range(N):
        os.remove("/home/sondre/Pictures/figs_simulation/img{0:0=3d}.png".format(i))
 
if __name__ == "__main__":

    #spin_video_parallel("../fig/coupled_spins_para","/home/sondre/Pictures/figs_simulation/")
    
    S = np.load("../data/X_coupled.npy")
    spin_video(S,"../fig/coupled_spins","/home/sondre/Pictures/figs_simulation/",False)

    #S = np.load("../data/X_coupled_alpha=0.1.npy")
    #spin_video(S,"../fig/coupled_spins_damped","/home/sondre/Pictures/figs_simulation/")

    #S = np.load("../data/X_coupled_anti.npy")
    #spin_video(S,"../fig/coupled_spins_anti","/home/sondre/Pictures/figs_simulation/")
