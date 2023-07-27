import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_states = ListedColormap(["k","b","r"]) # custom colormap for the 3 states : 'forward', 'left' or 'right'

def plot_transition_matrix(ax, T, labels=["forward", "left", "right"]):
    """Routine to plot transition matrices.
    
    Parameters :
    ------------
    ax : matplotlib.Axes
        matplotlib axis where to plot the transition matrix.
    T : 2D square array
        transition probability matrix.
    labels : list of strings
        labels for each state of the transition matrix.
        Must have the same lenght as :T:.
        default : ["forward", "left", "right"]
    """
    assert T.ndim==2, f":T: must be 2D. You gave {T.ndim}D."
    assert T.shape[0]==T.shape[1], f":T: must be a square matrix. You gave {T.shape}."
    assert ((T>=0) * (T<=1)).all(), f":T: must be a matrix of probabilities. Some values you gave were not between 0 and 1."
    h = ax.imshow(T.T, vmin=0, vmax=1, cmap="plasma", origin="lower", aspect="equal")
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            ax.text(i,j, f"{T[i,j]:0.2f}", ha="center", va="center")
    ax.set_xticks([0,1,2], labels)
    ax.set_yticks([0,1,2], labels)
    ax.set_xlabel("Current step")
    ax.set_ylabel("Next step")
    return h

def plot_angle_sequence(ax, seq, states=None, probas=None, cmap=None):
    """Routine to plot a sequence of re-orientation angles.
    
    Parameters :
    ------------
    ax : matplotlib.Axes
        matplotlib axis where to plot the transition matrix.
    seq : 1D array of floats
        sequence of re-orientation angles (in degrees).
    states : 1D array of ints
        sequence bout types (0, 1, 2).
    probas : 1D array of floats
        probability associated with each bout.
    """
    assert seq.ndim==1, ":seq: should be 1D."
    cond_s, cond_p = states is not None, probas is not None
    if cond_s and cond_p:
        raise TypeError("only :states: or :probas: should be given. Not both.")
    elif cond_s:
        assert states.ndim==1, ":states: should be 1D."
        assert len(states)==len(seq), ":states: and :seq: should have the same lenght."
        c = states
        if cmap is None:
            cmap = cmap_states
            vmin, vmax = 0,2
    elif cond_p:
        assert probas.ndim==1, ":probas: should be 1D."
        assert len(probas)==len(seq), ":probas: and :seq: should have the same lenght."
        c = probas
        if cmap is None:
            cmap = "plasma"
            vmin, vmax = 0,1
    else:
        c = "k"
        vmin, vmax = None, None
    
    x = np.arange(len(seq))
    ax.plot(x, seq, color="grey")
    h = ax.scatter(x, seq, c=c, cmap=cmap, vmin=vmin, vmax=vmax)

    if cmap is not None:
        cbar = ax.figure.colorbar(h, ax=ax)
        if cond_p:
            cbar.set_label("Probability")
        elif cond_s:
            cbar.set_label("States")
            cbar.set_ticks([0.33,1,1.66], labels=["Forward","Left","Right"])
    
    ax.set_ylabel(r"$\delta \theta$ (degree)")
    ax.set_xlabel("Bout Number")