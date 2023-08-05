import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import interact
from sklearn.preprocessing import scale
from Helper_Functions.OrthoViewer import OrthoAxes
import itertools


def plot_dff_traces(signal,times,indices,dff):
    """Plot signal and a dff trace selected with a slider from a list of possible indices.

    Parameters :
    ------------
    signal : 1d array
        external signal, shape (n_timepoints)
        
    times : 1d array
        timepoints of signal and dff, shape (n_timepoints)

    indices : 1d array
        indices of the neurons that can be selected with the slider
     
    dff : 2d array
        rescaled fluorescence traces for all neurons, shape (n_neurons,n_timepoints)
    """
    @interact(neuron=(0,len(indices)-1,1))
    def _plot_dff_traces(neuron=0):
        plt.close()
        fig,ax=plt.subplots()
        ax.plot(times,scale(signal),label='external signal')
        ax.plot(times,scale(dff[indices[neuron]]),'k',label='neuron $\Delta F/F$')
        ax.set_xlim(times[0],times[-1])
        ax.set_xlabel('time (s)')
        ax.legend()
        plt.show()
        
        
def plot_dff_raster(signal,times,indices,dff):
    """Plot dff of neurons with certain indices in a raster plot, also plot signal for comparison.

    Parameters :
    ------------
    signal : 1d array
        external signal, shape (n_timepoints)
        
    times : 1d array
        timepoints of signal and dff, shape (n_timepoints)

    dff : 2d array
        rescaled fluorescence traces for all neurons, shape (n_neurons,n_timepoints)
    
    indices : 1d array
        indices of the neurons that can be selected with the slider
    """
    fig,ax=plt.subplots()
    divider=make_axes_locatable(ax)
    tax=divider.append_axes('top',size='10%',pad=0.1,sharex=ax)
    cax=divider.append_axes('right',size='2%',pad=0.1)

    dt=np.mean(np.diff(times))
    extent=[times[0]-dt/2,times[-1]+dt/2,len(indices)-0.5,-0.5]
    cmap='inferno'

    tax.imshow(signal.reshape(1,-1),cmap=cmap,aspect='auto',interpolation='none',extent=extent)
    tax.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    tax.set_title('signal')

    im=ax.imshow(dff[indices],cmap=cmap,interpolation='none',aspect='auto',extent=extent)
    cbar=plt.colorbar(im,cax=cax)
    cbar.set_label(r'$\Delta F/F$')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('neurons')
    plt.show()
    
    
def plot_brain_layers(coords,opacity,mask=None,alpha_out=0.5,c='blue',s=4):
    """Plot neuron positions in a single layer that can be selected with a slider.
    Opacity values are given as inputs. They are cropped to positive valuesand rescaled to the range [0,1].
    Optionally a mask can be given so that neurons outside the mask are plotted in light grey.
    
    Parameters :
    ------------
    coords : 2d array
        neuron positions, shape (3,n_neurons)
        
    opacity : 1d array
        opacity values for each neuron

    mask : 1d array
        boolean array to select which neurons will be plotted in color
        default :  None
        
    alpha_out : float
        opacity of neurons outside the mask
        default : 0.5
        
    c : string
        color of the dots
        default : 'blue'
        
    s : float
        size of the dots
        default : 4
    """
    zs=np.unique(coords[2])
    zs=-np.sort(-zs)
    alpha=opacity/np.max(opacity)
    alpha[alpha<0]=0
    @interact(layer=(0,len(zs)-1,1))
    def _plot_brain_layer(layer=0):
        plt.close()
        fig,ax=plt.subplots()
        
        if mask is None:
            x_layer=coords[0,coords[2]==zs[layer]]
            y_layer=coords[1,coords[2]==zs[layer]]
            alpha_layer=alpha[coords[2]==zs[layer]]
            if len(x_layer)>0:
                ax.scatter(x_layer,y_layer,s=s,c=c,alpha=alpha_layer,edgecolors='none')
        
        else:
            x_layer=coords[0,(coords[2]==zs[layer])&(~mask)]
            y_layer=coords[1,(coords[2]==zs[layer])&(~mask)]
            if len(x_layer)>0:
                ax.scatter(x_layer,y_layer,s=s,c='lightgrey',alpha=alpha_out,edgecolors='none')

            x_layer=coords[0,(coords[2]==zs[layer])&(mask)]
            y_layer=coords[1,(coords[2]==zs[layer])&(mask)]
            alpha_layer=alpha[(coords[2]==zs[layer])&(mask)]
            if len(x_layer)>0:
                ax.scatter(x_layer,y_layer,s=s,c=c,alpha=alpha_layer,edgecolors='none')

        ax.set_aspect('equal')
        ax.set_xlim(np.min(coords[0]),np.max(coords[0]))
        ax.set_ylim(np.min(coords[1]),np.max(coords[1]))
        ax.set_title('$z = $'+str(round(zs[layer]))+' μm')
        ax.set_xlabel('$x$ (μm)')
        ax.set_ylabel('$y$ (μm)')
        plt.show()
        
        
def plot_brain_projections(coords,opacity,mask=None,alpha_out=0.1,c='blue',s=2,interactive=False):
    """Plot neuron positions projected in three orthogonal planes.
    Opacity values are given as inputs. They are cropped to positive values and rescaled to the range [0,1].
    Optionally a mask can be given so that neurons outside the mask are plotted in light grey.

    Parameters :
    ------------
    coords : 2d array
        neuron positions, shape (3,n_neurons)
        
    opacity : 1d array
        opacity values for each neuron

    mask : 1d array
        boolean array to select which neurons will be plotted in color
        default :  None
        
    alpha_out : float
        opacity of neurons outside the mask
        default : 0.1
        
    c : string
        color of the dots
        default : 'blue'
        
    s : float
        size of the dots
        default : 2
        
    interactive : boolean
        flag for interactive plotting
        default : False
    """
    alpha=opacity/np.max(opacity)
    alpha[alpha<0]=0
    fig=plt.figure()
    ortho=OrthoAxes(fig,coords,interactive=interactive)
    if mask is None:
        ortho.scatter(coords,c=c,alpha=alpha,s=s)
    else:
        alpha=alpha[mask]
        ortho.scatter(coords[:,~mask],c='lightgrey',alpha=alpha_out,s=s)
        ortho.scatter(coords[:,mask],c=c,alpha=alpha,s=s)
    plt.show()
    if interactive:
        return ortho   
    
    
def plot_brain_projections_colorbar(coords,color,mask,cmap='Blues',alpha_in=0.3,alpha_out=0.1,s=3,vmin=0,vmax=10):
    """Plot neuron positions projected in three orthogonal planes, with colorbar.
    Color values are given as inputs. 
    Neurons outside the mask are plotted in light grey.

    Parameters :
    ------------
    coords : 2d array
        neuron positions, shape (3,n_neurons)
        
    color : 1d array
        color values for each neuron

    mask : 1d array
        boolean array to select which neurons will be plotted in color
        default :  None
       
    cmap : string
        colormap
        default : 'Blues'
        
    alpha_in : float
        opacity of neurons inside the mask, of all neurons if no mask is given
        default : 0.3
        
    alpha_out : float
        opacity of neurons outside the mask
        default : 0.1
        
    s : float
        size of the dots
        default : 2
        
    interactive : boolean
        flag for interactive plotting
        default : False
    """
    fig,ax=plt.subplots()
    divider=make_axes_locatable(ax)
    ax2=divider.append_axes('right',size='40%',pad=0,sharey=ax)
    ax3=divider.append_axes('top',size='20%',pad=0,sharex=ax)
    cax=divider.append_axes('right',size='5%',pad=0)

    ax.scatter(coords[0,~mask],coords[1,~mask],s=s,c='lightgrey',alpha=alpha_out,edgecolors='none')
    im=ax.scatter(coords[0,mask],coords[1,mask],s=s,c=color[mask],vmin=vmin,vmax=vmax,cmap=cmap,alpha=alpha_in,edgecolors='none')

    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (μm)')
    ax.set_ylabel('$y$ (μm)')
    
    ax2.scatter(coords[2,~mask],coords[1,~mask],s=s,c='lightgrey',alpha=alpha_out,edgecolors='none')
    ax2.scatter(coords[2,mask],coords[1,mask],s=s,c=color[mask],vmin=vmin,vmax=vmax,cmap=cmap,alpha=alpha_in,edgecolors='none')

    ax2.tick_params(left=False,labelleft=False)
    ax2.set_aspect('equal')
    ax2.set_xlabel('$z$ (μm)')
    
    ax3.scatter(coords[0,~mask],coords[2,~mask],s=s,c='lightgrey',alpha=alpha_out,edgecolors='none')
    ax3.scatter(coords[0,mask],coords[2,mask],s=s,c=color[mask],vmin=vmin,vmax=vmax,cmap=cmap,alpha=alpha_in,edgecolors='none')

    ax3.tick_params(bottom=False,labelbottom=False)
    ax3.set_aspect('equal')
    ax3.set_ylabel('$z$ (μm)')

    cbar=plt.colorbar(im,cax=cax)
    cbar.set_label('colorbar')
    plt.show()
    

def plot_brain_projections_color(coords,color,mask=None,cmap='coolwarm',alpha_in=0.5,alpha_out=0.1,s=2,interactive=False):
    """Plot neuron positions projected in three orthogonal planes.
    Color values are given as inputs.
    Optionally a mask can be given so that neurons outside the mask are plotted in light grey.

    Parameters :
    ------------
    coords : 2d array
        neuron positions, shape (3,n_neurons)
        
    color : 1d array
        color values for each neuron

    mask : 1d array
        boolean array to select which neurons will be plotted in color
        default :  None
        
    cmap : string
        colormap
        default : 'coolwarm'
       
    alpha_in : float
        opacity of neurons inside the mask, of all neurons if no mask is given
        default : 0.5
        
    alpha_out : float
        opacity of neurons outside the mask
        default : 0.1
        
    s : float
        size of the dots
        default : 2
        
    interactive : boolean
        flag for interactive plotting
        default : False
    """
    fig=plt.figure()
    ortho=OrthoAxes(fig,coords,interactive=interactive)
    if mask is None:
        ortho.scatter(coords,c=color,cmap=cmap,
                      vmin=-np.std(color),vmax=np.std(color),alpha=alpha_in,s=s)
    else:
        color=color[mask]
        ortho.scatter(coords[:,~mask],color='lightgrey',alpha=alpha_out,s=s)
        ortho.scatter(coords[:,mask],c=color,cmap=cmap,
                      vmin=-np.std(color),vmax=np.std(color),alpha=alpha_in,s=s)
    plt.show()
    if interactive:
        return ortho   

    
def plot_coefficients(coords,coef,mask=None,interactive=False,**kwargs):
    """Plot neuron positions projected in three orthogonal planes.
    Different sets of color values, corresponding to different regressors can be given as inputs and selected with a slider.
    Optionally a mask can be given so that neurons outside the mask are plotted in light grey.

    Parameters :
    ------------
    coords : 2d array
        neuron positions, shape (3,n_neurons)
        
    coef : 1d or 2d array
        color values for each neuron, possibly multiple sets corresponding to different regressors, shape (n_neurons) or (n_neurons,n_regressors)

    mask : 1d array
        boolean array to select which neurons will be plotted in color
        default :  None
        
    interactive : boolean
        flag for interactive plotting
        default : False
    """
    if len(coef.shape)==1:
        ortho=plot_brain_projections_color(coords,coef,mask,interactive=interactive,**kwargs)
        if interactive:
            return ortho   
    else:
        @interact(regressor=(0,coef.shape[1]-1))
        def _plot_coefficients(regressor=0):
            plt.close('all')
            ortho=plot_brain_projections_color(coords,coef[:,regressor],mask,interactive=interactive,**kwargs)
            if interactive:
                return ortho    
                
                
def plot_neurons_per_label(neuron_labels,indices,opacity,coords,regressors,dff,times,interactive=False): 
    """For a given regression label inputted as a string: plot position of neurons having that label and their average activity.
    
    Parameters :
    ------------
    neuron_labels : 2d array
        1 for positive coefficient, -1 for negative, 0 for not significant, shape (n_selected_neurons,n_regressors)
        
    indices : 1d
        indices of neurons for which labels are given, shape (n_selected_neurons) 

    opacity : 1d array
        opacity values for all neuron, shape (n_neurons) 
    
    coords : 2d array
        neuron positions for all neuron, shape (3,n_neurons)

    regressors : 2d array
        array of regressors, shape (n_timepoints,n_regressors)
        default :  None

    dff : 2d array
        rescaled fluorescence traces for all neurons, shape (n_neurons,n_timepoints)    

    times : 1d array
        timepoints of regressors and dff, shape (n_timepoints)
        
    interactive : boolean
        flag for interactive plotting
        default : False
    """
    n_regressors=len(regressors)
    all_possible_labels=list(itertools.product([0,1,-1],repeat=n_regressors)) #get all possible combinations of 0, 1 and -1
    n_neurons_per_label=[]
    text_labels=[]
    for label in all_possible_labels:
        n_neurons_per_label.append(np.sum(np.all(neuron_labels==label,axis=1))) #number of neurons with a certain label
        text_labels.append(str(label)) #each label saved as a string
    n_neurons_per_label=np.array(n_neurons_per_label)
    text_labels=np.array(text_labels)

    @interact(label=text_labels[np.argsort(n_neurons_per_label)[-1]])
    def _plot_neurons_per_label(label):
        
        i=np.nonzero(text_labels==label)[0]
        if len(i)==0:
            print('invalid combination')
        else:
            i=int(i)
            label_i=all_possible_labels[i]
            mask_label=np.zeros(len(dff))
            mask_label[indices[np.all(neuron_labels==label_i,axis=1)]]=1
            mask_label=mask_label.astype(bool)
            print('number of neurons: '+str(n_neurons_per_label[i]))
            if n_neurons_per_label[i]>0:
                plt.close('all')
                ortho=plot_brain_projections(coords,opacity,mask_label,interactive=interactive)
                fig,ax=plt.subplots()
                for n,regressor in enumerate(regressors):
                    ax.plot(times,scale(regressor),label='regressor '+str(n))
                ax.plot(times,scale(dff[mask_label].mean(axis=0)),'k',label='average $\Delta F/F$ with '+label)
                ax.set_xlim(times[0],times[-1])
                ax.set_xlabel('time (s)')
                ax.legend()
                plt.show()
                if interactive:
                    return ortho    


class ComplexRaster:
    def __init__(self, ax, X, 
                 signal=None, times=None, order=None, weights=None,
                 X_params={"cmap":"plasma", "vmin":None, "vmax":None},
                 signal_params={"cmap":"plasma", "vmin":None, "vmax":None},
                 weights_params={"cmap":"plasma", "vmin":None, "vmax":None},
                ):
        self.ax = ax
        self.X = X
        self.signal = signal
        self.times = times
        self.order = order
        self.weights = weights

        self._X_params = X_params
        self._signal_params = signal_params
        self._weights_params = weights_params

        assert self.X.ndim==2
        self._n, self._t  = self.X.shape        

        divider = make_axes_locatable(ax)
        if self.signal is not None:
            assert self.signal.ndim==1
            assert len(self.signal)==self._t
            self._ax_sig = divider.append_axes('top',size=0.2,pad=0.1,sharex=ax)
            self._ax_sig.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
            
        if self.weights is not None:
            assert self.weights.ndim==1
            assert len(self.weights)==self._n
            self._ax_wei = divider.append_axes('left',size=0.2,pad=0.1,sharey=ax)
            self._ax_wei.tick_params(axis='both',which='both',bottom=False,left=True,labelbottom=False,labelleft=True)
            ax.tick_params(axis='both',which='both',bottom=True,left=False,labelbottom=True,labelleft=False)
            self._ax_wei.set_ylabel("Neurons & Weight (re-ordered)")
        else:
            ax.set_ylabel("Neurons")
            
        if self.times is not None:
            assert self.times.ndim==1
            assert len(self.times)==self._t
            ax.set_xlabel("Time (s)")
        else:
            self.times = np.arange(self.X.shape[1])
            ax.set_xlabel("Time (frame)")
        dt = np.mean(np.diff(self.times))
        self._extent = [self.times[0]-dt/2,self.times[-1]+dt/2,self.X.shape[0]-0.5,-0.5]

        if (self.weights is not None) and (self.order is None):
            self.order = np.argsort(self.weights)[::-1]
        if self.order is not None:
            assert self.order.ndim==1
            assert len(self.order)==self._n
            self.X = self.X[self.order]
            if self.weights is not None:
                self.weights = self.weights[self.order]

        
        self._im_dff = self.ax.imshow(
            self.X,
            **self._X_params,
            interpolation='none', aspect='auto', extent=self._extent)
        if self.signal is not None:
            self._im_sig = self._ax_sig.imshow(
                self.signal.reshape(1,-1), 
                cmap="plasma", vmin=-10, vmax=+10, 
                aspect='auto', interpolation='none', extent=self._extent
            )
        if self.weights is not None:
            self._im_wei = self._ax_wei.imshow(
                self.weights.reshape(-1,1), 
                cmap="seismic", vmin=-0.01, vmax=+0.01, 
                aspect='auto', interpolation='none', extent=self._extent
            )

    @classmethod
    def update_img(cls, ax_img, img):
        """update the data of an matplotlib.image.AxesImage. """
        ax_img.set_data(img)

    def update_order(self, order):
        self.update_img(self._im_dff, self.X[order])
        if self.weights is not None:
            self.update_img(self._im_wei, self.weights[order].reshape(-1,1))

class PCA_raster(ComplexRaster):
    def __init__(self, ax, X, pca, k=0, times=None):
        self.pca = pca
        self.k = k
        self.x = X.T
        self.Y = self.pca.transform(X)
        super().__init__(
            ax, 
            self.x,
            signal=self.Y[:,self.k], 
            times=times,
            weights=self.pca.components_[self.k],
            X_params={"cmap":"plasma", "vmin":0, "vmax":1},
            signal_params={"cmap":"plasma", "vmin":-10, "vmax":+10},
            weights_params={"cmap":"plasma", "vmin":None, "vmax":None},
        )

    def change_component(self,k):
        print(self.X.shape)
        self.k = k
        self.update_img(self._im_sig, self.Y[:,self.k].reshape(1,-1))
        self.weights = self.pca.components_[self.k]
        order = np.argsort(self.weights)[::-1]
        self.update_img(self._im_wei, self.weights[order].reshape(-1,1))
        self.update_img(self._im_dff, self.x[order])



def plot_angle_colorbar(mappable, ax=None, label=""):
    """Add a circular colorbar to a plot.
    
    :mappable: matplotlib.cm.ScalarMappable
        see `plt.colorbar` or matplotlib.figure.Figure.colobar for more details
    :ax: ~.axes.Axes
        matplotlib parent axis
    :label: str
        label of the colorbar.
    """
    from matplotlib.projections.polar import PolarAxes
    angles = mappable.get_array()
    angles = np.unique(angles[np.isfinite(angles)])
    
    rad = np.linspace(0.9, 1.1, 2)
    rg, tg = np.meshgrid(rad,angles)

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    _ax = divider.append_axes('right',size="20%",pad=0.5, axes_class=PolarAxes)
    _ax.set_title(label)

    vmin, vmax = mappable.get_clim()
    _ax.pcolormesh(np.deg2rad(angles), rad, tg.T, cmap=mappable.get_cmap(), vmin=vmin, vmax=vmax)
    t = np.rad2deg(_ax.get_xticks())
    t[t > 180] -= 360
    _ax.set_xticklabels([f"{int(tt)}°" for tt in t], fontdict={'fontweight':"bold"})
    _ax.axes.get_yaxis().set_visible(False)
    _ax.grid(False)
    _ax.axes.spines['polar'].set_visible(False)