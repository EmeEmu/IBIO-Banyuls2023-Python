import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import interact
from sklearn.preprocessing import scale
from Helper_Functions.OrthoViewer import OrthoAxes
import itertools


def plot_dff_traces(signal,times,indices,dff,dpi=100):
    @interact(neuron=(0,len(indices)-1,1))
    def _plot_dff_traces(neuron=0):
        plt.close()
        fig,ax=plt.subplots(dpi=dpi)
        ax.plot(times,scale(signal),label='external signal')
        ax.plot(times,scale(dff[indices[neuron]]),'k',label='neuron $\Delta F/F$')
        ax.set_xlim(times[0],times[-1])
        ax.set_xlabel('time (s)')
        ax.legend()
        plt.show()
        
        
def plot_dff_raster(signal,signal_times,indices,dff,dpi=100):
    fig,ax=plt.subplots(dpi=dpi)
    divider=make_axes_locatable(ax)
    tax=divider.append_axes('top',size='10%',pad=0.1,sharex=ax)
    cax=divider.append_axes('right',size='2%',pad=0.1)

    dt=np.mean(np.diff(signal_times))
    extent=[signal_times[0]-dt/2,signal_times[-1]+dt/2,len(indices)-0.5,-0.5]
    cmap='inferno'

    tax.imshow(signal.reshape(1,-1),cmap=cmap,interpolation='none',aspect='auto',extent=extent)
    tax.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
    tax.set_title('signal')

    im=ax.imshow(dff[indices][:,:],cmap=cmap,interpolation='none',aspect='auto',extent=extent)
    cbar=plt.colorbar(im,cax=cax)
    cbar.set_label(r'$\Delta F/F$')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('neurons')
    plt.show()
    
    
def plot_brain_layers(coords,mask,statistic,alpha_out=0.5,s=1,dpi=100):
    zs=np.unique(coords[2])
    zs=-np.sort(-zs)
    alpha=statistic/np.max(statistic)
    alpha[alpha<0]=0
    @interact(layer=(0,len(zs)-1,1))
    def _plot_brain_layer(layer=0):
        plt.close()
        fig,ax=plt.subplots(dpi=dpi)

        x_layer=coords[0,(coords[2]==zs[layer])&(~mask)]
        y_layer=coords[1,(coords[2]==zs[layer])&(~mask)]
        if len(x_layer)>0:
            ax.scatter(x_layer,y_layer,s=s,c='lightgrey',alpha=alpha_out)

        x_layer=coords[0,(coords[2]==zs[layer])&(mask)]
        y_layer=coords[1,(coords[2]==zs[layer])&(mask)]
        alpha_layer=alpha[(coords[2]==zs[layer])&(mask)]
        if len(x_layer)>0:
            ax.scatter(x_layer,y_layer,s=s,c='blue',alpha=alpha_layer)

        ax.set_aspect('equal')
        ax.set_xlim(np.min(coords[0]),np.max(coords[0]))
        ax.set_ylim(np.min(coords[1]),np.max(coords[1]))
        ax.set_title('$z = $'+str(round(zs[layer]))+' μm')
        ax.set_xlabel('$x$ (μm)')
        ax.set_ylabel('$y$ (μm)')
        plt.show()
        
        
def plot_brain_projections(coords,statistic,mask=None,alpha_out=0.1,c='blue',s=2,interactive=False,dpi=100):
    alpha=statistic/np.max(statistic)
    alpha[alpha<0]=0
    fig=plt.figure(figsize=(6,6),dpi=dpi)
    ortho=OrthoAxes(fig,coords,interactive=interactive)
    if mask is None:
        ortho.scatter(coords,c=c,alpha=alpha,s=s)
    else:
        alpha=alpha[mask]
        ortho.scatter(coords[:,~mask],c='lightgrey',alpha=alpha_out,s=s)
        ortho.scatter(coords[:,mask],c=c,alpha=alpha,s=s)
    plt.show()
    
    
def plot_brain_projections_color(coords,mask,color,cmap='Blues',s=2,interactive=False,**kwargs):
    fig=plt.figure(figsize=(6,6),dpi=dpi)
    ortho=OrthoAxes(fig,coords,interactive=interactive)
    ortho.scatter(coords[:,~mask],c='lightgrey',alpha=0.1,s=s)
    ortho.scatter(coords[:,mask],c=color[mask],cmap=cmap,alpha=0.2,s=s,**kwargs)
    plt.show()
    
    
def plot_brain_projections_colorbar(coords,mask,statistic,alpha_in=0.3,alpha_out=0.1,s=3,cmap='Blues',vmin=0,vmax=10,dpi=100):
    fig,ax=plt.subplots(figsize=(6,6),dpi=dpi)
    divider=make_axes_locatable(ax)
    ax2=divider.append_axes('right',size='40%',pad=0,sharey=ax)
    ax3=divider.append_axes('top',size='20%',pad=0,sharex=ax)
    cax=divider.append_axes('right',size='5%',pad=0)

    ax.scatter(coords[0,~mask],coords[1,~mask],s=s,c='lightgrey',alpha=alpha_out,edgecolors='none')
    im=ax.scatter(coords[0,mask],coords[1,mask],s=s,c=statistic[mask],vmin=vmin,vmax=vmax,cmap=cmap,alpha=alpha_in,edgecolors='none')

    ax.set_aspect('equal')
    ax.set_xlabel('$x$ (μm)')
    ax.set_ylabel('$y$ (μm)')
    
    ax2.scatter(coords[2,~mask],coords[1,~mask],s=s,c='lightgrey',alpha=alpha_out,edgecolors='none')
    ax2.scatter(coords[2,mask],coords[1,mask],s=s,c=statistic[mask],vmin=vmin,vmax=vmax,cmap=cmap,alpha=alpha_in,edgecolors='none')

    ax2.tick_params(left=False,labelleft=False)
    ax2.set_aspect('equal')
    ax2.set_xlabel('$z$ (μm)')
    
    ax3.scatter(coords[0,~mask],coords[2,~mask],s=s,c='lightgrey',alpha=alpha_out,edgecolors='none')
    ax3.scatter(coords[0,mask],coords[2,mask],s=s,c=statistic[mask],vmin=vmin,vmax=vmax,cmap=cmap,alpha=alpha_in,edgecolors='none')

    ax3.tick_params(bottom=False,labelbottom=False)
    ax3.set_aspect('equal')
    ax3.set_ylabel('$z$ (μm)')

    cbar=plt.colorbar(im,cax=cax)
    #cbar.set_label(r'statistic')
    plt.show()
    
    
def plot_coefficients(coords,coef,mask=None,alpha_in=0.5,alpha_out=0.1,s=2,dpi=100):
    if len(coef.shape)==1:
        fig=plt.figure(figsize=(6,6),dpi=dpi)
        ortho=OrthoAxes(fig,coords,interactive=0)
        if mask is None:
            color=coef
            ortho.scatter(coords,c=color,cmap='coolwarm',
                          vmin=-np.std(color),vmax=np.std(color),alpha=alpha_in,s=s)
        else:
            color=coef[mask]
            ortho.scatter(coords[:,~mask],color='lightgrey',alpha=alpha_out,s=s)
            ortho.scatter(coords[:,mask],c=color,cmap='coolwarm',
                          vmin=-np.std(color),vmax=np.std(color),alpha=alpha_in,s=s)
        plt.show()
    else:
        @interact(regressor=(0,coef.shape[1]-1))
        def _plot_coefficients(regressor=0):
            plt.close()
            fig=plt.figure(figsize=(6,6),dpi=dpi)
            ortho=OrthoAxes(fig,coords,interactive=0)
                
            if mask is None:
                color=coef[:,regressor]
                ortho.scatter(coords,c=color,cmap='coolwarm',
                              vmin=-np.std(color),vmax=np.std(color),alpha=alpha_in,s=s)
            else:
                color=coef[mask,regressor]
                ortho.scatter(coords[:,~mask],color='lightgrey',alpha=alpha_out,s=s)
                ortho.scatter(coords[:,mask],c=color,cmap='coolwarm',
                              vmin=-np.std(color),vmax=np.std(color),alpha=alpha_in,s=s)
            plt.show()
        

def plot_neurons_per_label(coef_sign,indices_significant,statistic,regressors,coords,dff,times,dpi=100):
    n_regressors=coef_sign.shape[1]
    sign_combinations=list(itertools.product([0,1,-1],repeat=n_regressors))
    sizes=[]
    labels=[]
    for sign_combination in sign_combinations:
        sizes.append(np.sum(np.all(coef_sign==sign_combination,axis=1)))
        labels.append(str(sign_combination))
    sizes=np.array(sizes)
    labels=np.array(labels)
    
    @interact(combination=labels[np.argsort(sizes)[-1]])
    def _plot_neurons_per_label(combination):
        i=np.nonzero(np.array(labels)==combination)[0]
        if len(i)==0:
            print('invalid combination')
        else:
            i=int(i)
            sign_combination=sign_combinations[i]
            mask_coef=np.zeros(len(dff))
            mask_coef[indices_significant[np.all(coef_sign==sign_combination,axis=1)]]=1
            mask_coef=mask_coef.astype(bool)
            print('number of neurons: '+str(sizes[i]))
            if sizes[i]>0:
                plt.close('all')
                plot_brain_projections(coords,statistic,mask_coef)
                fig,ax=plt.subplots(dpi=dpi)
                for n,regressor in enumerate(regressors):
                    ax.plot(times,scale(regressor),label='regressor '+str(n))
                ax.plot(times,scale(dff[mask_coef].mean(axis=0)),'k',label='average $\Delta F/F$ with '+combination)
                ax.set_xlim(times[0],times[-1])
                ax.set_xlabel('time (s)')
                ax.legend()
                plt.show()
                
