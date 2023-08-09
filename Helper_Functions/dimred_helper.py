import numpy as np


def tuning_curve_discrete(angle, signal):
    """Compute the tuning curve and tuning factor of a signal 
    given a discrete experimental parameter.

    Parameters :
    ------------
    :angle: 1D array
        discrete experimental parameter in time.
    :signal: 1D array
        signal in time.

    Return :
    --------
    :tuning_factor: float
        the tuning_factor.
    :theta: 1D array
        the unique parameters from :angle:.
    :s_theta: 1D array
        the mean of the signal for each value of :theta:.
    :ds_theta: 1D array
        the std of the signal for each value of :theta:.
    """
    assert angle.shape==signal.shape, ":angle: and :signal: should be 1D vectors and have the same number of elements."
    theta = np.unique(angle)
    s_theta = np.empty_like(theta)
    ds_theta = np.empty_like(theta)
    for i in range(len(theta)):
        signal_theta = signal[np.where(angle==theta[i])[0]]
        if len(signal_theta) > 0:
            s_theta[i] = np.nanmean(signal_theta)
            ds_theta[i] = np.nanstd(signal_theta)
        else:
            s_theta[i] = np.nan
            ds_theta[i] = np.nan
    tuning_factor = np.nanmean(ds_theta) / np.nanstd(s_theta)
    return tuning_factor, theta, s_theta, ds_theta

def plot_tuning_curve(ax, angle, signal):
    tf, theta, s_theta, ds_theta = tuning_curve_discrete(angle, signal)
    ax.scatter(angle, signal, alpha=0.1, color="k", label="time points")
    ax.plot(theta, s_theta, color="orange", label="mean")
    ax.fill_between(theta, s_theta-ds_theta, s_theta+ds_theta, color="orange", alpha=0.5, label="std")
    ax.text(0.01, 0.95, f"Tuning factor : {tf:0.2f}", transform=ax.transAxes)

def fake_bump(ts, t0, tau=10):
    """Create ΔF/F of a spike train.
    
    Parameters :
    ------------
    :ts: 1D array
        times at which we "observe" the neurons.
    :t0: 1D array or int
        spike train (aka. frames at which the neuron spikes)
    :tau: float
        characteristic time of the calcium decay.

    Return :
    :y: 1D array
        ΔF/F associated with the spike train :t0:.
    """
    assert ts.ndim == 1, ":ts: should be a 1D array."
    y = np.zeros(len(ts))
    i0 = np.argmin(np.abs(ts[:,np.newaxis] - t0), axis=0)
    y[i0] = 1
    if tau > 0:
        dt = np.mean(np.diff(brain_time))
        kernel_size=6
        n_points=int(kernel_size*tau/dt)
        kernel_times=np.linspace(-n_points*dt,n_points*dt,2*n_points+1)
        kernel=np.exp(-kernel_times/tau)
        kernel[kernel_times<0]=0
        y = np.convolve(y,kernel,mode='same')*dt
    return y

def fake_chirp(ts, N=30, tau=10):
    """Create ΔF/F matrix of sequential neuron activation.
    
    Parameters :
    ------------
    :ts: 1D array
        times at which we "observe" the neurons.
    :N: int
        number of neurons
    :tau: float
        characteristic time of the calcium decay.

    Return :
    :Y: 2D array, (N, len(:ts:))
        ΔF/F associated with the spike trains.
    """
    Y = np.empty((N, len(ts)))
    for i in range(N):
        Y[i] = fake_bump(ts, i*len(brain_time)/N/2, tau=tau)
    return Y