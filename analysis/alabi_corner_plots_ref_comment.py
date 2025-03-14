from alabi.cache_utils import load_model_cache
import os
import sys
sys.path.append(os.path.realpath("../src"))
import tidal
import warnings
import corner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
font = {'family' : 'normal',
        'weight' : 'light'}
rc('font', **font)

import astropy.constants as c
import astropy.units as u


def plot_posterior(config_id, title=None, scatter_cmap="Blues_r", true_color="darkorange", cb_rng=None, compute_true=False):
    
    file = f"../analysis/config/config_{config_id}.yaml"

    synth = tidal.SyntheticModel(file, verbose=False, compute_true=compute_true)
    sm = load_model_cache(f"../analysis/results_alabi/config_{config_id}/")

    ndim = sm.ndim
    yy = -sm.y 
    print(len(yy), "samples")

    warnings.simplefilter("ignore")
    
    fig = corner.corner(sm.theta, c=yy, labels=sm.labels, 
            plot_datapoints=False, plot_density=False, plot_contours=False,
            label_kwargs={"fontsize": 22}, data_kwargs={'alpha':1.0},
            range=sm.bounds, dpi=800)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    fig.subplots_adjust(top=1.1, right=1.05, left=.1)

    axes = np.array(fig.axes).reshape((ndim, ndim))
    if cb_rng is None:
        cb_rng = [np.log10(yy.min()), np.log10(yy.max())]

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            im = ax.scatter(sm.theta.T[xi], sm.theta.T[yi], c=yy, s=2, cmap=scatter_cmap, 
                            norm=colors.LogNorm(vmin=10**min(cb_rng), vmax=10**max(cb_rng)),
                            alpha=1.0)
    
    truths = synth.inparams_var.true
    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        fig.delaxes(ax)

    # plot truth values
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(truths[xi], color=true_color, linestyle="--")
            ax.axhline(truths[yi], color=true_color, linestyle="--")
            ax.plot(truths[xi], truths[yi], true_color, linestyle="--")

    cb = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', anchor=(0,.6), 
                        shrink=.55, pad=-0.3)
    cb.set_label(r'$\log(-\log\mathcal{P})$', fontsize=25, labelpad=-100)
    cb_ticks = np.arange(np.ceil(min(cb_rng)), np.ceil(max(cb_rng)))
    cb.set_ticks(10**cb_ticks)
    cb.set_ticklabels(cb_ticks)
    cb.ax.tick_params(labelsize=18)
    if title is not None:
        plt.suptitle(title, fontsize=30)
    plt.show()
    plt.close()
    
    return fig


if __name__ == "__main__":

    config_id = 126
    file = f"config/config_{config_id}.yaml"
    synth = tidal.SyntheticModel(file, verbose=False, compute_true=True)
    fig = plot_posterior(config_id, title=f"CTL simulated posterior (50 Myr)")