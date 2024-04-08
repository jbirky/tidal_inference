import vplanet_inference as vpi
import numpy as np
import os
import time
from functools import partial
import multiprocessing as mp
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=36)
rc('ytick', labelsize=36)

__all__ = ["run_likelihood_tide",
           "plot_likelihood_1param"]


# =====================================================================================
# 1D likelihood tests
# =====================================================================================

def lnlike(tide_test, vpm=None, like_data=None, theta_true=None, tide_ind=None, verbose=False):

    theta_var = np.copy(theta_true)
    for ind in tide_ind:
        theta_var[ind] = tide_test
    sim_result = vpm.run_model(theta_var)
    lnl = -0.5 * np.sum(((sim_result - like_data.T[0])/like_data.T[1])**2)

    if verbose == True:
        print("tide true: \t" + str(theta_true[tide_ind[0]]) + 
              "\ntide test: \t" + str(tide_test[0]) + 
              "\nlnlike: \t" + str(lnl) + "\n") 

    return lnl


def run_likelihood_tide(inparams=None, 
                        outparams=None, 
                        truevals=None,
                        outputunc=None,
                        inpath=None,
                        tide_model="ctl",
                        true_tides=np.round(np.arange(-3, 3, 1),1),
                        bounds=[(-4, 2)],
                        true_ages=[10, 50, 100, 500, 1000, 5000, 10000],
                        grid_step=0.05,
                        respath="results",
                        verbose=False,
                        ncore=mp.cpu_count()):

    if len(outparams) != len(outputunc):
        raise Exception(f"Dimensions of outparams and unc do not match." + 
                        f"len(outparams)={len(outparams)}, len(outputunc)={len(outputunc)}")

    tide_model = tide_model.lower()
    if tide_model == "ctl":
        tide_param_key_name = "dTidalTau"
    elif tide_model == "cpl":
        tide_param_key_name = "dTidalQ"
    else:
        raise Exception(f"{tide_model} is not a valid EQTIDE model." + 
                        f"Choose either tide_model='ctl' or Choose either tide_model='cpl'.")

    vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams, 
                           verbose=False, time_init=5e6*u.yr)

    results = ["tide_"+str(tv).replace("-", "n") for tv in true_tides]

    # grid of tide values to compute likelihood for
    tide_test_grid = np.arange(min(bounds[0]), max(bounds[0]), grid_step).reshape(-1,1)

    for age_true in true_ages:

        for ii in range(len(true_tides)):

            t0 = time.time()
            tide_true = true_tides[ii]
            
            # Parse input parameters
            theta_true = np.zeros(len(inparams))
            tide_ind = []
            for jj, key in enumerate(inparams.keys()):
                if key.split(".")[1] == tide_param_key_name:
                    theta_true[jj] = tide_true
                    tide_ind.append(jj)
                elif key.split(".")[1] == "dStopTime":
                    theta_true[jj] = age_true
                elif key in truevals:
                    theta_true[jj] = truevals[key]
                else:
                    raise Exception(f"key {key} not found in truevals")

            # Parse output parameters
            unc = np.zeros(len(outparams))
            for jj, key in enumerate(outparams.keys()):
                if key in outputunc:
                    unc[jj] = outputunc[key]
                else:
                    raise Exception(f"key {key} not found in outputunc")

            # Run true model 
            output = vpm.run_model(theta_true)
            like_data =  np.array([output, unc]).T

            # Compute likelihood with tide varied
            tide_lnlike = partial(lnlike, vpm=vpm, like_data=like_data, 
                                 theta_true=theta_true, tide_ind=tide_ind,
                                 verbose=verbose)
                
            with mp.Pool(ncore) as p:
                lnlike_results = p.map(tide_lnlike, tide_test_grid)

            subpath = f"{respath}/1param_{tide_model}_{age_true}_myr/{results[ii]}/"
            if not os.path.exists(subpath):
                os.makedirs(subpath)

            np.savez(f"{subpath}/tide_likelihood_sample.npz",
                     theta=tide_test_grid, 
                     y=np.array(lnlike_results))

            print("tide", tide_true, "age", age_true, f"{np.round(time.time() - t0)}s")


# =====================================================================================
# 1D likelihood plotting functions
# =====================================================================================

def load_sims(file):    
    
    sims = np.load(file)
    theta = sims['theta'].T[0]
    y = sims['y']

    resort = np.argsort(theta)
    theta = theta[resort]
    y = y[resort]
    
    return theta, y


def plot_likelihood_1param(true_tides=None,
                           true_ages=None,
                           tide_model="ctl",
                           respath="results",
                           plotpath="plots",
                           title=None,
                           colors=["b", "orange", "g", "r", "m", "c"]):

    tide_model = tide_model.lower()
    if tide_model == "ctl":
        tide_label = r"$\log\tau$"
    elif tide_model == "cpl":
        tide_label = r"$\log\mathcal{Q}$"
    else:
        raise Exception(f"{tide_model} is not a valid EQTIDE model." + 
                        f"Choose either tide_model='ctl' or Choose either tide_model='cpl'.")

    if not os.path.exists(plotpath):
        os.makedirs(plotpath)

    for age_true in true_ages:

        plt.figure(figsize=[14,12])
        ax = plt.gca()

        results = ["tide_"+str(tv).replace("-", "n") for tv in true_tides]

        for ii, res in enumerate(results):
            theta, y = load_sims(f"{respath}/1param_{tide_model}_{age_true}_myr/{res}/tide_likelihood_sample.npz")
            # yarr.append(np.log10(-y))

            pl_label = tide_label + r"$=%s$"%(true_tides[ii])
            plt.plot(theta, np.log10(-y), linewidth=2, color=colors[ii], label=pl_label)
            plt.axvline(x=true_tides[ii], linestyle='--', color='k', linewidth=1, alpha=0.6)

        if title is None:
            plt.title(f"{tide_model.upper()} ({age_true} Myr)", fontsize=55)
        else:
            plt.title(f"{title} ({age_true} Myr)", fontsize=55)
        plt.xlabel(tide_label, fontsize=45)
        plt.ylabel(r"$\log(-\log\mathcal{L})$", fontsize=45)
        plt.legend(loc='best', fontsize=30)
        plt.xlim(min(theta), max(theta))
        ax.invert_yaxis()
        loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)
        plt.minorticks_on()
        plt.savefig(f"{plotpath}/vary_tide_likelihood_{age_true}_myr.png", bbox_inches="tight")
        plt.close()


# =====================================================================================
# Run evolution sims varying tides
# =====================================================================================

def run_evolution_vary_tides():

    vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams, 
                        outpath="results/1param_evolution/",
                        verbose=True, timesteps=1e5*u.yr)

    qvals = np.round(np.arange(4, 9, .2),1)
    sdir = [str(tv) for tv in qvals]

    def run_evol_q(q_ind):
        q = qvals[q_ind]
        tt = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, q, q, 0.0, 0.0, 0.15, 7.0, 10])
        output = vpm.run_model(tt, remove=False, outsubpath=sdir[q_ind])
        
    with mp.Pool(len(qvals)) as p:
        p.map(run_evol_q, range(len(qvals)))


# =====================================================================================
# Plot evolution sims varying tides
# =====================================================================================

def plot_evolution_vary_tides():

    savedir = "results/1param_evolution"

    qvals = np.round(np.arange(4, 9, .2),1)
    results = [str(tv) for tv in qvals]

    c = np.arange(0, len(qvals))

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, axs = plt.subplots(1, 3, figsize=[30,16], sharex=True)

    for ii in range(len(qvals)):
        keys = ['time', 'prot', 'porb', 'ecc', 'none']
        pri = pd.read_csv(f'{savedir}/{results[ii]}/system.primary.forward', sep=' ', names=keys)[1:]
        sec = pd.read_csv(f'{savedir}/{results[ii]}/system.secondary.forward', sep=' ', names=keys)[1:]
        pri['time'] = (np.array(pri['time']) * u.sec).to(u.yr).value
        sec['time'] = (np.array(sec['time']) * u.sec).to(u.yr).value
        
        axs[0].plot(sec['time'], sec['porb'], c=cmap.to_rgba(ii))
        axs[1].plot(pri['time'], pri['prot'], c=cmap.to_rgba(ii))
        axs[1].plot(sec['time'], sec['prot'], c=cmap.to_rgba(ii), linestyle="--")
        axs[2].plot(sec['time'], sec['ecc'], c=cmap.to_rgba(ii))
            
        axs[0].set_ylabel("Orbital Period [d]", fontsize=20)
        axs[1].set_ylabel("Rotation Period [d]", fontsize=20)
    #     axs[0].legend(loc='best', fontsize=16)
        axs[2].set_ylabel("Eccentricity", fontsize=20)

    fig.colorbar(cmap, ticks=c).ax.set_yticklabels(qvals)
    plt.xscale('log')
    plt.xlim(min(sec['time']), max(sec['time']))
    axs[0].minorticks_on()
    axs[1].minorticks_on()
    plt.tight_layout()
    plt.savefig("plots/vary_q_evolution.png")
    plt.show()