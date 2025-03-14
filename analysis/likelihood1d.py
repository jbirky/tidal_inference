import vplanet_inference as vpi
import numpy as np
import os
import time
from functools import partial
import multiprocessing as mp
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=35)
rc('ytick', labelsize=35)

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


def plot_likelihood_1param(true_tides=None,
                           true_ages=None,
                           tide_model="ctl",
                           respath="results",
                           plotpath="plots",
                           colors=["b", "orange", "g", "r", "m", "c"],
                           fs=35):

    tide_model = tide_model.lower()
    if tide_model == "ctl":
        tide_label = r"$\tau$"
    elif tide_model == "cpl":
        tide_label = r"$Q$"
    else:
        raise Exception(f"{tide_model} is not a valid EQTIDE model." + 
                        f"Choose either tide_model='ctl' or Choose either tide_model='cpl'.")

    if not os.path.exists(plotpath):
        os.makedirs(plotpath)

    for age_true in true_ages:

        plt.figure(figsize=[12,10])
        yarr = []

        results = ["tide_"+str(tv).replace("-", "n") for tv in true_tides]

        for ii, res in enumerate(results):
            theta, y = load_sims(f"{respath}/1param_{tide_model}_{age_true}_myr/{res}/tide_likelihood_sample.npz")
            yarr.append(np.log10(-y))

            pl_label = tide_label + r"$=%s$"%(true_tides[ii])
            plt.plot(theta, np.log10(-y), linewidth=2, color=colors[ii], label=pl_label)
            plt.axvline(x=true_tides[ii], linestyle='--', color='k', linewidth=1)

        plt.xlabel(tide_label, fontsize=fs)
        plt.ylabel(r"$\log(-\log\mathcal{L})$", fontsize=fs)
        plt.legend(loc='upper left', fontsize=25)
        plt.xlim(min(theta), max(theta))
        yf = np.array(yarr).flatten()
        plt.ylim(np.nanmax(yf[np.isfinite(yf)])+1, np.nanmin(yf[np.isfinite(yf)])-1)
        plt.minorticks_on()
        plt.tight_layout()
        plt.savefig(f"{plotpath}/vary_tide_likelihood_{age_true}_myr.png", bbox_inches="tight")
        plt.close()