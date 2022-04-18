import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import vplanet_inference as vpi
import numpy as np
import scipy
import os
from functools import partial
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
import pdb

os.nice(10)

# ========================================================
# Configure vplanet forward model
# ========================================================

inpath = os.path.join(vpi.INFILE_DIR, "stellar_eqtide/cpl")

inparams = {"primary.dMass": u.Msun, 
            "secondary.dMass": u.Msun, 
            "primary.dRotPeriod": u.day, 
            "secondary.dRotPeriod": u.day, 
            "primary.dTidalQ": u.dex(u.dimensionless_unscaled), 
            "secondary.dTidalQ": u.dex(u.dimensionless_unscaled), 
            "primary.dObliquity": u.deg, 
            "secondary.dObliquity": u.deg, 
            "secondary.dEcc": u.dimensionless_unscaled, 
            "secondary.dOrbPeriod": u.day,
            "vpl.dStopTime": u.Myr}

outparams = {"final.primary.Radius": u.Rsun,
             "final.secondary.Radius": u.Rsun,
             "final.primary.Luminosity": u.Lsun,
             "final.secondary.Luminosity": u.Lsun,
             "final.primary.RotPer": u.day, 
             "final.secondary.RotPer": u.day,
             "final.secondary.OrbPeriod": u.day,
             "final.secondary.Eccentricity": u.dimensionless_unscaled}

vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams, verbose=False, time_init=5e6*u.yr)

qvals = np.round(np.arange(4, 9, 1),1)
results = ["q_"+str(tv).replace("-", "n") for tv in qvals]

# age_true = 50

for age_true in [10, 50, 100, 500, 1000, 5000, 10000]:

    for ii in range(len(qvals)):
        
        q_true = qvals[ii]
        tt = np.array([1.0, 1.0, 0.5, 0.5, q_true, q_true, 0.0, 0.0, 0.15, 7.0, age_true])
        output = vpm.run_model(tt)

        # ========================================================
        # Observational constraints
        # ========================================================

        unc = np.array([0.01, 0.01, 0.09, 0.09, 0.4, 0.4, 0.00001, 0.002])
        like_data =  np.array([output, unc]).T

        # Prior bounds
        bounds = [(4, 9)]

        # ========================================================
        # Configure prior 
        # ========================================================

        # Prior sampler - alabi format
        ps = partial(ut.prior_sampler, bounds=bounds, sampler="grid")

        # ========================================================
        # Configure likelihood
        # ========================================================

        def lnlike(theta_var):

            theta = tt 
            theta[4] = theta_var[0]
            theta[5] = theta_var[0]

            sim_result = vpm.run_model(theta)
            
            lnl = -0.5 * np.sum(((sim_result - like_data.T[0])/like_data.T[1])**2)
            
            print('lnlike', lnl)

            return lnl


        # ========================================================
        # Run alabi
        # ========================================================

        # breakpoint()

        sm = alabi.SurrogateModel(fn=lnlike, bounds=bounds, prior_sampler=ps, 
                                savedir=f"results/1param_cpl_{age_true}_myr/{results[ii]}")
        sm.init_samples(ntrain=200, ntest=0, reload=False)


    # ========================================================
    # Make plot
    # ========================================================

    def load_sims(file):    
        
        sims = np.load(file)
        theta = sims['theta'].T[0]
        y = sims['y']

        resort = np.argsort(theta)
        theta = theta[resort]
        y = y[resort]
        
        return theta, y


    plt.figure(figsize=[12,10])
    colors = ["b", "orange", "g", "r", "m", "c"]

    qvals = np.round(np.arange(4, 9, 1),1)
    results = ["q_"+str(tv).replace("-", "n") for tv in qvals]

    yarr = []

    for ii, res in enumerate(results):
        theta, y = load_sims(f"results/1param_cpl_{age_true}_myr/{res}/initial_training_sample.npz")
        yarr.append(np.log10(-y))

        plt.plot(theta, np.log10(-y), linewidth=1.2, color=colors[ii], label=r"$Q=%s$"%(qvals[ii]))
        plt.axvline(x=qvals[ii], linestyle='--', color='k', linewidth=1)

    plt.xlabel(r"$Q$", fontsize=22)
    plt.ylabel(r"$\log(-\log\mathcal{L})$", fontsize=22)
    plt.legend(loc='best', fontsize=18)
    plt.xlim(4, 9)
    plt.ylim(10, -3)
    plt.minorticks_on()
    plt.savefig(f"plots/vary_q_likelihood_{age_true}_myr.png")
    plt.show()