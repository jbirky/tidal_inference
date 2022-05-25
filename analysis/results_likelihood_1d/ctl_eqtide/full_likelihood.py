import numpy as np
import astropy.units as u
import os
import sys
sys.path.append('../../src')
import run_sims
import vplanet_inference as vpi

os.nice(10)


# =========================================================
# Configure Simulations
# =========================================================

module = "eqtide"
tide_model = "ctl"
inpath = os.path.join(vpi.INFILE_DIR, module, tide_model)

inparams = {"primary.dMass": u.Msun, 
            "secondary.dMass": u.Msun, 
            "primary.dRadius": u.Rsun,
            "secondary.dRadius": u.Rsun,
            "primary.dRotPeriod": u.day, 
            "secondary.dRotPeriod": u.day, 
            "primary.dTidalTau": u.dex(u.s), 
            "secondary.dTidalTau": u.dex(u.s), 
            "primary.dObliquity": u.deg, 
            "secondary.dObliquity": u.deg, 
            "secondary.dEcc": u.dimensionless_unscaled, 
            "secondary.dOrbPeriod": u.day,
            "vpl.dStopTime": u.Myr}

outparams = {"final.primary.RotPer": u.day, 
             "final.secondary.RotPer": u.day,
             "final.secondary.OrbPeriod": u.day,
             "final.secondary.Eccentricity": u.dimensionless_unscaled}

truevals = {"primary.dMass": 1.0, 
            "secondary.dMass": 1.0, 
            "primary.dRadius": 1.0,
            "secondary.dRadius": 1.0,
            "primary.dRotPeriod": 0.5, 
            "secondary.dRotPeriod": 0.5, 
            "primary.dObliquity": 0.0, 
            "secondary.dObliquity": 0.0, 
            "secondary.dEcc": 0.15, 
            "secondary.dOrbPeriod": 7.0}

outputunc = {"final.primary.RotPer": 0.4, 
            "final.secondary.RotPer": 0.4,
            "final.secondary.OrbPeriod": 0.0001,
            "final.secondary.Eccentricity": 0.1}

true_ages = [10, 50, 100, 500, 1000, 5000, 10000]
true_tides = [-3, -2, -1, 0, 1]
bounds = [(-4, 2)]
respath = "results/full_likelihood"
plotpath = "plots/full_likelihood"


# =========================================================
# Run Simulations
# =========================================================

run_sims.run_likelihood_tide(inparams=inparams, 
                             outparams=outparams, 
                             truevals=truevals,
                             outputunc=outputunc,
                             inpath=inpath,
                             tide_model=tide_model,
                             true_tides=true_tides,
                             bounds=bounds,
                             grid_step=0.063,
                             true_ages=true_ages,
                             respath=respath,
                             verbose=False,
                             ncore=30)

run_sims.plot_likelihood_1param(true_tides=true_tides,
                                true_ages=true_ages,
                                tide_model=tide_model,
                                respath=respath,
                                plotpath=plotpath)