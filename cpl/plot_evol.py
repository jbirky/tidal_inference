import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import vplanet_inference as vpi
import numpy as np
import pandas as pd
import scipy
import os
from functools import partial
import multiprocessing as mp
import astropy.units as u
import matplotlib as mpl
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
            "vpl.dStopTime": u.Gyr}

outparams = {"final.primary.RotPer": u.day, 
             "final.secondary.RotPer": u.day,
             "final.secondary.OrbPeriod": u.day,
             "final.secondary.Eccentricity": u.dimensionless_unscaled}

vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams, 
                       outpath="results/1param_stellar_evolution/",
                       verbose=True, timesteps=1e5*u.yr)

qvals = np.round(np.arange(4, 9, .2),1)
sdir = [str(tv) for tv in qvals]

def run_evol_q(q_ind):
    q = qvals[q_ind]
    tt = np.array([1.0, 1.0, 0.5, 0.5, q, q, 0.0, 0.0, 0.15, 7.0, 10])
    output = vpm.run_model(tt, remove=False, outsubpath=sdir[q_ind])
    
with mp.Pool(len(qvals)) as p:
    p.map(run_evol_q, range(len(qvals)))


# ========================================================
# Plot evolution
# ========================================================

savedir = "results/1param_stellar_evolution"

qvals = np.round(np.arange(4, 9, .2),1)
results = [str(tv) for tv in qvals]

c = np.arange(0, len(qvals))

norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])

fig, axs = plt.subplots(1, 3, figsize=[30,16], sharex=True)

for ii in range(len(qvals)):
    keys = ['time', 'rad', 'lum', 'prot', 'porb', 'ecc', 'none']
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
