import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import vplanet_inference as vpi
import numpy as np
import scipy
import os
from functools import partial
import multiprocessing as mp
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

inpath = os.path.join(vpi.INFILE_DIR, "stellar_eqtide/ctl")

inparams = {"primary.dMass": u.Msun, 
            "secondary.dMass": u.Msun, 
            "primary.dRotPeriod": u.day, 
            "secondary.dRotPeriod": u.day, 
            "primary.dTidalTau": u.dex(u.s), 
            "secondary.dTidalTau": u.dex(u.s), 
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

tvals = np.round(np.arange(-4, 2, .1),1)
sdir = [str(tv) for tv in tvals]

def run_evol_tau(tau_ind):
    tau = tvals[tau_ind]
    tt = np.array([1.0, 1.0, 0.5, 0.5, tau, tau, 0.0, 0.0, 0.15, 7.0, 10])
    output = vpm.run_model(tt, remove=False, outsubpath=sdir[tau_ind])
    
with mp.Pool(len(tvals)) as p:
    p.map(run_evol_tau, range(len(tvals)))


# ========================================================
# Plot evolution
# ========================================================

savedir = "results/1param_stellar_evolution"

tvals = np.round(np.arange(-4, 2, .2),1)
results = [str(tv) for tv in tvals]

c = np.arange(0, len(tvals))

norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
cmap.set_array([])

fig, axs = plt.subplots(1, 3, figsize=[30,16], sharex=True)

for ii in range(len(tvals)):
    keys = ['time', 'rad', 'lum', 'prot', 'porb', 'ecc', 'none']
    pri = pd.read_csv(f'{savedir}/{results[ii]}/system.primary.forward', sep=' ', names=keys)[1:]
    sec = pd.read_csv(f'{savedir}/{results[ii]}/system.secondary.forward', sep=' ', names=keys)[1:]
    pri['time'] = (np.array(pri['time']) * u.sec).to(u.yr).value
    sec['time'] = (np.array(sec['time']) * u.sec).to(u.yr).value
    
    axs[2].plot(sec['time'], sec['ecc'], c=cmap.to_rgba(ii))
        
    axs[0].set_ylabel("Orbital Period [d]", fontsize=20)
    axs[1].set_ylabel("Rotation Period [d]", fontsize=20)
#     axs[0].legend(loc='best', fontsize=16)
    axs[2].set_ylabel("Eccentricity", fontsize=20)

fig.colorbar(cmap, ticks=c).ax.set_yticklabels(tvals)
plt.xscale('log')
plt.xlim(min(sec['time']), max(sec['time']))
axs[0].minorticks_on()
axs[1].minorticks_on()
plt.tight_layout()
plt.savefig("plots/vary_tau_evolution.png")
plt.show()
