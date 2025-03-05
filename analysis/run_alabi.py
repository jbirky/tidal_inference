import os
import numpy as np
from functools import partial
import sys
sys.path.append(os.path.realpath("../src"))
import tidal
import alabi
from alabi.cache_utils import load_model_cache
from alabi import utility as ut


config = 131
ncore = 5

# EXECUTABLE = "/home/jbirky/Dropbox/packages/vplanet-private/bin/vplanet"
EXECUTABLE = "vplanet"
vpm_kwargs = {"executable": EXECUTABLE}

synth = tidal.SyntheticModel(f"config/config_{config}.yaml", 
                             verbose=False, 
                             ncore=ncore, 
                             compute_true=True, 
                             vpm_kwargs=vpm_kwargs)

prior_sampler = partial(alabi.utility.prior_sampler, bounds=synth.inparams_var.bounds, sampler='uniform')

sm = alabi.SurrogateModel(fn=synth.lnlike, 
                          bounds=synth.inparams_var.bounds, 
                          prior_sampler=prior_sampler,
                          savedir=os.path.join(synth.outpath, "results_alabi/", synth.config_id), 
                          labels=synth.inparams_var.labels,
                          verbose=True,
                          cache=True,
                          scale="nlog")

sm.init_samples(ntrain=1500, ntest=100, reload=False)
sm.init_gp(kernel="ExpSquaredKernel", fit_amp=False, fit_mean=True, white_noise=-15)

# sm = load_model_cache(f"results_alabi/config_{config}/")
sm.active_train(niter=1500, algorithm="bape", gp_opt_freq=10, save_progress=True)
sm.plot(plots=["gp_all"])

sm = load_model_cache(f"results_alabi/config_{config}/")

prior_data = np.full((len(synth.inparams_var.bounds), 2), None)

# Prior - emcee format
lnprior = partial(ut.lnprior_uniform, bounds=synth.inparams_var.bounds)

# Prior - dynesty format
prior_transform = partial(ut.prior_transform_uniform, bounds=synth.inparams_var.bounds)

# sm.run_emcee(lnprior=lnprior, nwalkers=50, nsteps=5e4, opt_init=False)
# sm.plot(plots=["emcee_corner"])

# sm.run_dynesty(ptform=prior_transform, mode='dynamic')
# sm.plot(plots=["dynesty_all"])


# =============================================
# Run dynesty

# synth.run_mcmc(method="dynesty")

# =============================================
## Run alabi

# synth.run_mcmc(method="alabi", 
#                reload_sm=False, 
#                reload_samp=True,
#                kernel="ExpSquaredKernel",
#                ntrain=1000, 
#                ntest=1, 
#                niter=1000)

# sims = np.load("results_alabi/config_125/initial_training_sample.npz")

# sm = load_model_cache(f"results_alabi/config_{config}/")
# sm.active_train(niter=1000, algorithm="bape", gp_opt_freq=10, save_progress=True)
# sm.plot(plots=["gp_train_corner"])