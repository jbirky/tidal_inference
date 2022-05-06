import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc

import numpy as np
from functools import partial
import multiprocessing as mp
import os
import yaml
from yaml.loader import SafeLoader
from collections import OrderedDict
from astropy import units as u

import vplanet_inference as vpi
from alabi import utility as ut


__all__ = ["load_config",
           "lnlike_base",
           "configure_model",
           "run_dynesty_save"]


def load_config(cfile):
    
    with open(cfile) as f:
        data = yaml.load(f, Loader=SafeLoader)
        
    bounds = []
    prior_data = []
    theta_true = []
    labels = []
    inparams = OrderedDict()
    for key in data['input'].keys():
        bounds.append(eval(data['input'][key]['prior_bounds']))
        prior_data.append(eval(data['input'][key]['prior_data']))
        theta_true.append(float(data['input'][key]['true_value']))
        labels.append(str(data['input'][key]['label']))
        inparams[key] = eval(data['input'][key]['units'])
        
    output_unc = []
    outparams = OrderedDict()
    for key in data['output'].keys():
        output_unc.append(float(data['output'][key]['uncertainty']))
        outparams[key] = eval(data['output'][key]['units'])
        
    return data, inparams, bounds, prior_data, theta_true, outparams, output_unc, labels


def lnlike_base(theta, vpm=None, like_data=None, inparams=None, fix_tide=True):
    
    if fix_tide == True:
        pri_tide_ind = np.where(np.array(list(inparams.keys())) == 'primary.dTidalTau')[0]
        sec_tide_ind = np.where(np.array(list(inparams.keys())) == 'secondary.dTidalTau')[0]
        theta[sec_tide_ind] = theta[pri_tide_ind]
        
    output = vpm.run_model(theta)
    
    lnlike = -0.5 * np.sum(((output - like_data.T[0])/like_data.T[1])**2)

    return lnlike
    
        
def configure_model(cfile, fix_tide=True):
    
    data, inparams, bounds, prior_data, theta_true, outparams, output_unc, labels = load_config(cfile)
    
    inpath = os.path.join(vpi.INFILE_DIR, data['module'], data['tide_model'])
    vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams)
    
    output_true = vpm.run_model(theta_true)
    like_data = np.vstack([output_true, output_unc]).T
    
    lnlike = partial(lnlike_base, vpm=vpm, like_data=like_data, inparams=inparams, fix_tide=fix_tide)
    
    ptform = partial(ut.prior_transform_normal, bounds=bounds, data=prior_data)
    
    return lnlike, ptform, labels, bounds


def run_dynesty_save(lnlike, ptform, ndim, save_iter=100, savedir="results"):

    ncore = mp.cpu_count()
    pool = mp.Pool(ncore)
    pool.size = ncore

    # Initialize nested sampler
    dsampler = NestedSampler(lnlike, ptform, ndim, pool=pool)

    
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    run_sampler = True
    last_iter = 0
    while run_sampler == True:
        dsampler.run_nested(maxiter=save_iter)

        file = os.path.join(savedir, "dynesty_sampler.pkl")

        # pickle dynesty sampler object
        with open(file, "wb") as f:        
            pickle.dump(dsampler, f)

        res = dsampler.results
        samples = res.samples  
        weights = np.exp(res.logwt - res.logz[-1])

        # Resample weighted samples.
        dynesty_samples = dyfunc.resample_equal(samples, weights)

        np.savez(savedir+"/dynesty_samples.npz", samples=dynesty_samples)

        # check if converged (i.e. hasn't run for more iterations)
        if dsampler.results.niter > last_iter:
            last_iter = dsampler.results.niter
            run_sampler = True
        else:
            run_sampler = False

    res = dsampler.results
    samples = res.samples  
    weights = np.exp(res.logwt - res.logz[-1])

    # Resample weighted samples.
    dynesty_samples = dyfunc.resample_equal(samples, weights)

    np.savez(savedir+"/dynesty_samples_final.npz", samples=dynesty_samples)