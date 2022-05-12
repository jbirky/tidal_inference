import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import utils as dyfunc

import numpy as np
from functools import partial
import multiprocessing as mp
import os
import copy
import yaml
from yaml.loader import SafeLoader
from collections import OrderedDict
from astropy import units as u

import vplanet_inference as vpi
from alabi import utility as ut


__all__ = ["SyntheticModel"]


class SyntheticModel(object):
    
    def __init__(self, cfile, fix_tide=True, verbose=True):
    
        # load YAML config file
        with open(cfile) as f:
            data = yaml.load(f, Loader=SafeLoader)

        # format fixed input parameters
        self.inparams_fix = OrderedDict()
        self.theta_true = OrderedDict()

        for key in data['input_fix'].keys():
            self.inparams_fix[key] = eval(data['input_fix'][key]['units'])
            self.theta_true[key] = float(data['input_fix'][key]['true_value'])

        # format variable input parameters
        self.inparams_var = OrderedDict()
        self.bounds = []
        self.prior_data = []
        self.labels = []

        for key in data['input_var'].keys():
            self.inparams_var[key] = eval(data['input_var'][key]['units'])
            self.theta_true[key] = float(data['input_var'][key]['true_value'])

            self.labels.append(str(data['input_var'][key]['label']))
            self.bounds.append(eval(data['input_var'][key]['prior_bounds']))
            self.prior_data.append(eval(data['input_var'][key]['prior_data']))

        # format output parameters
        self.outparams = OrderedDict()
        self.output_unc = []

        for key in data['output'].keys():
            self.output_unc.append(float(data['output'][key]['uncertainty']))
            self.outparams[key] = eval(data['output'][key]['units'])

        self.fix_tide = fix_tide

        # load config file
        self.inparams_all = {**self.inparams_fix, **self.inparams_var}

        # initialize vplanet model
        inpath = os.path.join(vpi.INFILE_DIR, data['module'], data['tide_model'])
        self.vpm = vpi.VplanetModel(self.inparams_all, inpath=inpath, outparams=self.outparams, verbose=verbose)

        # run vplanet model on true parameters
        self.output_true = self.vpm.run_model(np.array(list(self.theta_true.values())))
        self.like_data = np.vstack([self.output_true, self.output_unc]).T

        # set up prior for dynesty
        self.ptform = partial(ut.prior_transform_normal, bounds=self.bounds, data=self.prior_data)


    def format_theta(self, theta_var):

        theta_run = copy.copy(self.theta_true)
        theta_var = dict(zip(list(self.inparams_var.keys()), theta_var))

        for key in self.inparams_var.keys():
            theta_run[key] = theta_var[key]

        if self.fix_tide == True:
            theta_run['secondary.dTidalTau'] = theta_run['primary.dTidalTau']

        return np.array(list(theta_run.values()))


    def lnlike(self, theta_var):

        # format fixed theta + variable theta
        theta_run = self.format_theta(theta_var)

        # run model
        output = self.vpm.run_model(theta_run)

        # compute log likelihood
        lnl = -0.5 * np.sum(((output - self.like_data.T[0])/self.like_data.T[1])**2)

        return lnl
