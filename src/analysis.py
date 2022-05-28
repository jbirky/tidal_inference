import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from functools import partial
import multiprocessing as mp
import os
import copy
import yaml
from yaml.loader import SafeLoader
from astropy import units as u

import vplanet_inference as vpi


__all__ = ["SyntheticModel"]


class SyntheticModel(vpi.AnalyzeVplanetModel):

    def __init__(self, 
                 cfile, 
                 outpath=".", 
                 verbose=True, 
                 compute_true=True,
                 ncore=mp.cpu_count(),
                 fix_tide=True, 
                 fix_porb=False):

        # load YAML config file
        with open(cfile) as f:
            data = yaml.load(f, Loader=SafeLoader)

        self.tide_model = data['tide_model']
        self.module = data['module']
        inpath = os.path.join(vpi.INFILE_DIR, data['module'], data['tide_model'])

        # parent class inheritance
        super().__init__(cfile, 
                         inpath=inpath,
                         outpath=outpath, 
                         verbose=verbose, 
                         compute_true=compute_true,
                         ncore=ncore)

        # set primary and secondary to same value?
        self.fix_tide = fix_tide

        # compute init porb from final porb, ecc, tau, ...
        self.fix_porb = fix_porb


    def format_theta(self, theta_var):

        theta_run = copy.copy(self.theta_true)
        theta_var = dict(zip(list(self.inparams_var.keys()), theta_var))

        for key in self.inparams_var.keys():
            theta_run[key] = theta_var[key]

        if self.fix_tide == True:
            theta_run['secondary.dTidalTau'] = theta_run['primary.dTidalTau']

        # if self.fix_porb == True:
            # back_model = vpi.VplanetModel(self.inparams_all, 
            #                               inpath=self.inpath, 
            #                               outparams={"final.primary.OrbPeriod": u.day},
            #                               forward=False)
            # porb_init = back_model.run()

        return np.array(list(theta_run.values()))