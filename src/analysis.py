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
from astropy import constants as const

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
                 fix_porb=False,
                 fix_radius=False):

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

        if data['module'] == "eqtide":
            self.fix_radius = True
        else:
            self.fix_radius = fix_radius

        # print(f"fix_tide: {self.fix_tide}, fix_porb: {self.fix_porb}, fix_radius: {self.fix_radius}")


    # def compute_porb(self, theta_run):

    #     # prot1_f = self.outparams.get_true_units("final.primary.RotPer")
    #     # prot2_f = self.outparams.get_true_units("final.secondary.RotPer")
    #     # porb_f = self.outparams.get_true_units("final.secondary.OrbPeriod")
    #     # ecc_f = self.outparams.get_true_units("final.secondary.Eccentricity")

    #     m1 = theta_run["primary.dMass"] * self.inparams.units["primary.dMass"]
    #     m2 = theta_run["secondary.dMass"] * self.inparams.units["secondary.dMass"]
    #     prot1_i = theta_run["primary.dRotPeriod"] * self.inparams.units["primary.dRotPeriod"]
    #     prot2_i = theta_run["secondary.dRotPeriod"] * self.inparams.units["secondary.dRotPeriod"]
    #     ecc_i = theta_run["secondary.dEcc"] * self.inparams.units["secondary.dEcc"]
    #     r1_i = 1.0 * u.Rsun
    #     r2_i = 1.0 * u.Rsun
    #     r1_i = 0.5 
    #     r2_i = 0.5

    #     Jrot1_f = self.outparams.get_true_units("final.primary.RotAngMom")
    #     Jrot2_f = self.outparams.get_true_units("final.secondary.RotAngMom")
    #     Jorb_f = self.outparams.get_true_units("final.secondary.OrbAngMom")
    #     Jtot_f = Jrot1_f + Jrot2_f + Jorb_f

    #     Jrot1_i = m1 * (rg1_i * r1_i)**2 * (2*np.pi / prot1_i)
    #     Jrot2_i = m2 * (rg2_i * r2_i)**2 * (2*np.pi / prot2_i)
    #     Jorb_i = Jtot_f - Jrot1_i - Jrot2_i

    #     return 0


    def format_theta(self, theta_var):

        theta_run = copy.copy(self.inparams_all.dict_true)
        theta_var = dict(zip(self.inparams_var.names, theta_var))

        for key in self.inparams_var.names:
            theta_run[key] = theta_var[key]

        if self.fix_tide == True:
            theta_run['secondary.dTidalTau'] = theta_run['primary.dTidalTau']

        if self.fix_radius == True:
            theta_run['primary.dRadius'] = theta_run['primary.dMass']
            theta_run['secondary.dRadius'] = theta_run['secondary.dMass']

        # theta_run = self.compute_porb(theta_run)

        # if self.fix_porb == True:
            # back_model = vpi.VplanetModel(self.inparams_all, 
            #                               inpath=self.inpath, 
            #                               outparams={"final.primary.OrbPeriod": u.day},
            #                               forward=False)
            # porb_init = back_model.run()

        return np.array(list(theta_run.values()))