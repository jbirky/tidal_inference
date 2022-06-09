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
                 ncore=mp.cpu_count()):

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
        if "secondary.dTidalTau" in self.inparams_fix.names:
            self.fix_tide = True 
        elif "secondary.dTidalQ" in self.inparams_fix.names:
            self.fix_tide = True 
        else:
            self.fix_tdie = False

        # compute init porb from final porb, ecc, tau, ...
        if "secondary.dOrbPeriod" in self.inparams_fix.names:
            self.fix_porb = True 
        else:
            self.fix_porb = False

        if data['module'] == "eqtide":
            self.fix_radius = True
        else:
            self.fix_radius = False

        if verbose == True:
            print(f"fix_tide: {self.fix_tide}, fix_porb: {self.fix_porb}, fix_radius: {self.fix_radius}")


    def compute_porb(self, theta_run):

        m1 = theta_run["primary.dMass"] * self.inparams_all.dict_units["primary.dMass"]
        m2 = theta_run["secondary.dMass"] * self.inparams_all.dict_units["secondary.dMass"]
        r1 = theta_run["primary.dRadius"] * self.inparams_all.dict_units["primary.dRadius"]
        r2 = theta_run["secondary.dRadius"] * self.inparams_all.dict_units["secondary.dRadius"]
        prot1_i = theta_run["primary.dRotPeriod"] * self.inparams_all.dict_units["primary.dRotPeriod"]
        prot2_i = theta_run["secondary.dRotPeriod"] * self.inparams_all.dict_units["secondary.dRotPeriod"]
        ecc_i = theta_run["secondary.dEcc"] * self.inparams_all.dict_units["secondary.dEcc"]
        rg1, rg2 = 0.45, 0.45

        # m1 = self.inparams_all.dict_true["primary.dMass"] * self.inparams_all.dict_units["primary.dMass"]
        # m2 = self.inparams_all.dict_true["secondary.dMass"] * self.inparams_all.dict_units["secondary.dMass"]
        # prot1_i = self.inparams_all.dict_true["primary.dRotPeriod"] * self.inparams_all.dict_units["primary.dRotPeriod"]
        # prot2_i = self.inparams_all.dict_true["secondary.dRotPeriod"] * self.inparams_all.dict_units["secondary.dRotPeriod"]
        # ecc_i = self.inparams_all.dict_true["secondary.dEcc"] * self.inparams_all.dict_units["secondary.dEcc"]

        # # get initial radius and radius of gyration
        # stellar_in = {"star.dMass": self.inparams_all.dict_units["primary.dMass"], "vpl.dStopTime": u.Myr}
        # stellar_out = {"initial.star.Radius": u.Rsun, "initial.star.RadGyra": u.dimensionless_unscaled}
        # stellar_path = os.path.join(vpi.INFILE_DIR, "stellar")
        # stel = vpi.VplanetModel(stellar_in, inpath=stellar_path, outparams=stellar_out, verbose=False)
        # r1_i, rg1_i = stel.run_model([theta_run["primary.dMass"], 5.0])
        # r2_i, rg2_i = stel.run_model([theta_run["secondary.dMass"], 5.0])
        # r1_i *= u.Rsun
        # r2_i *= u.Rsun

        Jrot1_f = self.outparams.dict_true["final.primary.RotAngMom"] * self.outparams.dict_units["final.primary.RotAngMom"]
        Jrot2_f = self.outparams.dict_true["final.secondary.RotAngMom"] * self.outparams.dict_units["final.secondary.RotAngMom"]
        Jorb_f = self.outparams.dict_true["final.secondary.OrbAngMom"] * self.outparams.dict_units["final.secondary.OrbAngMom"]
        Jtot_f = Jrot1_f + Jrot2_f + Jorb_f

        Jrot1_i = (m1 * (rg1 * r1)**2 * (2*np.pi / prot1_i)).si
        Jrot2_i = (m2 * (rg2 * r2)**2 * (2*np.pi / prot2_i)).si
        Jorb_i = Jtot_f - Jrot1_i - Jrot2_i

        a_i = (Jorb_i**2 * (m1 + m2) / (const.G * m1**2 * m2**2 * (1 - ecc_i**2))).si
        porb_i = (2*np.pi * np.sqrt(a_i**3 / (const.G * (m1 + m2)))).si

        porb_i_value = porb_i.to(self.inparams_all.dict_units["secondary.dOrbPeriod"]).value
        print(porb_i_value, theta_run["secondary.dOrbPeriod"])

        return porb_i_value


    def format_theta(self, theta_var):

        theta_run = copy.copy(self.inparams_all.dict_true)
        theta_var = dict(zip(self.inparams_var.names, theta_var))

        for key in self.inparams_var.names:
            theta_run[key] = theta_var[key]

        if self.fix_tide == True:
            if self.tide_model == "ctl":
                theta_run["secondary.dTidalTau"] = theta_run["primary.dTidalTau"]
            elif self.tide_model == "cpl":
                theta_run["secondary.dTidalQ"] = theta_run["primary.dTidalQ"]

        if self.fix_radius == True:
            theta_run["primary.dRadius"] = theta_run["primary.dMass"]
            theta_run["secondary.dRadius"] = theta_run["secondary.dMass"]

        # if self.fix_porb == True:
        #     theta_run["secondary.dOrbPeriod"] = self.compute_porb(theta_run)

        return np.array(list(theta_run.values()))