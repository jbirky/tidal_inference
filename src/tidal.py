import numpy as np
import multiprocessing as mp
import os
import tqdm
import copy
import yaml
from yaml.loader import SafeLoader
from astropy import units as u
from astropy import constants as const

from SALib.sample import saltelli
from SALib.analyze import sobol

import vplanet_inference as vpi
from vplanet_inference.parameters import VplanetParameters
from vplanet_inference import sort_yaml_key


__all__ = ["SyntheticModel"]


class SyntheticModel(vpi.AnalyzeVplanetModel):

    def __init__(self, 
                 cfile, 
                 outpath=".", 
                 verbose=True, 
                 compute_true=False,
                 ncore=mp.cpu_count(),
                 nsample=1024,
                #  timesteps=5e7*u.yr,
                #  time_init=5e6*u.yr,
                 vpm_kwargs={},
                 **kwargs):

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
                         ncore=ncore,
                         vpm_kwargs={},
                         **kwargs)

        #------------------------------
        # # super().__init__ functions

        # # number of CPU cores for parallelization
        # self.ncore = ncore
        # self.cfile = cfile
        # self.config_id = self.cfile.split("/")[-1].split(".yaml")[0].strip("/")

        # # directory to save results (defaults to local)
        # self.outpath = outpath
        # if not os.path.exists(self.outpath):
        #     os.makedirs(self.outpath)

        # # load YAML config file
        # with open(self.cfile) as f:
        #     data = yaml.load(f, Loader=SafeLoader)

        # inparams_fix = VplanetParameters(names=list(data["input_fix"].keys()),
        #                                      units=sort_yaml_key(data["input_fix"], "units"),
        #                                      true=sort_yaml_key(data["input_fix"], "true_value"),
        #                                      labels=sort_yaml_key(data["input_fix"], "label"))

        # inparams_var = VplanetParameters(names=list(data["input_var"].keys()),
        #                                      units=sort_yaml_key(data["input_var"], "units"),
        #                                      true=sort_yaml_key(data["input_var"], "true_value"),
        #                                      bounds=sort_yaml_key(data["input_var"], "bounds"),
        #                                      data=sort_yaml_key(data["input_var"], "data"),
        #                                      labels=sort_yaml_key(data["input_var"], "label"))

        # inparams_all = VplanetParameters(names=inparams_fix.names + inparams_var.names,
        #                                      units=inparams_fix.units + inparams_var.units,
        #                                      true=inparams_fix.true + inparams_var.true)

        # outparams = VplanetParameters(names=list(data["output"].keys()),
        #                                   units=sort_yaml_key(data["output"], "units"),
        #                                   data=sort_yaml_key(data["output"], "data"),
        #                                   uncertainty=sort_yaml_key(data["output"], "uncertainty"),
        #                                   labels=sort_yaml_key(data["output"], "label"))

        # # initialize vplanet model
        # if inpath is None:
        #     self.inpath = data["inpath"]
        # else:
        #     self.inpath = inpath

        # # vpm_kwargs["timesteps"] = timesteps
        # # vpm_kwargs["time_init"] = time_init
        # self.vpm = vpi.VplanetModel(inparams_all.dict_units, 
        #                         inpath=self.inpath, 
        #                         outparams=outparams.dict_units, 
        #                         verbose=verbose,
        #                         **vpm_kwargs)

        # # if this is a synthetic model test, run vplanet model on true parameters
        # if (outparams.data[0] is None) & (outparams.uncertainty[0] is not None) & (compute_true == True):
        #     output_true = self.vpm.run_model(inparams_all.true)
        #     outparams.set_data(output_true)

        # self.inparams_fix = inparams_fix
        # self.inparams_var = inparams_var
        # self.inparams_all = inparams_all
        # self.outparams = outparams

        # ---------------------------------
        # tidal inference specific functions

        # set primary and secondary to same value?
        if "secondary.dTidalTau" in self.inparams_fix.names:
            self.fix_tide = True 
        elif "secondary.dTidalQ" in self.inparams_fix.names:
            self.fix_tide = True 
        else:
            self.fix_tide = False

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


        problem = {
            "num_vars": self.inparams_var.num,
            "names": self.inparams_var.names,
            "bounds": self.inparams_var.bounds
        }

        self.saltelli_sample = saltelli.sample(problem, nsample)
    

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


    def run_model_format(self, theta_var, **kwargs):

        # format fixed theta + variable theta
        theta_run = self.format_theta(theta_var)

        # run model
        output = self.vpm.run_model(theta_run, remove=True)

        return output


    def run_models(self, theta_var_array, **kwargs):

        if self.ncore <= 1:
            outputs = np.zeros(theta_var_array.shape[0])
            for ii, tt in tqdm.tqdm(enumerate(theta_var_array)):
                outputs[ii] = self.run_model_format(tt, **kwargs)
        else:
            with mp.Pool(self.ncore) as p:
                outputs = []
                for result in tqdm.tqdm(p.imap(func=self.run_model_format, iterable=theta_var_array), total=len(theta_var_array)):
                    outputs.append(result)
                outputs = np.array(outputs)

        return outputs



    def compute_evolutions(self, theta_var_array=None, save=True):

        if theta_var_array is None:
            theta_var_array = self.saltelli_sample
            
        Y = self.run_models(theta_var_array)

        # save samples to npz file
        savedir = os.path.join(self.outpath, "results_sensitivity", self.config_id)
        if not os.path.exists(savedir):
            os.makedirs(savedir)    
        if save == True:       
            np.savez(f"{savedir}/model_evolution.npz", theta=theta_var_array, Y=Y)