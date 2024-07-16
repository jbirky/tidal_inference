import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sn
import os
import tqdm
import copy
import yaml
from yaml.loader import SafeLoader
from astropy import units as u
from astropy import constants as const
from functools import partial

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
        # super().__init__(cfile, 
        #                  inpath=inpath,
        #                  outpath=outpath, 
        #                  verbose=verbose, 
        #                  compute_true=compute_true,
        #                  ncore=ncore,
        #                  vpm_kwargs={},
        #                  **kwargs)

        #------------------------------
        # # super().__init__ functions (do not use super() for now)

        # number of CPU cores for parallelization
        self.ncore = ncore
        self.cfile = cfile
        self.config_id = self.cfile.split("/")[-1].split(".yaml")[0].strip("/")

        # directory to save results (defaults to local)
        self.outpath = outpath
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        # load YAML config file
        with open(self.cfile) as f:
            data = yaml.load(f, Loader=SafeLoader)

        # make porb/prot ratio an output parameter?
        data_output = copy.copy(data["output"])
        if "final.OrbPeriod_RotPer" in list(data["output"].keys()):
            self.period_ratio = True
            del data_output["final.OrbPeriod_RotPer"]
            self.porb_arg = list(data_output.keys()).index("final.secondary.OrbPeriod")
            self.prot_arg = list(data_output.keys()).index("final.primary.RotPer")
        else:
            self.period_ratio = False


        inparams_fix = VplanetParameters(names=list(data["input_fix"].keys()),
                                             units=sort_yaml_key(data["input_fix"], "units"),
                                             true=sort_yaml_key(data["input_fix"], "true_value"),
                                             labels=sort_yaml_key(data["input_fix"], "label"))

        inparams_var = VplanetParameters(names=list(data["input_var"].keys()),
                                             units=sort_yaml_key(data["input_var"], "units"),
                                             true=sort_yaml_key(data["input_var"], "true_value"),
                                             bounds=sort_yaml_key(data["input_var"], "bounds"),
                                             data=sort_yaml_key(data["input_var"], "data"),
                                             labels=sort_yaml_key(data["input_var"], "label"))

        inparams_all = VplanetParameters(names=inparams_fix.names + inparams_var.names,
                                             units=inparams_fix.units + inparams_var.units,
                                             true=inparams_fix.true + inparams_var.true)

        outparams = VplanetParameters(names=list(data_output.keys()),
                                          units=sort_yaml_key(data_output, "units"),
                                          data=sort_yaml_key(data_output, "data"),
                                          uncertainty=sort_yaml_key(data_output, "uncertainty"),
                                          labels=sort_yaml_key(data_output, "label"))

        # initialize vplanet model
        if inpath is None:
            self.inpath = data["inpath"]
        else:
            self.inpath = inpath

        # vpm_kwargs["timesteps"] = timesteps
        # vpm_kwargs["time_init"] = time_init
        self.vpm = vpi.VplanetModel(inparams_all.dict_units, 
                                    inpath=self.inpath, 
                                    outparams=outparams.dict_units, 
                                    verbose=verbose,
                                    **vpm_kwargs)

        # if this is a synthetic model test, run vplanet model on true parameters
        if (outparams.data[0] is None) & (outparams.uncertainty[0] is not None) & (compute_true == True):
            output_true = self.vpm.run_model(inparams_all.true)
            outparams.set_data(output_true)

        # format period ratio as an output parameter
        if self.period_ratio == True:
            output_names = list(data["output"].keys())
            output_unc = sort_yaml_key(data["output"], "uncertainty")
            if "output_true" not in locals():
                output_true = self.vpm.run_model(inparams_all.true)
            output_true = np.append(output_true, output_true[self.prot_arg]/output_true[self.porb_arg])
            output_data = np.array([output_true, output_unc]).T

            outparams.true = output_true
            outparams.data = output_data
            
            outparams.dict_true = dict(zip(output_names, output_true))
            outparams.dict_data = dict(zip(output_names, output_data))

        # for alabi
        self.like_data = outparams.data

        self.inparams_fix = inparams_fix
        self.inparams_var = inparams_var
        self.inparams_all = inparams_all
        self.outparams = outparams

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

        if self.period_ratio == True:
            output = np.append(output, output[self.porb_arg]/output[self.prot_arg])

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


    def lnlike(self, theta_var):

        output = self.run_model_format(theta_var)

        # compute log likelihood
        lnl = -0.5 * np.sum(((output - self.outparams.data.T[0])/self.outparams.data.T[1])**2)

        return lnl


    def run_mcmc(self, method="dynesty", 
                       reload_sm=False, 
                       kernel="ExpSquaredKernel",
                       ntrain=1000, 
                       ntest=100, 
                       niter=500,
                       reload_samp=False):

        if self.like_data is None:
            raise Exception("No likelihood data specified.")

        try:
            import alabi
        except:
            raise Exception("Dependency 'alabi' not installed. To install alabi run: \n\n" + 
                            "git clone https://github.com/jbirky/alabi \n" +
                            "cd alabi \n" +
                            "python setup.py install")

        if method == "dynesty":
            savedir = os.path.join(self.outpath, "results_dynesty/", self.config_id)

        elif method == "alabi":
            savedir = os.path.join(self.outpath, "results_alabi/", self.config_id)

        # set up prior for dynesty
        self.ptform = partial(alabi.utility.prior_transform_normal, 
                              bounds=self.inparams_var.bounds, 
                              data=self.inparams_var.data)

        # Configure MCMC
        if reload_sm == True:
            sm = alabi.load_model_cache(savedir)
        else:
            prior_sampler = partial(alabi.utility.prior_sampler, bounds=self.inparams_var.bounds, sampler='uniform')
            sm = alabi.SurrogateModel(fn=self.lnlike, 
                                      bounds=self.inparams_var.bounds, 
                                      prior_sampler=prior_sampler,
                                      savedir=savedir, 
                                      labels=self.inparams_var.labels,
                                      verbose=True,
                                      cache=True)
            sm.init_samples(ntrain=ntrain, ntest=ntest, reload=reload_samp)
            sm.init_gp(kernel=kernel, fit_amp=False, fit_mean=True, white_noise=-15)

        # Run MCMC
        if method == "dynesty":
            sm.run_dynesty(like_fn="true", ptform=self.ptform, mode="static", multi_proc=True, save_iter=100)
            sm.plot(plots=["dynesty_all"])

        elif method == "alabi":
            sm.active_train(niter=niter, algorithm="bape", gp_opt_freq=10, save_progress=True)
            sm.plot(plots=["gp_all"])
            sm.run_dynesty(ptform=self.ptform, mode="static", multi_proc=True, save_iter=100)
            sm.plot(plots=["dynesty_all"])

        return None


    def plot_sensitivity_table(self, table):

        # plot sensitivity tables
        fig = plt.figure(figsize=[12,8])
        sn.heatmap(table, yticklabels=self.inparams_var.labels, xticklabels=self.outparams.labels, 
                   annot=True, annot_kws={"size": 18}, vmin=0, vmax=1, cmap="bone") 
        plt.title("First order sensitivity (S1) index", fontsize=25)
        plt.xticks(rotation=45, fontsize=18, ha='right')
        plt.yticks(rotation=0)
        plt.yticks(fontsize=18)
        plt.xlabel("Final Conditions", fontsize=22)
        plt.ylabel("Initial Conditions", fontsize=22)
        plt.close()

        return fig


    def variance_global_sensitivity(self, param_values=None, Y=None, nsample=1024, save=False, subpath="results_sensitivity"):

        from SALib.sample import saltelli
        from SALib.analyze import sobol

        problem = {
            "num_vars": self.inparams_var.num,
            "names": self.inparams_var.names,
            "bounds": self.inparams_var.bounds
        }

        if param_values is None:
            param_values = saltelli.sample(problem, nsample)

        if Y is None:
            Y = self.run_models(param_values)

        # save samples to npz file
        savedir = os.path.join(self.outpath, subpath, self.config_id)
        if not os.path.exists(savedir):
            os.makedirs(savedir)    
        if save == True:       
            np.savez(f"{savedir}/var_global_sensitivity_sample.npz", param_values=param_values, Y=Y)

        dict_s1 = {"input": self.inparams_var.names}
        dict_sT = {"input": self.inparams_var.names}
                        
        for ii in range(Y.shape[1]):
            res = sobol.analyze(problem, Y.T[ii])
            dict_s1[self.outparams.names[ii]] = res['S1']
            dict_sT[self.outparams.names[ii]] = res['ST']
            
        table_s1 = pd.DataFrame(data=dict_s1).round(2)
        table_s1 = table_s1.set_index("input").rename_axis(None, axis=0)
        table_s1[table_s1.values <= 0] = 0
        table_s1[table_s1.values > 1] = 1
        self.table_s1 = table_s1
        if save == True:
            table_s1.to_csv(f"{savedir}/sensitivity_table_s1.csv")

        table_sT = pd.DataFrame(data=dict_sT).round(2)
        table_sT = table_sT.set_index("input").rename_axis(None, axis=0)
        table_sT[table_sT.values <= 0] = 0
        table_sT[table_sT.values > 1] = 1
        self.table_sT = table_sT
        if save == True:
            table_sT.to_csv(f"{savedir}/sensitivity_table_sT.csv")

        self.fig_s1 = self.plot_sensitivity_table(self.table_s1)
        self.fig_sT = self.plot_sensitivity_table(self.table_sT)
        if save == True:
            self.fig_s1.savefig(f"{savedir}/sensitivity_table_s1.png", bbox_inches="tight")
            self.fig_sT.savefig(f"{savedir}/sensitivity_table_sT.png", bbox_inches="tight")