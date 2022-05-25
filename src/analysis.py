import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from functools import partial
import multiprocessing as mp
import os
import time
import tqdm
import copy
import yaml
from yaml.loader import SafeLoader
from collections import OrderedDict
from astropy import units as u

import vplanet_inference as vpi


__all__ = ["SyntheticModel"]


class SyntheticModel(object):
    
    def __init__(self, cfile, inpath=None, outpath=None, fix_tide=True, verbose=True, ncore=mp.cpu_count()):

        # number of CPU cores for parallelization
        self.ncore = ncore
        self.cfile = cfile
        self.config_id = self.cfile.split("/")[-1].split(".yaml")[0].strip("/")

        if outpath is None:
            self.outpath = os.path.join("results", self.config_id)
        else: 
            self.outpath = outpath
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)
    
        # load YAML config file
        with open(self.cfile) as f:
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

            self.labels.append(eval(data['input_var'][key]['label']))
            self.bounds.append(eval(data['input_var'][key]['prior_bounds']))
            self.prior_data.append(eval(data['input_var'][key]['prior_data']))

        # format output parameters
        self.outparams = OrderedDict()
        self.like_data = []     # for problems with observational constraints
        self.output_unc = []    # for problems with synthetic observations

        for key in data['output'].keys():
            self.outparams[key] = eval(data['output'][key]['units'])
            try:
                self.output_unc.append(float(data['output'][key]['uncertainty']))
                synthetic = True
            except: 
                self.like_data.append(eval(data['output'][key]['like_data']))
                synthetic = False

        self.fix_tide = fix_tide

        # combined dictionary of all input parameters
        self.inparams_all = {**self.inparams_fix, **self.inparams_var}

        # initialize vplanet model
        if 'inpath' is None:
            inpath = os.path.join(vpi.INFILE_DIR, data['module'], data['tide_model'])
        self.vpm = vpi.VplanetModel(self.inparams_all, inpath=inpath, outparams=self.outparams, verbose=verbose)

        # run vplanet model on true parameters
        if synthetic == True:
            self.output_true = self.vpm.run_model(np.array(list(self.theta_true.values())))
            self.like_data = np.vstack([self.output_true, self.output_unc]).T


    def format_theta(self, theta_var):

        theta_run = copy.copy(self.theta_true)
        theta_var = dict(zip(list(self.inparams_var.keys()), theta_var))

        for key in self.inparams_var.keys():
            theta_run[key] = theta_var[key]

        if self.fix_tide == True:
            theta_run['secondary.dTidalTau'] = theta_run['primary.dTidalTau']

        return np.array(list(theta_run.values()))


    def run_model_format(self, theta_var):

        # format fixed theta + variable theta
        theta_run = self.format_theta(theta_var)

        # run model
        output = self.vpm.run_model(theta_run)

        return output


    def run_models(self, theta_var_array):

        t0 = time.time()

        if ncore <= 1:
            outputs = np.zeros(theta_var_array.shape[0])
            for ii, tt in tqdm.tqdm(enumerate(theta_var_array)):
                outputs[ii] = self.run_model_format(tt)
        else:
            with mp.Pool(ncore) as p:
                # outputs = np.array(p.map(self.run_model_format, theta_var_array))

                outputs = []
                for result in tqdm(p.imap(func=self.run_model_format, iterable=theta_var_array), total=len(theta_var_array)):
                    outputs.append(result)
                outputs = np.array(outputs)

        tf = time.time()
        print(f"Computed {len(theta)} function evaluations: {np.round(tf - t0)}s \n")

        return outputs


    def lnlike(self, theta_var):

        output = self.run_model_format(theta_var)

        # compute log likelihood
        lnl = -0.5 * np.sum(((output - self.like_data.T[0])/self.like_data.T[1])**2)

        return lnl


    def run_mcmc(self, method="dynesty", reload=True, 
                       kernel="ExpSquaredKernel",
                       ntrain=1000, ntest=100, niter=500):

        import alabi

        if method == "dynesty":
            savedir = os.path.join("dynesty/", self.config_id)

        elif method == "alabi":
            savedir = os.path.join("alabi/", self.config_id)

        # set up prior for dynesty
        self.ptform = partial(alabi.utility.prior_transform_normal, bounds=self.bounds, data=self.prior_data)

        # Configure MCMC
        if reload == True:
            sm = alabi.load_model_cache(savedir)
        else:
            prior_sampler = partial(alabi.utility.prior_sampler, bounds=self.bounds, sampler='uniform')
            sm = alabi.SurrogateModel(fn=self.lnlike, bounds=self.bounds, prior_sampler=prior_sampler,
                                    savedir=savedir, labels=self.labels)
            sm.init_samples(ntrain=ntrain, ntest=test, reload=False)
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


    def variance_global_sensitivity(self, param_values=None, Y=None, nsample=1024):

        from SALib.sample import saltelli
        from SALib.analyze import sobol

        problem = {
            'num_vars': len(self.bounds),
            'names': self.labels,
            'bounds': self.bounds
        }

        if param_values is None:
            param_values = saltelli.sample(problem, nsample)

        if Y is None:
            Y = self.run_models(param_values)

        np.savez(self.outpath + "var_global_sensitivity_sample.npz", param_values=param_values, Y=Y)

        dict_s1 = {'input':synth.inparams_var.keys()}
        dict_sT = {'input':synth.inparams_var.keys()}
                        
        for ii in range(Y.shape[1]):
            res = sobol.analyze(problem, Y.T[ii])
            dict_s1[list(synth.outparams.keys())[ii]] = res['S1']
            dict_sT[list(synth.outparams.keys())[ii]] = res['ST']
            
        table_s1 = pd.DataFrame(data=dict_s1)
        table_s1 = table_s1.set_index("input").round(2)
        table_s1[table_s1.values <= 0] = 0
        table_s1[table_s1.values > 1] = 1
        self.table_s1 = table_s1

        table_sT = pd.DataFrame(data=dict_sT)
        table_sT = table_sT.set_index("input").round(2)
        table_sT[table_sT.values <= 0] = 0
        table_sT[table_sT.values > 1] = 1
        self.table_sT = table_sT

        return table_s1, table_sT


    def show_variance_table(self, table):

        fig = plt.figure(figsize=[12,8])
        sn.heatmap(table, annot=True, annot_kws={"size": 18}) 
        plt.xticks(rotation=45)
        plt.show()

        return fig