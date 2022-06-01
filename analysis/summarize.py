import numpy as np
import pandas as pd
import os 
import sys
sys.path.append(os.path.realpath("../src"))
import analysis


def summary_table(cdir):
    id_list = [f.split("_")[1].split(".yaml")[0] for f in os.listdir(cdir)]

    summ = {'config_id': [], 
            'module': [],
            'tide_model': [],
            'tide_value': [], 
            'age': [], 
            'ninvar': [], 
            'noutvar': [],
            'sensitivity': [],
            'mcmc_alabi': [],
            'mcmc_dynesty': [],
            'input_params': [],
            'output_params': []}
    
    for config_id in id_list:
        synth = analysis.SyntheticModel(f"config/config_{config_id}.yaml", verbose=False, compute_true=False)

        summ['config_id'].append(config_id)
        summ['module'].append(synth.module)
        summ['tide_model'].append(synth.tide_model.upper())
        summ['tide_value'].append(synth.inparams_all.dict_true['primary.dTidalTau'])
        summ['age'].append(int(synth.inparams_all.dict_true['vpl.dStopTime']))
        summ['ninvar'].append(synth.inparams_var.num)
        summ['input_params'].append(', '.join(list(synth.inparams_var.names)))
        summ['output_params'].append(', '.join(list(synth.outparams.names)))
        summ['noutvar'].append(synth.outparams.num)
        
        res_sens = [ff.strip("config_") for ff in os.listdir("results_sensitivity/")]
        if config_id in res_sens:
            summ['sensitivity'].append("X")
        else:
            summ['sensitivity'].append("")
            
        res_sens = [ff.strip("config_") for ff in os.listdir("results_mcmc/alabi/")]
        if config_id in res_sens:
            summ['mcmc_alabi'].append("X")
        else:
            summ['mcmc_alabi'].append("")
            
        res_sens = [ff.strip("config_") for ff in os.listdir("results_mcmc/dynesty/")]
        if config_id in res_sens:
            summ['mcmc_dynesty'].append("X")
        else:
            summ['mcmc_dynesty'].append("")
        
    dt = pd.DataFrame(data=summ)
    
    return dt


dt = summary_table("config").sort_values(by=['config_id'])
# dt.to_csv("config_summary.csv", index=None)
dt.to_markdown("README.md", index=None)