import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.realpath("../src"))
import analysis

import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)


def get_table_s1(config_id):
    
    synth = analysis.SyntheticModel(f"config/config_{config_id}.yaml", verbose=False, compute_true=False)
    samp = np.load(f"results_sensitivity/config_{config_id}/var_global_sensitivity_sample.npz")
    synth.variance_global_sensitivity(param_values=samp['param_values'], Y=samp['Y'])
    
    age = synth.inparams_all.dict_true['vpl.dStopTime']
    
    return age, synth.table_s1


def get_tables_sorted(config_id_list):
    
    ages = []
    tables = []
    for cid in config_id_list:
        age, tab = get_table_s1(cid)
        ages.append(age)
        tables.append(tab)
        
    synth = analysis.SyntheticModel(f"config/config_{config_id_list[0]}.yaml", verbose=False, compute_true=False)
    tsort = dict(zip(synth.outparams.names, [{} for ii in range(synth.outparams.num)]))

    for ii, tab in enumerate(tables):
        for key in tsort:
            tsort[key][str(int(ages[ii]))] = tab[key]
        
    return tsort, synth.inparams_var.labels


def plot_tables_sorted(tsort, labels, model="", save=False, show=True, cmap="bone"):
    
    for key in tsort.keys():
        df = pd.DataFrame(data=tsort[key])
        
        sn.heatmap(df, yticklabels=labels, annot=True, annot_kws={"size": 16}, vmin=0, vmax=1, cmap=cmap)
        plt.title(key + " variance fraction", fontsize=22)
        plt.yticks(rotation=0)
        plt.xlabel("Age [Myr]", fontsize=22)
        plt.ylabel("Initial Conditions", fontsize=22)
        if save == True:
            plt.savefig(f"../draft/figures/sensitivity_{model}_{key.replace('.', '_')}.png", bbox_inches="tight")
        if show == True:
            plt.show()
        plt.close()


ctl = ["050", "051", "052", "053", "054", "055", "056"]
cpl = ["057", "058", "059", "060", "061", "062", "063"]


config_ids = "ctl"

# for cid in eval(config_ids):
#     file = f"config/config_{cid}.yaml"
#     synth = analysis.SyntheticModel(file, verbose=False)
#     synth.variance_global_sensitivity(nsample=512)

if "cpl" in config_ids:
    cmap = "pink"
else:
    cmap = "bone"
tsort, labels = get_tables_sorted(eval(config_ids))
plot_tables_sorted(tsort, labels, model=config_ids, show=False, save=True, cmap=cmap)