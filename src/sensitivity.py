from SALib.sample import saltelli
from SALib.analyze import sobol

def variance_global_sensitivity(self, param_values=None, Y=None, nsample=1024, save=False):

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
    savedir = os.path.join(self.outpath, "results_sensitivity", self.config_id)
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