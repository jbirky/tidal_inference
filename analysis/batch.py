import os
import sys
sys.path.append(os.path.realpath("../src"))
import analysis

os.nice(10)

ctl = ["050", "051", "052", "053", "054", "055", "056"]
cpl = ["057", "058", "059", "060", "061", "062", "063"]
ctl_reduced = ["064", "065", "066", "067", "068", "069", "070"]
cpl_reduced = ["071", "072", "073", "074", "075", "076", "077"]
ctl_stellar = ["078", "079", "080", "081", "082", "083", "084"]
cpl_stellar = ["085", "086", "087", "088", "089", "090", "091"]
ctl_stellar_reduced = ["092", "093", "094", "095", "096", "097", "098"]
cpl_stellar_reduced = ["099", "100", "101", "102", "103", "104", "105"]

config_ids = "ctl_stellar"

for cid in eval(config_ids):
    if not os.path.exists(f"results_sensitivity/config_{cid}/var_global_sensitivity_sample.npz"):
        file = f"config/config_{cid}.yaml"
        synth = analysis.SyntheticModel(file, verbose=False, ncore=20)
        synth.variance_global_sensitivity(nsample=512)
    else:
        print(f"config {cid} already completed.")