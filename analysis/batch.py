import os
import sys
sys.path.append(os.path.realpath("../src"))
import analysis

ctl_only = ["050", "051", "052", "053", "054", "055", "056"]
cpl_only = ["057", "058", "059", "060", "061", "062", "063"]

for cid in cpl_only;
    file = f"config/config_{cid}.yaml"
    synth = analysis.SyntheticModel(file, verbose=False)
    synth.variance_global_sensitivity(nsample=512)