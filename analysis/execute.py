import argparse
import multiprocessing as mp
from astropy import units as u
import os
import sys
sys.path.append(os.path.realpath("../src"))
import tidal

ctl = ["050", "051", "052", "053", "054", "055", "056"]
cpl = ["057", "058", "059", "060", "061", "062", "063"]
ctl_reduced = ["064", "065", "066", "067", "068", "069", "070"]
cpl_reduced = ["071", "072", "073", "074", "075", "076", "077"]
ctl_stellar = ["078", "079", "080", "081", "082", "083", "084"]
cpl_stellar = ["085", "086", "087", "088", "089", "090", "091"]
ctl_stellar_reduced = ["092", "093", "094", "095", "096", "097", "098"]
cpl_stellar_reduced = ["099", "100", "101", "102", "103", "104", "105"]

EXECUTABLE = "/home/jbirky/Dropbox/packages/vplanet-private/bin/vplanet"

# Parse user commands
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--op", type=str, required=True)
parser.add_argument("--method", type=str, default="alabi")

# sensitivity options
parser.add_argument("--nsample", type=int, default=1024)

# alabi options
parser.add_argument("--reload", type=bool, default=False)
parser.add_argument("--niter", type=int, default=500)
parser.add_argument("--ntrain", type=int, default=1000)
parser.add_argument("--ntest", type=int, default=100)

# multiprocessing options
parser.add_argument("--ncore", type=int, default=mp.cpu_count())
parser.add_argument("--nice", type=int, default=10)

# vplanet model options
parser.add_argument("--timesteps", default=None)
parser.add_argument("--time_init", default=5e6*u.yr)

# collect args
args = parser.parse_args()
os.nice(args.nice)

# Parse config.yaml file
vpm_kwargs = {"executable": EXECUTABLE}
vpm_kwargs["timesteps"] = args.timesteps
vpm_kwargs["time_init"] = args.time_init

synth = tidal.SyntheticModel(args.file, verbose=False, ncore=args.ncore, compute_true=True, vpm_kwargs=vpm_kwargs)

if args.op.lower()[0:4] == "sens":
    synth.variance_global_sensitivity(nsample=args.nsample, save=True)

elif args.op.lower() == "mcmc":
    synth.run_mcmc(method=args.method, reload=args.reload, niter=args.niter,
                   ntrain=args.ntrain, ntest=args.ntest)