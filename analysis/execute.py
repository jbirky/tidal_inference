import argparse
from functools import partial
import multiprocessing as mp
import os
import sys
sys.path.append(os.path.realpath("../src"))
import analysis


# Parse user commands
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--op", type=str, required=True)
parser.add_argument("--method", type=str, default="dynesty")

# sensitivity options
parser.add_argument("--nsample", type=int, default=1024)

# alabi options
parser.add_argument("--reload", type=bool, default=False)
parser.add_argument("--niter", type=int, default=500)

# multiprocessing options
parser.add_argument("--ncore", type=int, default=mp.cpu_count())
parser.add_argument("--nice", type=int, default=10)

args = parser.parse_args()
os.nice(args.nice)

# Parse config.yaml file
synth = analysis.SyntheticModel(args.file, verbose=False, ncore=args.ncore)

if args.op.lower()[0:4] == "sens":
    synth.variance_global_sensitivity(nsample=args.nsample)

elif args.op.lower() == "mcmc":
    synth.run_mcmc(method=args.method, reload=args.reload, niter=args.niter)