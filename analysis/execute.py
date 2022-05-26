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
parser.add_argument("--operation", type=str, required=True)
parser.add_argument("--method", type=str, default="dynesty")
parser.add_argument("--reload", type=bool, default=False)
parser.add_argument("--niter", type=int, default=500)
parser.add_argument("--ncore", type=int, default=mp.cpu_count())
args = parser.parse_args()

# Parse config.yaml file
synth = analysis.SyntheticModel(args.file, verbose=False, ncore=args.ncore)

if args.operation == "sensitivity":
    synth.variance_global_sensitivity()

elif args.operation == "mcmc":
    synth.run_mcmc(method=args.method, reload=args.reload, niter=args.niter)