import argparse
from functools import partial
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
args = parser.parse_args()

# Parse config.yaml file
synth = analysis.SyntheticModel(args.file, verbose=True)

if args.operation == "sensitivity":
    synth.variance_global_sensitivity()

elif args.operation == "mcmc":
    synth.run_mcmc(method=args.method, reload=args.reload, niter=args.niter)
