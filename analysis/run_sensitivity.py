import argparse
from functools import partial
import multiprocessing as mp
import os
import sys
sys.path.append(os.path.realpath("../src"))
import tidal

EXECUTABLE = "/home/jbirky/Dropbox/packages/vplanet-private/bin/vplanet"

config_file = "config/config_106.yaml"
ncore = mp.cpu_count()
nsample = 16

# Parse config.yaml file
synth = tidal.SyntheticModel(config_file, verbose=False, ncore=ncore, nsample=nsample, vpm_kwargs={"executable": EXECUTABLE})
synth.compute_evolutions()