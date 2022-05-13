import argparse
import os
import sys
sys.path.append("../src/")
import mcmc
import alabi


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)
parser.add_argument("--method", type=str, default="dynesty")
parser.add_argument("--reload", type=bool, default=False)
args = parser.parse_args()

# # Parse config.yaml file
# lnlike, ptform, labels, bounds = mcmc.configure_model(args.file)

synth = mcmc.SyntheticModel(args.file, verbose=False)

# base_dir = os.path.dirname(os.path.realpath(args.file))
config_name = args.file.split("/")[-1].split(".yaml")[0].strip("/")
savedir = os.path.join("results/", config_name)

# Run dynesty
if args.reload == True:
    sm = alabi.load_model_cache(savedir)
else:
    sm = alabi.SurrogateModel(fn=synth.lnlike, bounds=synth.bounds, savedir=savedir, labels=synth.labels)

if args.method == "dynesty":
    sm.run_dynesty(like_fn="true", ptform=synth.ptform, mode="dynamic", multi_proc=True)
    # sm.plot(plots=["dynesty_all"])