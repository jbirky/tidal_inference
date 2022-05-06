import argparse
import os
import sys
sys.path.append("../src/")
import mcmc
import alabi


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()

# Parse config.yaml file
lnlike, ptform, labels, bounds = mcmc.configure_model(args.file)

base_dir = os.path.dirname(os.path.realpath(args.file))
config_name = args.file.split("/")[-1].split(".yaml")[0].strip("/")
savedir = os.path.join(base_dir, config_name)

# Run dynesty
sm = alabi.SurrogateModel(fn=lnlike, bounds=bounds, savedir=savedir, labels=labels)
sm.run_dynesty(like_fn="true", ptform=ptform, mode="dynamic")
sm.plot(plots=["dynesty_all"])