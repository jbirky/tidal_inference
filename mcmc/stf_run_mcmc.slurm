#!/bin/bash

## Job Name
#SBATCH --job-name=jbirky_mcmc
#SBATCH --account=stf
#SBATCH --partition=stf

#SBATCH --nodes=1   
#SBATCH --ntasks-per-node=28

#SBATCH --time=1-00:00:00
#SBATCH --mem=150G

## Specify the working directory for this job
#SBATCH --chdir=/gscratch/rory/jbirky/projects/tidal_inference/mcmc

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jbirky@uw.edu
#SBATCH --export=all

module load anaconda3_4.3.1
python execute.py --file config/config_009.yaml --method dynesty
