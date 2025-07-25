## Prospects of Constraining Equilibrium Tides in Low-Mass Binary Stars

Authors: Jessica Birky, Rory Barnes, James Davenport

Link to paper: [http://arxiv.org/abs/2507.12639](
http://arxiv.org/abs/2507.12639)

### Main code dependencies:

- [vplanet](https://github.com/VirtualPlanetaryLaboratory/vplanet)
- [vplanet_inference](https://github.com/jbirky/vplanet_inference)
- [alabi](https://github.com/jbirky/alabi)

### Code to reproduce figures:

Figure 1 - 2: [`evol_sim.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/evol_sim.ipynb)

Figure 3 - 4: [`conservation_figures.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/conservation_figures.ipynb)

Figure 5 - 6: [`model_sensitivity.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/model_sensitivity.ipynb)

Figure 7: [`likelihood_sensitivity.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/likelihood_sensitivity.ipynb)

Figure 8 - 11: [`model_sensitivity_ratio.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/model_sensitivity_ratio.ipynb)

Figure 12 - 15: [`alabi_corner_plots.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/alabi_corner_plots.ipynb)

Figure 16 - 17: [`alabi_corner_zoom.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/alabi_corner_zoom.ipynb)

Figure 18 - 20: [`plot_1d_likelihood.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/plot_1d_likelihood.ipynb)

Figure 21: [`degeneracy_evolutions_ctl.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/degeneracy_evolutions_ctl.ipynb)

Figure 22: [`degeneracy_evolutions_cpl.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/degeneracy_evolutions_cpl.ipynb)

Figure 23: [`plane3d.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/plane3d.ipynb)

Figure 24: [`evol_sim_ratio_ctl.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/evol_sim_ratio_ctl.ipynb)

Figure 25: [`evol_sim_ratio_ctl.ipynb`](https://github.com/jbirky/tidal_inference/blob/main/notebooks/evol_sim_ratio_ctl_stellar.ipynb)

### Configuration files for the published results:

Sensitivity analysis
| age (Myr) | ctl_stellar | cpl_stellar |
|-------|-------------------|-------------------|
| 10     | 078               | 085               |
| 50     | 079               | 086               |
| 100     | 080               | 087               |
| 500     | 081               | 088               |
| 1000     | 082               | 089               |
| 5000     | 083               | 090               |
| 10000     | 084               | 091               |

Posterior inference 
| age (Myr) | ctl_stellar | cpl_stellar |
|-------|-------------------|-------------------|
| 50    | 110               | 120               |
| 5000     | 111               | 121               |