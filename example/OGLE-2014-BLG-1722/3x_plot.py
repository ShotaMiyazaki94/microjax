import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

# idata 読み込み
idata = az.from_netcdf("example/OGLE-2014-BLG-1722/mcmc_full.nc")

param_names = ['alpha', 'log_q', 'log_q3', 'log_s', 'log_s2', 'log_tE', 'piEE', 'piEN', 'psi', 't0_diff', 'u0']

posterior_samples = idata.posterior
samples = np.concatenate(
    [posterior_samples[param].values.reshape(-1, 1) for param in param_names],
    axis=1
)

labels = [f"${name}$" for name in param_names]
figure = corner.corner(samples, labels=labels, show_titles=True,
                       title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 14})

figure.savefig("example/OGLE-2014-BLG-1722/corner_plot.png", dpi=300)