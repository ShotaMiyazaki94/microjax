import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

# idata 読み込み
idata = az.from_netcdf("example/OGLE-2014-BLG-1722/mcmc_full.nc")

param_names = ['t0_diff', 'u0', 'log_tE', 'log_q', 'log_s', 'alpha',
                'piEN', 'piEE', 'log_q3', 'log_s2', 'psi']

posterior_samples = idata.posterior
samples = np.concatenate(
    [posterior_samples[param].values.reshape(-1, 1) for param in param_names],
    axis=1
)

labels = [
    r"$t_0^\prime$",        # プライム記号は右上に
    r"$u_0$",               # サブスクリプト付き
    r"$\log t_E$",          # 対数は roman に
    r"$\log q$",            
    r"$\log s$",            
    r"$\alpha$",            # ギリシャ文字
    r"$\pi_{E,N}$",         # 下付き添字にカンマ（E方向）
    r"$\pi_{E,E}$",         # N方向
    r"$\log q_3$",          
    r"$\log s_2$",          
    r"$\psi$",              
]
figure = corner.corner(samples, labels=labels, show_titles=True,
                       title_kwargs={"fontsize": 12}, label_kwargs={"fontsize": 14})

figure.savefig("example/OGLE-2014-BLG-1722/corner_plot.pdf", dpi=300)