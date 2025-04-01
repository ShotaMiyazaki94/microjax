import MulensModel
import os.path
import numpy as np

my_1S2L_model = MulensModel.Model(
    {'t_0': 2452848.06, 'u_0': 0.133, 't_E': 61.5, 'rho': 0.00096, 
     'q': 0.0039, 's': 1.120, 'alpha': 223.8})
my_1S2L_model.set_magnification_methods([2452833., 'VBBL', 2452845.])


OGLE_data = MulensModel.MulensData(
    file_name=os.path.join(
        MulensModel.DATA_PATH, "photometry_files", "OB03235", "OB03235_OGLE.tbl.txt"),
    comments=['\\','|'], plot_properties={'label': 'OGLE', 'color': 'black'})

MOA_data = MulensModel.MulensData(
    file_name=os.path.join(
        MulensModel.DATA_PATH, "photometry_files", "OB03235", "OB03235_MOA.tbl.txt"),
        phot_fmt='flux', comments=['\\','|'], plot_properties={'label': 'MOA', 'color': 'red'})

my_event = MulensModel.Event(datasets=[MOA_data, OGLE_data], model=my_1S2L_model)

fs_moa,  fb_moa  = my_event.get_flux_for_dataset(dataset=0)
fs_ogle, fb_ogle = my_event.get_flux_for_dataset(dataset=1)

flux_moa  = MOA_data.flux
flux_ogle = OGLE_data.flux 
fluxe_moa  = MOA_data.err_flux
fluxe_ogle = OGLE_data.err_flux 

def align_flux(flux, fluxe, fs, fb, fs_ref, fb_ref):
    flux_aligned  = (flux-fb)*fs_ref/fs + fb_ref
    fluxe_aligned = fluxe/fs*fs_ref
    return flux_aligned, fluxe_aligned

flux_moa_align, fluxe_moa_align = align_flux(flux_moa, fluxe_moa, fs_moa, fb_moa, fs_ogle, fb_ogle)

data_moa  = np.array([MOA_data.time, flux_moa_align, fluxe_moa_align])
data_ogle = np.array([OGLE_data.time, flux_ogle, fluxe_ogle])

np.save("example/ogle-2003-blg-235/flux_moa", data_moa)
np.save("example/ogle-2003-blg-235/flux_ogle", data_ogle)
