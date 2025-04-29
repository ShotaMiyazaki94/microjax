import pandas as pd
import numpy as np

photo_file = pd.read_csv("example/data/ogle-2014-blg-0124/phot.dat", delim_whitespace=True, header=None)
photo_file.columns = ["HJD", "I", "Ie", "seeing", "sky"]
photo_file["HJD"] = photo_file["HJD"].astype(float)
photo_file["I"] = photo_file["I"].astype(float)
photo_file["Ie"] = photo_file["Ie"].astype(float)


mag0 = 18.0
Flux = 10**(-0.4 * (photo_file["I"].values - mag0))
Flux_err = 0.4 * np.log(10) * photo_file["Ie"].values * Flux
Flux_err = np.abs(Flux_err)

# Convert to numpy array
data = np.array([photo_file["HJD"].values, Flux, Flux_err, photo_file["seeing"].values, photo_file["sky"].values])
data = data[:, (data[0] > 2456300)]
print(data)
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 4))
plt.errorbar(data[0], data[1], yerr=data[2], fmt='o', markersize=2, label='Data')
plt.xlabel('HJD')
plt.ylabel('Flux')
plt.title('Photometry Data')
plt.savefig("example/ob140124/flux.png", dpi=100)
# Save to .npy file
np.save("example/ob140124/flux", data)