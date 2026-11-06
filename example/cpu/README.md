# CPU example

Run from the repository root:

```bash
python -m pip install -e . matplotlib
python example/cpu/lightcurve.py
```

[`lightcurve.py`](lightcurve.py) evaluates a binary-lens light curve with linear
limb darkening (`u1=0.5`) and uses `jax.jvp` to obtain the source-radius derivative
in the same forward pass. The trajectory, lens parameters and limb-darkening
coefficient stay fixed when differentiating. The script enables float64 and
selects the CPU explicitly; the first evaluation includes JAX compilation.

The preview uses the script's default parameters. The upper panel shows
magnification; the lower panel shows its derivative with respect to source
radius, holding the other parameters fixed.

![CPU binary-lens magnification and its forward source-radius derivative](lightcurve.png)

Edit the trajectory and `rho, s, q, u1` near the top of `main()` to explore another
event. `u1=0` uses the same implementation for a uniform source.

The two-panel figure and its numerical data are written to
`results/cpu/example/lightcurve.png` and `lightcurve.csv`. These runtime outputs
stay local and are excluded from Git. Only the representative README preview is
versioned here; to refresh it after running the script:

```bash
cp results/cpu/example/lightcurve.png example/cpu/lightcurve.png
```

This is a usage example. Accuracy and speed measurements live in
[`tools/cpu_benchmark.py`](../../tools/cpu_benchmark.py); regional maps and
uncertainty plots are described in [CONTRIBUTING.md](../../CONTRIBUTING.md).
