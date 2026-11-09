microJAX
========

microJAX is a Python package for modeling gravitational microlensing by
binary and triple lenses. It calculates how a star's brightness changes as
it passes behind a lens system, including the effects of the star's size
and limb darkening (the dimmer appearance of its edge). The lenses can
describe a star with planetary companions or a system of stars.

The package is built with JAX, so you can also calculate how a light curve
changes with each model parameter. These derivatives are useful when fitting
an event or exploring which parts of a light curve constrain the lens.
Both the magnification and its derivatives can be evaluated on CPU or GPU.

If you are new to microJAX, start with :doc:`getting_started`, then work
through :doc:`usage`. That tutorial goes from event parameters to the
light curve below. :doc:`autodiff` builds on the same event. Complete
scripts are in :doc:`examples/index`.

.. figure:: _static/binary_lightcurve.png
   :alt: A binary microlensing light curve with a short feature near its peak
   :width: 100%
   :target: usage.html

   A binary-lens event calculated in the first tutorial.

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   getting_started

.. toctree::
   :caption: Tutorials
   :maxdepth: 1

   usage
   autodiff
   inference
   examples/index

.. toctree::
   :caption: Reference
   :maxdepth: 1

   working_with_jax
   benchmarks
   modules
   citing

The methods are described in
`Miyazaki & Kawahara (2025) <https://doi.org/10.3847/1538-4357/ae1005>`_.
See :doc:`citing` if you use microJAX in your work. For single-lens models,
see `singlens <https://github.com/ShotaMiyazaki94/singlens>`_.

Questions, bug reports, and suggestions are welcome on
`GitHub <https://github.com/ShotaMiyazaki94/microjax/issues>`_.
