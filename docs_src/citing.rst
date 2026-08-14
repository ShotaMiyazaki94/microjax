Citing and Reproducibility
==========================

Methods paper and software archive
----------------------------------

If you use microJAX, cite the methods paper and the archived software version:

- Miyazaki, S., & Kawahara, H. 2025, ApJ, 994, 144,
  `doi:10.3847/1538-4357/ae1005 <https://doi.org/10.3847/1538-4357/ae1005>`_
- microJAX software archive,
  `doi:10.5281/zenodo.17247892 <https://doi.org/10.5281/zenodo.17247892>`_

The methods paper corresponds to the archived ``v0.1.1`` implementation. The
``0.2`` series substantially redesigns the finite-source solver. Work using
the redesigned solver should therefore report the exact ``0.2.x`` release or
Git commit in addition to citing the paper and software archive.

BibTeX
------

.. code-block:: bibtex

   @ARTICLE{2025ApJ...994..144M,
     author = {{Miyazaki}, Shota and {Kawahara}, Hajime},
     title = {microJAX: A Differentiable Framework for Microlensing Modeling
              with GPU-accelerated Image-centered Ray Shooting},
     journal = {The Astrophysical Journal},
     year = {2025},
     volume = {994},
     number = {2},
     pages = {144},
     doi = {10.3847/1538-4357/ae1005}
   }

   @software{microjax_zenodo_17247892,
     author = {Miyazaki, Shota},
     title = {microJAX},
     year = {2025},
     publisher = {Zenodo},
     doi = {10.5281/zenodo.17247892},
     url = {https://doi.org/10.5281/zenodo.17247892}
   }

Reproducible numerical results
------------------------------

Record the following with a published result or benchmark:

- the microJAX version or Git commit;
- the JAX and JAXLIB versions;
- CPU or accelerator model and x64 setting;
- selected backend and source profile;
- lens, source, and trajectory parameters;
- applicable ``BinaryMagConfig`` or ``TripleMagConfig`` values;
- whether timing includes compilation;
- the independent reference implementation, its version, and its tolerances.

CPU callers using ``return_info=True`` should retain status values alongside
magnifications. Store microJAX failures separately from failures or
non-convergence in an external reference solver.
