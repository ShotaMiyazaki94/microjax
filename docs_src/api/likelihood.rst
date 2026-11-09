Flux fitting and likelihoods
============================

Once you have a magnification curve ``A``, ``linear_chi2(A, flux, sigma)``
fits the source and blend fluxes for independent observational errors.
The likelihood functions instead marginalize over those two fluxes with
Gaussian priors. They accept a design matrix with columns ``[A, 1]``.

.. autofunction:: microjax.likelihood.linear_chi2

.. autofunction:: microjax.likelihood.nll_ulens

.. autofunction:: microjax.likelihood.nll_ulens_general
