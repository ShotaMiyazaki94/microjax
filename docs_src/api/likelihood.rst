Flux fitting and likelihoods
============================

Once you have a magnification curve ``A``, ``linear_chi2(A, flux, sigma)``
fits the source and blend fluxes for independent observational errors.
The likelihood functions integrate over the possible values of both fluxes
with Gaussian priors, accounting for their uncertainty. This is analytic
marginalization; see :doc:`../inference` for a worked example. Their input
is a two-column matrix: the magnifications ``A`` and a column of ones.

.. autofunction:: microjax.likelihood.linear_chi2

.. autofunction:: microjax.likelihood.nll_ulens

.. autofunction:: microjax.likelihood.nll_ulens_general
