"""Fixed-shape angular and radial quadrature kernels.

``angular`` integrates a brightness profile between validated angular roots.
``radial`` integrates the resulting ring measure across topology intervals and
can attach one chart parameter to every interval. ``rules`` contains only the
immutable numerical nodes and weights.

The kernels consume geometry and root results; they do not discover images or
choose between the multipole and boundary solvers.
"""
