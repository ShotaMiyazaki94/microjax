"""Exact angular source-boundary representation and root solving.

``level_set`` constructs the finite Fourier series for one polar ring.
``angular`` converts that series to a fixed-degree self-inversive polynomial,
solves and polishes its unit-circle roots, then returns inside-angle intervals.

No radial integration or source-profile quadrature is performed here.
"""
