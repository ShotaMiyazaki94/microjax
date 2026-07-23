"""Shared fixed-node quadrature rules for boundary integration."""

from __future__ import annotations

import numpy as np


def _symmetric(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values[:-1], values[::-1]))


_GK31_X_POSITIVE_DESC = np.asarray(
    [
        0.9980022986933971,
        0.9879925180204854,
        0.9677390756791391,
        0.9372733924007059,
        0.8972645323440819,
        0.8482065834104272,
        0.7904185014424659,
        0.7244177313601701,
        0.650996741297417,
        0.5709721726085388,
        0.4850818636402397,
        0.3941513470775634,
        0.2991800071531688,
        0.2011940939974345,
        0.1011420669187175,
        0.0,
    ]
)
_GK31_W_POSITIVE_DESC = np.asarray(
    [
        0.005377479872923349,
        0.015007947329316123,
        0.02546084732671532,
        0.035346360791375846,
        0.04458975132476488,
        0.05348152469092809,
        0.06200956780067064,
        0.06985412131872826,
        0.07684968075772038,
        0.08308050282313302,
        0.08856444305621177,
        0.09312659817082532,
        0.09664272698362368,
        0.09917359872179196,
        0.1007698455238756,
        0.10133000701479155,
    ]
)
_G15_W_POSITIVE_DESC = np.asarray(
    [
        0.0,
        0.03075324199611727,
        0.0,
        0.07036604748810812,
        0.0,
        0.10715922046717194,
        0.0,
        0.13957067792615432,
        0.0,
        0.16626920581699394,
        0.0,
        0.1861610000155622,
        0.0,
        0.19843148532711158,
        0.0,
        0.20257824192556127,
    ]
)

GK31_X = np.concatenate(
    (-_GK31_X_POSITIVE_DESC[:-1], _GK31_X_POSITIVE_DESC[::-1])
)
GK31_W = _symmetric(_GK31_W_POSITIVE_DESC)
G15_W_ON_GK31 = _symmetric(_G15_W_POSITIVE_DESC)

# Independent one-cell rules used by the fast binary path.  Their 70 total
# nodes remain far cheaper than uniform radial subdivision while resolving
# sharper interior structure than the embedded G15/K31 pair.
GL23_X, GL23_W = np.polynomial.legendre.leggauss(23)
GL47_X, GL47_W = np.polynomial.legendre.leggauss(47)
