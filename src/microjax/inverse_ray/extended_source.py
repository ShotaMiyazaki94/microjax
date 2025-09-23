"""Inverse-ray finite-source magnification integrators.

This module provides polar-grid, inverse-ray integrators for extended sources
subject to gravitational microlensing by multiple lenses. Two brightness
profiles are implemented:

- ``mag_uniform``: uniform surface brightness disk.
- ``mag_limb_dark``: linear limb-darkened disk.

The integration domain in image space is seeded using the mapped source limb
(`calc_source_limb`) and partitioned into radial/azimuthal regions
(`define_regions`) to concentrate samples near caustics. Boundary crossings in
angle are handled by a numerically stable 4-point cubic interpolation
(`cubic_interp`), with optional smoothing factors to reduce aliasing at the
source limb.

Notes
- Both routines shift coordinates to the center-of-mass frame for improved
  stability and consistency with point-source helpers.
- ``lax.scan`` over regions reduces peak memory usage relative to a full
  vectorized ``vmap`` across all regions.
- Increase ``bins_*`` and resolutions when approaching caustics or for larger
  sources to improve accuracy (with corresponding runtime/memory costs).
"""

import jax 
import jax.numpy as jnp
from jax import jit, lax, vmap, custom_jvp
from functools import partial
from microjax.point_source import lens_eq, _images_point_source
from microjax.inverse_ray.merge_area import calc_source_limb, define_regions
from microjax.inverse_ray.limb_darkening import Is_limb_1st
from microjax.inverse_ray.boundary import in_source, distance_from_source, calc_facB
from typing import Mapping, Sequence, Tuple, Callable, Optional, Union

# Simple alias for readability in type hints
Array = jnp.ndarray

#@partial(jit, static_argnames=("nlenses", "cubic", "r_resolution", "th_resolution", "Nlimb", "u1",
#                               "offset_r", "offset_th", "delta_c"))
def mag_limb_dark(
    w_center: complex,
    rho: float,
    nlenses: int = 2,
    u1: float = 0.0,
    r_resolution: int = 500,
    th_resolution: int = 500,
    Nlimb: int = 500,
    bins_r: int = 50,
    bins_th: int = 120,
    margin_r: float = 0.5,
    margin_th: float = 0.5,
    delta_c: float = 0.01,
    **_params: float,
) -> Array:
    """Compute finite-source magnification with linear limb darkening.

    This routine evaluates the magnification of a circular source centered at
    `w_center` with radius `rho` using a polar grid integration in image space
    and an inverse ray approach. It supports binary (``nlenses=2``) and a
    specific triple-lens configuration (``nlenses=3``). The surface brightness
    profile on the source is linear limb-darkened:

        I(r)/I0 = 1 - u1 * (1 - sqrt(1 - (r/rho)^2)),

    implemented via ``Is_limb_1st``. The image-space region to integrate is
    constructed from points on the mapped source limb using
    ``calc_source_limb`` and partitioned by ``define_regions`` for better
    conditioning around caustics. Angular boundary crossings are located with a
    stable 4-point cubic (Lagrange) interpolant (``cubic_interp``). A smooth
    transition factor ``calc_facB`` controlled by ``delta_c`` reduces aliasing
    at the limb.

    Parameters
    - w_center: complex – Source center in lens plane units (Einstein radius).
    - rho: float – Source radius (same units as `w_center`).
    - nlenses: int – Number of lenses (2 supported; 3 supported for provided
      params).
    - u1: float – Linear limb-darkening coefficient in [0, 1].
    - r_resolution: int – Number of radial samples per region.
    - th_resolution: int – Number of angular samples per region.
    - Nlimb: int – Number of samples on the source limb used to seed regions.
    - bins_r: int – Number of radial bins to split regions.
    - bins_th: int – Number of angular bins to split regions.
    - margin_r: float – Extra radial margin added to each bin (in units of
      `rho`).
    - margin_th: float – Extra angular margin added to each bin (radians).
    - delta_c: float – Smoothing scale for boundary contribution factor
      ``calc_facB``; smaller sharpens boundary, larger smooths.
    - **_params: Mapping of lens parameters depending on ``nlenses``.
      For nlenses=2 expect ``q`` and ``s``; for nlenses=3 expect ``s``, ``q``,
      ``q3``, ``r3``, ``psi``.

    Returns
    - magnification: float – Limb-darkened finite-source magnification.

    Notes
    - Coordinates are internally shifted by the center-of-mass offset for
      numerical stability and consistency with point-source helpers.
    - The result is normalized by ``rho**2`` (no factor of π) because the limb
      darkening weights are included explicitly in the integrand.
    - For large ``rho`` or near-caustic configurations, increase ``bins_*`` and
      ``*_resolution`` to improve accuracy at the cost of memory/runtime.
    """
    if nlenses == 2:
        q, s = _params["q"], _params["s"]
        a  = 0.5 * s
        e1 = q / (1.0 + q)
        _params = {"q": q, "s": s, "a": a, "e1": e1}
    elif nlenses == 3:
        s, q, q3, r3, psi = _params["s"], _params["q"], _params["q3"], _params["r3"], _params["psi"]
        a = 0.5 * s
        total_mass = 1.0 + q + q3
        e1 = q / total_mass
        e2 = 1.0 / total_mass 
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2, "q": q, "s": s, "q3": q3, "psi": psi}
    
    shifted = 0.5 * s * (1 - q) / (1 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params)
    r_scan, th_scan = define_regions(image_limb, mask_limb, rho, bins_r=bins_r, bins_th=bins_th, 
                                     margin_r=margin_r, margin_th=margin_th, nlenses=nlenses)

    def _process_r(r0: float, th_values: Array) -> Array:
        """Integrand over angle for a fixed radius.

        Computes the limb-darkened contribution for radius ``r0`` by sampling
        angles ``th_values``. It classifies samples as inside/outside the source
        disk via ``in_source(distance_from_source(...))`` and adds smoothed
        boundary terms using ``cubic_interp`` and ``calc_facB``.

        Returns the summed area contribution (not yet multiplied by ``dr``).
        """
        dth = (th_values[1] - th_values[0])
        distances = distance_from_source(r0, th_values, w_center_shifted, shifted, nlenses=nlenses, **_params)
        in_num = in_source(distances, rho)
        Is     = Is_limb_1st(distances / rho, u1=u1)
        zero_term = 1e-10
        in0_num, in1_num, in2_num, in3_num, in4_num = in_num[:-4], in_num[1:-3], in_num[2:-2], in_num[3:-1], in_num[4:]
        d0, d1, d2, d3, d4 = distances[:-4], distances[1:-3], distances[2:-2], distances[3:-1], distances[4:]
        th0, th1, th2, th3 = jnp.arange(4)
        num_inside  = in1_num * in2_num * in3_num
        num_B1      = (1.0 - in1_num) * in2_num * in3_num
        num_B2      = in1_num * in2_num * (1.0 - in3_num)
        th_est_B1   = cubic_interp(rho, d0, d1, d2, d3, th0, th1, th2, th3, epsilon=zero_term)
        th_est_B2   = cubic_interp(rho, d1, d2, d3, d4, th0, th1, th2, th3, epsilon=zero_term)
        delta_B1    = jnp.clip(th2 - th_est_B1, 0.0, 1.0) + zero_term
        delta_B2    = jnp.clip(th_est_B2 - th1, 0.0, 1.0) + zero_term
        fac_B1 = calc_facB(delta_B1, delta_c)
        fac_B2 = calc_facB(delta_B2, delta_c)
        area_inside = r0 * dth * Is[2:-2] * num_inside
        area_B1     = r0 * dth * Is[2:-2] * fac_B1 * num_B1
        area_B2     = r0 * dth * Is[2:-2] * fac_B2 * num_B2
        return jnp.sum(area_inside + area_B1 + area_B2)

    #@jax.checkpoint 
    def _compute_for_range(r_range: Array, th_range: Array) -> Array:
        """Integrate over a rectangular image-space subregion.

        - r_range: length-2 array giving [r_min, r_max].
        - th_range: length-2 array giving [theta_min, theta_max].

        Builds uniform 1D grids of sizes ``r_resolution`` and ``th_resolution``
        and performs a rectangle-rule accumulation over radius with per-radius
        angular sums from ``_process_r``. Returns the total area contribution
        of this subregion.
        """
        r_values = jnp.linspace(r_range[0], r_range[1], r_resolution, endpoint=True)
        th_values = jnp.linspace(th_range[0], th_range[1], th_resolution, endpoint=True)
        area_r = vmap(lambda r: _process_r(r, th_values))(r_values)
        dr = r_values[1] - r_values[0]
        total_area = dr * jnp.sum(area_r) # trapezoidal integration
        return total_area
    
    inputs = (r_scan, th_scan)
    if(1): # memory efficient but seems complex implementation for jax.checkpoint.
        #@jax.checkpoint
        def scan_images(carry, inputs):
            r_range, th_range = inputs
            total_area = _compute_for_range(r_range, th_range)
            return carry + total_area, None
        magnification_unnorm, _ = lax.scan(scan_images, 0.0, inputs, unroll=1)
    if(0): # vmap case. subtle improvement in speed but worse in memory. More careful for chunking size.
        total_areas = vmap(_compute_for_range, in_axes=(0, 0))(r_scan, th_scan)
        magnification_unnorm = jnp.sum(total_areas)
    magnification = magnification_unnorm / rho**2 
    return magnification 

#@partial(jit, static_argnames=("nlenses", "r_resolution", "th_resolution", "Nlimb", "offset_r", "offset_th", "cubic",))
def mag_uniform(
    w_center: complex,
    rho: float,
    nlenses: int = 2,
    r_resolution: int = 500,
    th_resolution: int = 500,
    Nlimb: int = 500,
    bins_r: int = 50,
    bins_th: int = 120,
    margin_r: float = 0.5,
    margin_th: float = 0.5,
    **_params: float,
) -> Array:
    """Compute finite-source magnification for a uniform-brightness disk.

    Uses the same region construction and polar-grid integration strategy as
    ``mag_limb_dark`` but with a uniform surface brightness profile. The
    integrand is the area fraction inside the source with sub-cell angular
    crossing handled by a stable cubic interpolation in angle.

    Parameters
    - w_center: complex – Source center in Einstein-radius units.
    - rho: float – Source radius.
    - nlenses: int – Number of lenses (2 supported; 3 supported for provided
      params).
    - r_resolution: int – Number of radial samples per region.
    - th_resolution: int – Number of angular samples per region.
    - Nlimb: int – Number of samples on the source limb used to seed regions.
    - bins_r: int – Number of radial bins for region partitioning.
    - bins_th: int – Number of angular bins for region partitioning.
    - margin_r: float – Extra radial margin per bin (in units of ``rho``).
    - margin_th: float – Extra angular margin per bin (radians).
    - **_params: Lens parameters depending on ``nlenses`` (same as in
      ``mag_limb_dark``).

    Returns
    - magnification: float – Uniform finite-source magnification normalized by
      ``rho**2 * pi``.

    Notes
    - Internally shifts coordinates by the lens center-of-mass offset.
    - Region-wise ``lax.scan`` is default for better peak-memory behavior; a
      fully vectorized alternative via ``vmap`` is available but uses more
      memory.
    - Sensitivity near caustics can be improved by increasing ``bins_*`` and
      ``*_resolution`` or broadening ``margin_*``.
    """
    
    if nlenses == 2:
        q, s = _params["q"], _params["s"]
        a  = 0.5 * s
        e1 = q / (1.0 + q)
        _params = {"q": q, "s": s, "a": a, "e1": e1}
    elif nlenses == 3:
        s, q, q3, r3, psi = _params["s"], _params["q"], _params["q3"], _params["r3"], _params["psi"]
        a = 0.5 * s
        total_mass = 1.0 + q + q3
        e1 = q / total_mass
        e2 = 1.0 / total_mass 
        _params = {"a": a, "r3": r3, "e1": e1, "e2": e2, "q": q, "s": s, "q3": q3, "psi": psi}
    
    shifted = a * (1.0 - q) / (1.0 + q)  
    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params, nlenses=nlenses)
    r_scan, th_scan = define_regions(image_limb, mask_limb, rho, bins_r=bins_r, bins_th=bins_th, 
                                     margin_r=margin_r, margin_th=margin_th, nlenses=nlenses)

    #@jax.checkpoint 
    def _process_r(r0: float, th_values: Array) -> Array:
        """Angular accumulation at fixed radius for a uniform source.

        Classifies points as inside/outside the source and corrects the two
        nearest angular cells that cross the source limb using a cubic estimate
        of the crossing angle. Returns the summed (angular) area at radius
        ``r0`` (prior to multiplying by ``dr``).
        """
        dth = (th_values[1] - th_values[0])
        distances = distance_from_source(r0, th_values, w_center_shifted, shifted, nlenses=nlenses, **_params)
        in_num = in_source(distances, rho)
        zero_term = 1e-10
        in0_num, in1_num, in2_num, in3_num = in_num[:-3], in_num[1:-2], in_num[2:-1], in_num[3:]
        d0, d1, d2, d3 = distances[:-3], distances[1:-2], distances[2:-1], distances[3:]
        th0, th1, th2, th3 = jnp.arange(4)
        num_inside  = in1_num * in2_num
        num_in2out  = in1_num * (1.0 - in2_num)
        num_out2in  = (1.0 - in1_num) * in2_num
        th_est      = cubic_interp(rho, d0, d1, d2, d3, th0, th1, th2, th3, epsilon=zero_term)
        frac_in2out = jnp.clip((th_est - th1), 0.0, 1.0)
        frac_out2in = jnp.clip((th2 - th_est), 0.0, 1.0)
        area_inside = r0 * dth * num_inside
        area_crossing = r0 * dth * (num_in2out * frac_in2out + num_out2in * frac_out2in)
        return jnp.sum(area_inside + area_crossing)  

    #@jax.checkpoint
    def _compute_for_range(r_range: Array, th_range: Array) -> Array:
        """Integrate over a given ``(r, theta)`` rectangle using uniform grids.

        Returns the area contribution of the subregion via a rectangle-rule
        sum across the per-radius angular integrals from ``_process_r``.
        """
        r_values = jnp.linspace(r_range[0], r_range[1], r_resolution, endpoint=True)
        th_values = jnp.linspace(th_range[0], th_range[1], th_resolution, endpoint=True)
        #area_r = jax.checkpoint(vmap(lambda r: _process_r(r, th_values, cubic)))(r_values)
        area_r = vmap(lambda r: _process_r(r, th_values))(r_values)
        dr = r_values[1] - r_values[0]
        total_area = dr * jnp.sum(area_r) # trapezoidal integration
        return total_area
    
    #_process_r = jax.checkpoint(_process_r, prevent_cse=True)
    #_compute_for_range = jax.checkpoint(_compute_for_range, prevent_cse=True)
    
    inputs = (r_scan, th_scan)
    if(1): # memory efficient but seems complex implementation for jax.checkpoint.
        def scan_images(carry, inputs):
            r_range, th_range = inputs
            total_area = _compute_for_range(r_range, th_range)
            #total_area = _compute_for_range(r_range, th_range, cubic=cubic)
            return carry + total_area, None
        magnification_unnorm, _ = lax.scan(jax.checkpoint(scan_images), 0.0, inputs, unroll=1)
    if(0): # vmap case. subtle improvement in speed but worse in memory. More careful for chunking size.
        total_areas = vmap(_compute_for_range, in_axes=(0, 0))(r_scan, th_scan)
        magnification_unnorm = jnp.sum(total_areas)
    
    magnification = magnification_unnorm / rho**2 / jnp.pi
    return magnification 

def cubic_interp(
    x: Union[float, Array],
    x0: Union[float, Array],
    x1: Union[float, Array],
    x2: Union[float, Array],
    x3: Union[float, Array],
    y0: Union[float, Array],
    y1: Union[float, Array],
    y2: Union[float, Array],
    y3: Union[float, Array],
    epsilon: float = 1e-12,
) -> Union[float, Array]:
    """Stable 4-point cubic (Lagrange) interpolation with scaling.

    Evaluates the cubic interpolant passing through the four points
    ``(xk, yk)`` for ``k=0..3`` at position ``x``. To improve numerical
    stability when the abscissas are nearly collinear or clustered, the
    abscissa domain is rescaled to ``[0, 1]`` before computing the Lagrange
    basis. Small ``epsilon`` terms guard against division by zero in degenerate
    configurations.

    Parameters
    - x: float/array – Evaluation abscissa.
    - x0, x1, x2, x3: float/array – Sample abscissas.
    - y0, y1, y2, y3: float/array – Sample ordinates corresponding to each
      abscissa.
    - epsilon: float – Small positive value to avoid division by zero in the
      basis denominators.

    Returns
    - y: float/array – Interpolated value at ``x``.

    Notes
    - This function is used to estimate the angular crossing location of the
      source limb within a four-cell angular stencil.
    - For monotonic constraints or fewer samples consider alternative schemes.
    """
    # Implemented algebraically; faster and more memory-efficient than a
    # matrix-based polyfit for JAX transformations.
    x_min = jnp.min(jnp.array([x0, x1, x2, x3]))
    x_max = jnp.max(jnp.array([x0, x1, x2, x3]))
    scale = jnp.maximum(x_max - x_min, epsilon)
    x_hat = (x - x_min) / scale
    x0_hat, x1_hat, x2_hat, x3_hat = (x0 - x_min) / scale, (x1 - x_min) / scale, (x2 - x_min) / scale, (x3 - x_min) / scale
    L0 = ((x_hat - x1_hat) * (x_hat - x2_hat) * (x_hat - x3_hat)) / \
        ((x0_hat - x1_hat + epsilon) * (x0_hat - x2_hat + epsilon) * (x0_hat - x3_hat + epsilon))
    L1 = ((x_hat - x0_hat) * (x_hat - x2_hat) * (x_hat - x3_hat)) / \
        ((x1_hat - x0_hat + epsilon) * (x1_hat - x2_hat + epsilon) * (x1_hat - x3_hat + epsilon))
    L2 = ((x_hat - x0_hat) * (x_hat - x1_hat) * (x_hat - x3_hat)) / \
        ((x2_hat - x0_hat + epsilon) * (x2_hat - x1_hat + epsilon) * (x2_hat - x3_hat + epsilon))
    L3 = ((x_hat - x0_hat) * (x_hat - x1_hat) * (x_hat - x2_hat)) / \
        ((x3_hat - x0_hat + epsilon) * (x3_hat - x1_hat + epsilon) * (x3_hat - x2_hat + epsilon))
    return y0 * L0 + y1 * L1 + y2 * L2 + y3 * L3
