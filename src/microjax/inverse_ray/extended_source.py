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

if __name__ == "__main__":
    import time
    jax.config.update("jax_enable_x64", True)
    #jax.config.update("jax_debug_nans", True)
    q = 0.05
    s = 1.0
    alpha = jnp.deg2rad(10) 
    tE = 30 
    t0 = 0.0 
    u0 = 0.0 
    rho = 0.06

    nlenses = 2
    a = 0.5 * s
    e1 = q / (1.0 + q)
    _params = {"a": a, "e1": e1}
    x_cm = a * (1.0 - q) / (1.0 + q)

    num_points = 2000
    t  =  jnp.linspace(-1.0*tE, 1.0*tE, num_points)
    tau = (t - t0)/tE
    y1 = -u0*jnp.sin(alpha) + tau*jnp.cos(alpha)
    y2 = u0*jnp.cos(alpha) + tau*jnp.sin(alpha) 
    w_points = jnp.array(y1 + y2 * 1j, dtype=complex)
    test_params = {"q": q, "s": s}  # Lens parameters

    Nlimb = 500
    r_resolution  = 500
    th_resolution = 1000
    cubic = True

    bins_r = 50
    bins_th = 120
    margin_r = 1.0
    margin_th= 1.0

    from microjax.caustics.extended_source import mag_extended_source
    import MulensModel as mm
    def mag_vbbl(w0: complex, rho: float, u1: float = 0.0, accuracy: float = 1e-4) -> Array:
        """Reference magnification via VBBinaryLensing for benchmarking.

        Parameters
        - w0: complex – Source center.
        - rho: float – Source radius.
        - u1: float – Linear limb-darkening coefficient passed to VBBL.
        - accuracy: float – Target accuracy for VBBL integration.

        Returns
        - magnification: float – VBBL magnification for comparison plots.
        """
        a  = 0.5 * s
        e1 = 1.0 / (1.0 + q)
        e2 = 1.0 - e1  
        bl = mm.BinaryLens(e1, e2, 2*a)
        return bl.vbbl_magnification(w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1)
    #magn  = lambda w: mag_uniform(w, rho, r_resolution=2000, th_resolution=1000, **test_params, cubic=True)
    @jit
    def mag_mj(w: complex) -> Array:
        """JIT-wrapped microjax uniform finite-source magnification."""
        return mag_uniform(w, rho, s=s, q=q, Nlimb=Nlimb, bins_r=bins_r, bins_th=bins_th,
                           r_resolution=r_resolution, th_resolution=th_resolution, 
                           margin_r = margin_r, margin_th=margin_th, cubic=cubic)
    def chunked_vmap(func: Callable[[complex], Array], data: Array, chunk_size: int) -> Array:
        """Apply ``vmap`` in chunks to reduce peak memory usage."""
        results = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            results.append(jax.vmap(func)(chunk))
        return jnp.concatenate(results)

    magn2  = lambda w0: jnp.array([mag_vbbl(w, rho) for w in w0])
    #magn2 =  jit(vmap(magn2, in_axes=(0,)))
    #magn =  jit(vmap(magn, in_axes=(0,)))

    #_ = magn(w_points).block_until_ready()
    @jax.jit
    def scan_mag_mj(w_points: Array) -> Array:
        """Compute magnifications via ``lax.scan`` over input points."""
        def body_fun(carry, w):
            result = mag_mj(w)
            return carry, result
        _, results = lax.scan(body_fun, None, w_points)
        return results

    print("number of data points: %d"%(num_points))
    from microjax.point_source import mag_point_source, critical_and_caustic_curves
    mag_point_source(w_points, s=s, q=q)
    start = time.time()
    mags_poi = mag_point_source(w_points, s=s, q=q)
    mags_poi.block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for point-source in microjax"%(end-start, 1000*(end - start)/num_points)) 

    from microjax.multipole import _mag_hexadecapole
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    _, _ = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params) 
    start = time.time()
    z, z_mask = _images_point_source(w_points - x_cm, nlenses=nlenses, **_params)
    mu_multi, delta_mu_multi = _mag_hexadecapole(z, z_mask, rho, nlenses=nlenses, **_params)
    mu_multi.block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for hexadecapole in microjax"%(end-start, 1000*(end - start)/num_points)) 

    start = time.time()
    magnifications2 = magn2(w_points)
    magnifications2.block_until_ready() 
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for VBBinaryLensing"%(end - start,1000*(end - start)/num_points))

    chunk_size = 500 
    _ = chunked_vmap(mag_mj, w_points, chunk_size).block_until_ready()
    print("start computation with mag_lc_uniform, %d chunk_size, %d rbin, %d thbin"%(chunk_size, r_resolution, th_resolution))
    print("start computation with vmap")
    start = time.time()
    #magnifications = mag_uniform(w_points, rho, s=s, q=q, Nlimb=2000, r_resolution=r_resolution, th_resolution=th_resolution).block_until_ready()
    magnifications = chunked_vmap(mag_mj, w_points, chunk_size).block_until_ready()
    end = time.time()
    print("computation time: %.3f sec (%.3f ms per points) for vmap in microjax"%(end-start, 1000*(end - start)/num_points))
    
    if(0):
        print("start computation with lax.scan")
        start = time.time()
        magnifications = scan_mag_mj(w_points).block_until_ready()
        end = time.time()
        print("computation time: %.3f sec (%.3f ms per points) for lax.scan in microjax"%(end-start, 1000*(end - start)/num_points))
    #print("computation time: %.3f ms per points for lax.scan in microjax" % (1000 * (end - start) / num_points))
    
   
    # Print out the result
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.ticker as ticker
    import seaborn as sns
    sns.set_theme(style="ticks")

    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, npts=100, s=s, q=q)

    fig, ax_ = plt.subplots(2,1,figsize=(8,6), sharex=True, gridspec_kw=dict(hspace=0.1, height_ratios=[4,1]))
    ax  = ax_[0]
    ax1 = ax_[1]
    ax_in = inset_axes(ax,
        width="60%", height="60%", 
        bbox_transform=ax.transAxes,
        bbox_to_anchor=(0.35, 0.35, 0.6, 0.6)
    )
    ax_in.set_aspect(1)
    ax_in.set(xlabel="$\mathrm{Re}(w)$", ylabel="$\mathrm{Im}(w)$")
    for cc in caustic_curves:
        ax_in.plot(cc.real, cc.imag, color='red', lw=0.7)
    circles = [plt.Circle((xi,yi), radius=rho, fill=False, facecolor=None, ec="blue", zorder=2) 
               for xi, yi in zip(w_points.real, w_points.imag)
               ]
    c = mpl.collections.PatchCollection(circles, match_original=True, alpha=0.5)
    ax_in.add_collection(c)
    ax_in.set_aspect(1)
    ax_in.set(xlim=(-1., 1.2), ylim=(-1.0, 1.))
    ax_in.plot(-q/(1+q) * s, 0 , ".",c="k")
    ax_in.plot((1.0)/(1+q) * s, 0 ,".",c="k")

    ax.plot(t, magnifications, ".", label="microjax", zorder=1)
    ax.plot(t, magnifications2, "-", label="VBBinaryLensing", zorder=2)
    ylim = ax.get_ylim()
    #ax.plot(t, mags_poi, "--", label="point-source", zorder=-1, color="gray")
    ax.set_title("mag_uniform")
    ax.grid(ls=":")
    ax.set_ylabel("magnification")
    ax.plot(t, mags_poi, "--", label="point_source", zorder=-1, color="gray")
    ax.plot(t, mu_multi, ":", label="hexadecapole", zorder=-2, color="orange")
    ax.set_ylim(ylim[0], ylim[1])
    ax1.plot(t, jnp.abs(magnifications - magnifications2)/magnifications2, "-", ms=1)
    #ax1.plot(t, jnp.abs(mags_poi - magnifications2)/magnifications2, "-", ms=1)
    #ax1.plot(t, jnp.abs(mu_multi - magnifications2)/magnifications2, "-", ms=1)
    ax1.grid(ls=":")
    ax1.set_yticks(10**jnp.arange(-4, -2, 1))
    ax1.set_ylabel("relative diff")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0, 10**-2, 10**-4, 10**-6], numticks=10))
    ax1.set_ylim(1e-6, 1e-2)
    ax.legend(loc="upper left")
    ax1.set_xlabel("time (days)")
    plt.show()
    #plt.savefig("z_fig/mag_uniform.png", bbox_inches="tight", dpi=300)
    #plt.close()

    if(1):
        diff = jnp.abs(magnifications - magnifications2)/magnifications2
        label = diff > 1e-3
        #for i, (r, th) in enumerate(zip(w_points, diff)):
        for i, (r, th) in enumerate(zip(w_points[label], diff[label])):
            print(i, "%.5f"%(th), r)
        #print("errnous \n", w_points[label], diff[label])
