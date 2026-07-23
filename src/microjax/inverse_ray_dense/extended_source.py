"""Legacy dense polar-grid finite-source magnification integrators.

The backend samples rectangular polar grids and scans image regions
sequentially, keeping the large ``(region, radius, theta)`` tensor within the
configured memory budget.

All implementations use lens-centre-of-mass coordinates consistently with the
point-source utilities.
"""

import jax
import jax.numpy as jnp
from jax import lax, vmap
from microjax.lens_geometry import triple_lens_geometry
from .merge_area import calc_source_limb, define_regions
from .limb_darkening import Is_limb_1st
from .boundary import in_source, distance_from_source, calc_facB
from typing import Mapping, Union

# Simple alias for readability in type hints
Array = jnp.ndarray
def _dense_grid_dtypes(w_center: complex, grid_fp32: bool) -> tuple[jnp.dtype, jnp.dtype]:
    """Return real/complex dtypes used by the dense inverse-ray grid."""
    if grid_fp32:
        return jnp.float32, jnp.complex64

    real_dtype = jnp.asarray(w_center).real.dtype
    complex_dtype = jnp.complex64 if real_dtype == jnp.float32 else jnp.complex128
    return real_dtype, complex_dtype


def _cast_grid_params(params: Mapping[str, float], real_dtype: jnp.dtype) -> dict[str, Array]:
    """Cast lens parameters once for dense-grid evaluation."""
    return {key: jnp.asarray(value, dtype=real_dtype) for key, value in params.items()}

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
    grid_fp32: bool = False,
    **_params: float,
) -> Array:
    """Finite-source magnification with linear limb darkening.

    Parameters
    ----------
    w_center : complex
        Source-centre position in Einstein-radius units.
    rho : float
        Angular source radius in the same units as ``w_center``.
    nlenses : int, optional
        Number of lenses in the system. Only binary (``2``) and the supported
        triple-lens (``3``) configuration are handled.
    u1 : float, optional
        Linear limb-darkening coefficient in ``[0, 1]``.
    r_resolution : int, optional
        Number of radial samples per image-space region.
    th_resolution : int, optional
        Number of azimuthal samples per image-space region.
    Nlimb : int, optional
        Number of source-limb samples used to seed the image-plane regions.
    bins_r : int, optional
        Radial bin count for region partitioning.
    bins_th : int, optional
        Azimuthal bin count for region partitioning.
    margin_r : float, optional
        Additional radial margin (in units of ``rho``) attached to each bin.
    margin_th : float, optional
        Additional azimuthal margin (in radians) attached to each bin.
    delta_c : float, optional
        Smoothing scale supplied to ``calc_facB`` when blending limb crossings.
    grid_fp32 : bool, optional
        When ``True``, cast the dense polar integration grid and lens-evaluation
        path to ``float32``/``complex64``. Limb tracing and region construction
        stay in the ambient precision.
    **_params : float
        Lens parameters forwarded to ``calc_source_limb`` and the underlying
        lens-equation helpers. For a binary lens provide ``s`` (separation) and
        ``q`` (mass ratio); for the triple-lens branch also provide ``q3``,
        ``r3``, and ``psi``.

    Returns
    -------
    Array
        Limb-darkened magnification normalised by ``rho**2``. The value is
        returned as a scalar ``jax.numpy`` array.

    Notes
    -----
    - ``calc_source_limb`` and ``define_regions`` build the polar integration
      mesh from the mapped source limb.
    - ``cubic_interp`` locates angular boundary crossings and ``calc_facB``
      reduces aliasing at the source edge.
    - Increase ``r_resolution``, ``th_resolution``, or the bin counts near
      caustics or when modelling large sources. Triple-lens usage remains
      experimental and mirrors the binary parameterisation.
    """
    if nlenses == 2:
        q, s = _params["q"], _params["s"]
        a  = 0.5 * s
        e1 = q / (1.0 + q)
        _params = {"q": q, "s": s, "a": a, "e1": e1}
        shifted = a * (1.0 - q) / (1.0 + q)
    elif nlenses == 3:
        s, q, q3, r3, psi = _params["s"], _params["q"], _params["q3"], _params["r3"], _params["psi"]
        geometry = triple_lens_geometry(s, q, q3, r3, psi)
        shifted = geometry.shifted
        _params = {
            "a": geometry.a,
            "r3": r3,
            "e1": geometry.e1,
            "e2": geometry.e2,
            "q": q,
            "s": s,
            "q3": q3,
            "psi": psi,
        }

    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(
        w_center, rho, Nlimb, nlenses=nlenses, **_params
    )
    r_scan, th_scan = define_regions(image_limb, mask_limb, rho, bins_r=bins_r, bins_th=bins_th,
                                     margin_r=margin_r, margin_th=margin_th, nlenses=nlenses)

    grid_real_dtype, grid_complex_dtype = _dense_grid_dtypes(
        w_center, grid_fp32
    )
    grid_params = _cast_grid_params(_params, grid_real_dtype)
    rho_grid = jnp.asarray(rho, dtype=grid_real_dtype)
    u1_grid = jnp.asarray(u1, dtype=grid_real_dtype)
    delta_c_grid = jnp.asarray(delta_c, dtype=grid_real_dtype)
    shifted_grid = jnp.asarray(shifted, dtype=grid_complex_dtype)
    w_center_shifted_grid = jnp.asarray(
        w_center_shifted, dtype=grid_complex_dtype
    )
    output_dtype = jnp.asarray(w_center).real.dtype

    def _process_r(r0: float, th_values: Array) -> Array:
        """Integrand over angle for a fixed radius.

        Computes the limb-darkened contribution for radius ``r0`` by sampling
        angles ``th_values``. It classifies samples as inside/outside the source
        disk via ``in_source(distance_from_source(...))`` and adds smoothed
        boundary terms using ``cubic_interp`` and ``calc_facB``.

        Returns the summed area contribution (not yet multiplied by ``dr``).
        """
        dth = (th_values[1] - th_values[0])
        distances = distance_from_source(
            jnp.asarray(r0, dtype=grid_real_dtype),
            th_values,
            w_center_shifted_grid,
            shifted_grid,
            nlenses=nlenses,
            **grid_params,
        )
        large_dist = jnp.asarray(1e10, dtype=grid_real_dtype)
        distances = jnp.nan_to_num(distances, nan=large_dist, posinf=large_dist, neginf=large_dist)
        in_num = in_source(distances, rho_grid)
        Is = Is_limb_1st(distances / rho_grid, u1=u1_grid)
        zero_term = jnp.asarray(1e-10, dtype=grid_real_dtype)
        in0_num, in1_num, in2_num, in3_num, in4_num = in_num[:-4], in_num[1:-3], in_num[2:-2], in_num[3:-1], in_num[4:]
        d0, d1, d2, d3, d4 = distances[:-4], distances[1:-3], distances[2:-2], distances[3:-1], distances[4:]
        th0, th1, th2, th3 = jnp.arange(4, dtype=grid_real_dtype)
        num_inside = in1_num * in2_num * in3_num
        num_B1 = (1.0 - in1_num) * in2_num * in3_num
        num_B2 = in1_num * in2_num * (1.0 - in3_num)
        th_est_B1 = cubic_interp(rho_grid, d0, d1, d2, d3, th0, th1, th2, th3, epsilon=zero_term)
        th_est_B2 = cubic_interp(rho_grid, d1, d2, d3, d4, th0, th1, th2, th3, epsilon=zero_term)
        delta_B1 = jnp.clip(th2 - th_est_B1, 0.0, 1.0) + zero_term
        delta_B2 = jnp.clip(th_est_B2 - th1, 0.0, 1.0) + zero_term
        fac_B1 = calc_facB(delta_B1, delta_c_grid)
        fac_B2 = calc_facB(delta_B2, delta_c_grid)
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
        r_values = jnp.linspace(
            jnp.asarray(r_range[0], dtype=grid_real_dtype),
            jnp.asarray(r_range[1], dtype=grid_real_dtype),
            r_resolution,
            endpoint=True,
        )
        th_values = jnp.linspace(
            jnp.asarray(th_range[0], dtype=grid_real_dtype),
            jnp.asarray(th_range[1], dtype=grid_real_dtype),
            th_resolution,
            endpoint=True,
        )
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
    magnification = magnification_unnorm / rho_grid**2
    return jnp.asarray(magnification, dtype=output_dtype)

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
    grid_fp32: bool = False,
    **_params: float,
) -> Array:
    """Finite-source magnification for a uniform-brightness disk.

    Parameters
    ----------
    w_center : complex
        Source-centre position in Einstein-radius units.
    rho : float
        Angular source radius in the same units as ``w_center``.
    nlenses : int, optional
        Number of lenses in the system. Only binary (``2``) and the supported
        triple-lens (``3``) configuration are handled.
    r_resolution : int, optional
        Number of radial samples per image-space region.
    th_resolution : int, optional
        Number of azimuthal samples per image-space region.
    Nlimb : int, optional
        Number of source-limb samples used to seed the image-plane regions.
    bins_r : int, optional
        Radial bin count for region partitioning.
    bins_th : int, optional
        Azimuthal bin count for region partitioning.
    margin_r : float, optional
        Additional radial margin (in units of ``rho``) attached to each bin.
    margin_th : float, optional
        Additional azimuthal margin (in radians) attached to each bin.
    grid_fp32 : bool, optional
        When ``True``, cast the dense polar integration grid and lens-evaluation
        path to ``float32``/``complex64``. Limb tracing and region construction
        stay in the ambient precision.
    **_params : float
        Lens parameters forwarded to ``calc_source_limb`` and the underlying
        lens-equation helpers. For a binary lens provide ``s`` (separation) and
        ``q`` (mass ratio); for the triple-lens branch also provide ``q3``,
        ``r3``, and ``psi``.

    Returns
    -------
    Array
        Uniform-source magnification normalised by ``rho**2 * pi``. The
        value is returned as a scalar ``jax.numpy`` array.

    Notes
    -----
    - The same region construction as :func:`mag_limb_dark` is used, but with a
      uniform surface-brightness integrand.
    - Angular crossings are identified with ``cubic_interp``.
    - ``lax.scan`` over regions is the default for peak-memory efficiency; a
      vectorised ``vmap`` alternative is retained for experimentation.
    - Increase ``r_resolution``, ``th_resolution``, or the bin counts near
      caustics or when modelling large sources. Triple-lens usage remains
      experimental and mirrors the binary parameterisation.
    """

    if nlenses == 2:
        q, s = _params["q"], _params["s"]
        a  = 0.5 * s
        e1 = q / (1.0 + q)
        _params = {"q": q, "s": s, "a": a, "e1": e1}
        shifted = a * (1.0 - q) / (1.0 + q)
    elif nlenses == 3:
        s, q, q3, r3, psi = _params["s"], _params["q"], _params["q3"], _params["r3"], _params["psi"]
        geometry = triple_lens_geometry(s, q, q3, r3, psi)
        shifted = geometry.shifted
        _params = {
            "a": geometry.a,
            "r3": r3,
            "e1": geometry.e1,
            "e2": geometry.e2,
            "q": q,
            "s": s,
            "q3": q3,
            "psi": psi,
        }

    w_center_shifted = w_center - shifted
    image_limb, mask_limb = calc_source_limb(w_center, rho, Nlimb, **_params, nlenses=nlenses)
    r_scan, th_scan = define_regions(image_limb, mask_limb, rho, bins_r=bins_r, bins_th=bins_th,
                                     margin_r=margin_r, margin_th=margin_th, nlenses=nlenses)

    grid_real_dtype, grid_complex_dtype = _dense_grid_dtypes(
        w_center, grid_fp32
    )
    grid_params = _cast_grid_params(_params, grid_real_dtype)
    rho_grid = jnp.asarray(rho, dtype=grid_real_dtype)
    shifted_grid = jnp.asarray(shifted, dtype=grid_complex_dtype)
    w_center_shifted_grid = jnp.asarray(
        w_center_shifted, dtype=grid_complex_dtype
    )
    output_dtype = jnp.asarray(w_center).real.dtype

    #@jax.checkpoint
    def _process_r(r0: float, th_values: Array) -> Array:
        """Angular accumulation at fixed radius for a uniform source.

        Classifies points as inside/outside the source and corrects the two
        nearest angular cells that cross the source limb using a cubic estimate
        of the crossing angle. Returns the summed (angular) area at radius
        ``r0`` (prior to multiplying by ``dr``).
        """
        dth = (th_values[1] - th_values[0])
        distances = distance_from_source(
            jnp.asarray(r0, dtype=grid_real_dtype),
            th_values,
            w_center_shifted_grid,
            shifted_grid,
            nlenses=nlenses,
            **grid_params,
        )
        large_dist = jnp.asarray(1e10, dtype=grid_real_dtype)
        distances = jnp.nan_to_num(distances, nan=large_dist, posinf=large_dist, neginf=large_dist)
        in_num = in_source(distances, rho_grid)
        zero_term = jnp.asarray(1e-10, dtype=grid_real_dtype)
        in0_num, in1_num, in2_num, in3_num = in_num[:-3], in_num[1:-2], in_num[2:-1], in_num[3:]
        d0, d1, d2, d3 = distances[:-3], distances[1:-2], distances[2:-1], distances[3:]
        th0, th1, th2, th3 = jnp.arange(4, dtype=grid_real_dtype)
        num_inside = in1_num * in2_num
        num_in2out = in1_num * (1.0 - in2_num)
        num_out2in = (1.0 - in1_num) * in2_num
        th_est = cubic_interp(rho_grid, d0, d1, d2, d3, th0, th1, th2, th3, epsilon=zero_term)
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
        r_values = jnp.linspace(
            jnp.asarray(r_range[0], dtype=grid_real_dtype),
            jnp.asarray(r_range[1], dtype=grid_real_dtype),
            r_resolution,
            endpoint=True,
        )
        th_values = jnp.linspace(
            jnp.asarray(th_range[0], dtype=grid_real_dtype),
            jnp.asarray(th_range[1], dtype=grid_real_dtype),
            th_resolution,
            endpoint=True,
        )
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

    magnification = magnification_unnorm / rho_grid**2 / jnp.pi
    return jnp.asarray(magnification, dtype=output_dtype)



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
    """Stable four-point cubic (Lagrange) interpolation with scaling.

    Evaluates the cubic interpolant passing through the four points
    ``(x_k, y_k)`` for ``k = 0..3`` at position ``x``. To improve numerical
    stability when the abscissas are nearly collinear or clustered, the domain
    is rescaled to ``[0, 1]`` prior to computing the Lagrange basis. The
    ``epsilon`` guard prevents divisions by zero in degenerate configurations.

    Parameters
    ----------
    x : float or Array
        Evaluation coordinate.
    x0, x1, x2, x3 : float or Array
        Sample abscissas defining the interpolant.
    y0, y1, y2, y3 : float or Array
        Ordinates associated with ``x0`` – ``x3``.
    epsilon : float, optional
        Small positive value added to denominators to avoid division by zero.

    Returns
    -------
    float or Array
        Interpolated value evaluated at ``x``.

    Notes
    -----
    - Used to estimate the angular crossing location of the source limb within
      a four-cell stencil.
    - Consider alternative schemes when monotonicity constraints are required.
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
