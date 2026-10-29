import jax
import jax.numpy as jnp
import numpy as np
import pytest

from microjax.inverse_ray.cpu.line_roots import binary_line_level_set_roots

jax.config.update("jax_enable_x64", True)


@pytest.mark.fast
def test_clustered_line_preserves_complex_pair_and_real_interval():
    """A four-root cluster must not manufacture two extra real boundaries."""

    source = -1.4998598061310287 + 0.003443021045947857j
    abscissa = -0.0004415497608313467
    rho = 0.00021142727844446993
    # Independent 100-digit roots of the unexpanded physical level equation.
    expected = np.asarray([
        -0.50019028574250084110,
        -0.49985138312807610138,
        -0.50000503056045717074 + 0.000034688541763715714188j,
        -0.50000503056045717074 - 0.000034688541763715714188j,
        1.99988667112577429298 + 0.0030145289751321613910j,
        1.99988667112577429298 - 0.0030145289751321613910j,
    ])
    solve = jax.jit(lambda w: binary_line_level_set_roots(
        abscissa * 1j, -1.0 + 0j, w, rho, s=0.5, q=1e-6
    ))
    widths = []
    for x in [np.nextafter(source.real, -np.inf), source.real,
              np.nextafter(source.real, np.inf)]:
        roots = np.asarray(solve(jnp.asarray(x + source.imag * 1j)))
        real = np.sort(roots[roots.imag == 0].real)
        assert len(real) == 2
        distances = np.abs(roots[:, None] - expected[None, :])
        assert np.max(np.min(distances, axis=1)) < 5e-10
        widths.append(real[1] - real[0])
    expected_width = expected[1].real - expected[0].real
    np.testing.assert_allclose(widths, expected_width, rtol=1e-8, atol=0)


@pytest.mark.fast
@pytest.mark.parametrize(
    "source,rho,s,q,abscissa,expected_width",
    [
        pytest.param(
            0.8313728903683649 - 1.4412236148901905e-05j,
            3e-5, 1.5, 2.6826957952797274e-6, -2.816294167668409e-5,
            2.6301710142506707e-5, id="case465-small-q",
        ),
        pytest.param(
            1.499909514814906 - 0.0036067133637885766j,
            0.00021142727844446993, 0.5, 1e6, 0.00037717753751994844,
            0.0001730582071526321, id="case8-reflected-large-q",
        ),
    ],
)
def test_thin_interval_width_matches_high_precision_on_both_mass_orders(
    source, rho, s, q, abscissa, expected_width
):
    """Independent 100-digit widths expose global-origin cancellation."""

    solve = jax.jit(lambda w: binary_line_level_set_roots(
        abscissa * 1j, -1.0 + 0j, w, rho, s=s, q=q
    ))
    roots = np.asarray(solve(jnp.asarray(source)))
    real = np.sort(roots[roots.imag == 0].real)
    assert len(real) == 2
    np.testing.assert_allclose(real[1] - real[0], expected_width, rtol=1e-8, atol=0)


def _physical_level(radius, parameters):
    x, y, rho, s, q, abscissa = parameters
    image = abscissa + 1j * radius
    a = s / 2
    e = q / (1 + q)
    shift = a * (1 - q) / (1 + q)
    centered_bar = jnp.conj(image) - shift
    denominator = (centered_bar - a) * (centered_bar + a)
    numerator = (image - (x + 1j * y)) * denominator - (
        centered_bar + a * (2 * e - 1)
    )
    return jnp.real(numerator * jnp.conj(numerator)) - rho**2 * jnp.real(
        denominator * jnp.conj(denominator)
    )


@pytest.mark.fast
def test_shifted_root_jvp_satisfies_original_physical_equation():
    """Origin motion and local normalization must cancel in physical AD."""

    s, q, rho, abscissa = 1.3, 0.3, 0.08, 0.35
    shift = s * (1 - q) / (2 * (1 + q))
    a, e = s / 2, q / (1 + q)
    image = abscissa + 0.6j
    zbar = np.conj(image) - shift
    source = image - e / (zbar - a) - (1 - e) / (zbar + a)
    parameters = jnp.asarray([source.real, source.imag, rho, s, q, abscissa])
    tangent = jnp.asarray([0.4, -0.1, 0.02, 0.03, 0.01, 0.2])

    def solve(p):
        return binary_line_level_set_roots(
            p[5] + 0j, 1j, p[0] + 1j * p[1], p[2], s=p[3], q=p[4]
        )

    roots, derivative = jax.jvp(solve, (parameters,), (tangent,))
    real = np.asarray(roots.imag == 0)
    assert real.any()
    radius = roots.real[real]
    slope = jax.vmap(jax.grad(_physical_level), in_axes=(0, None))(
        radius, parameters
    )
    _, level_tangent = jax.jvp(
        lambda p: jax.vmap(_physical_level, in_axes=(0, None))(radius, p),
        (parameters,), (tangent,)
    )
    np.testing.assert_allclose(
        np.asarray(derivative.real[real]), np.asarray(-level_tangent / slope),
        rtol=1e-8, atol=1e-9,
    )
