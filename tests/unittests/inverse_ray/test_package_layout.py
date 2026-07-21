import microjax.inverse_ray.lightcurve as current_lightcurve
import microjax.inverse_ray.extended_source as current_extended_source
import microjax.inverse_ray_dense.lightcurve as dense_lightcurve
import microjax.inverse_ray_dense.extended_source as dense_extended_source
import microjax.inverse_ray_retry.lightcurve as retry_lightcurve
import microjax.inverse_ray_retry.extended_source as retry_extended_source


def test_inverse_ray_packages_expose_disjoint_algorithms():
    assert current_lightcurve.__all__ == ["mag_binary", "mag_triple"]
    assert dense_lightcurve.__all__ == ["mag_binary_dense", "mag_triple"]
    assert retry_lightcurve.__all__ == ["mag_binary_safe"]

    assert not hasattr(current_lightcurve, "mag_binary_dense")
    assert not hasattr(current_lightcurve, "mag_binary_safe")
    assert not hasattr(dense_lightcurve, "mag_binary")
    assert not hasattr(dense_lightcurve, "mag_binary_safe")
    assert not hasattr(retry_lightcurve, "mag_binary")
    assert not hasattr(retry_lightcurve, "mag_binary_dense")

    assert not hasattr(current_extended_source, "mag_uniform_local_boundary")
    assert not hasattr(current_extended_source, "mag_uniform")
    assert not hasattr(dense_extended_source, "mag_uniform_boundary")
    assert not hasattr(retry_extended_source, "mag_uniform")


def test_backend_packages_use_their_own_extended_source_modules():
    assert dense_lightcurve.mag_uniform.__module__.startswith(
        "microjax.inverse_ray_dense."
    )
    assert retry_lightcurve.mag_uniform_boundary.__module__.startswith(
        "microjax.inverse_ray_retry."
    )
    assert current_lightcurve.mag_uniform_boundary.__module__.startswith(
        "microjax.inverse_ray."
    )


def test_current_boundary_facade_delegates_by_responsibility():
    assert current_extended_source.mag_uniform_boundary.__module__.endswith(
        ".integrators.uniform"
    )
    assert current_extended_source.mag_uniform_triple_boundary.__module__.endswith(
        ".integrators.triple"
    )
    assert current_extended_source.mag_radial_profile_boundary.__module__.endswith(
        ".integrators.profile"
    )
    assert current_extended_source.mag_limb_dark_boundary.__module__.endswith(
        ".integrators.limb_dark"
    )
