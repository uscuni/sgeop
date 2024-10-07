import geopandas.testing
import pytest


def geom_test(collection1, collection2, relax=False):
    """Testing helper -- geometry verification."""
    geopandas.testing.assert_geoseries_equal(
        geopandas.GeoSeries(collection1),
        geopandas.GeoSeries(collection2),
        check_less_precise=relax,
        normalize=relax,
    )


def pytest_configure(config):  # noqa: ARG001
    """PyTest session attributes, methods, etc."""

    pytest.geom_test = geom_test
