import geopandas.testing
import numpy
import pandas
import pytest
import shapely
from packaging.version import Version

GPD_GE_100 = Version(geopandas.__version__) >= Version("1.0.0")

line_collection = (
    list[shapely.LineString]
    | tuple[shapely.LineString]
    | numpy.ndarray[shapely.LineString]
    | pandas.Series
    | geopandas.GeoSeries
)


def polygonize(
    collection: line_collection, as_geom: bool = True
) -> shapely.Polygon | geopandas.GeoSeries:
    """Testing helper -- Create polygon from collection of lines."""

    def _collection(c):
        return shapely.polygonize(c)

    if isinstance(collection, pandas.Series | geopandas.GeoSeries):
        if GPD_GE_100:
            _poly = geopandas.GeoSeries(collection).polygonize().buffer(0)
        else:
            _poly = geopandas.GeoSeries(_collection(list(collection)))
        if as_geom:
            return _poly.squeeze()
        else:
            return _poly
    else:
        return shapely.polygonize(collection).buffer(0)


def geom_test(collection1, collection2, relax=False):
    """Testing helper -- geometry verification."""
    geopandas.testing.assert_geoseries_equal(
        geopandas.GeoSeries(collection1),
        geopandas.GeoSeries(collection2),
        check_less_precise=relax,
    )


def pytest_configure(config):  # noqa: ARG001
    """PyTest session attributes, methods, etc."""

    pytest.polygonize = polygonize
    pytest.geom_test = geom_test
