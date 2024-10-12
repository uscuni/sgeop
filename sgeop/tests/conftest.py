import geopandas.testing
import numpy
import pandas
import pytest
import shapely

line_collection = (
    list[shapely.LineString]
    | tuple[shapely.LineString]
    | numpy.ndarray
    | pandas.Series
    | geopandas.GeoSeries
)


geometry_collection = (
    list[shapely.GeometryCollection]
    | tuple[shapely.GeometryCollection]
    | numpy.ndarray
    | pandas.Series
    | geopandas.GeoSeries
)


def polygonize(
    collection: line_collection, as_geom: bool = True
) -> shapely.Polygon | geopandas.GeoSeries:
    """Testing helper -- Create polygon from collection of lines."""
    if isinstance(collection, pandas.Series | geopandas.GeoSeries):
        _poly = geopandas.GeoSeries(collection).polygonize().buffer(0)
        if as_geom:
            return _poly.squeeze()
        else:
            return _poly
    else:
        return shapely.polygonize(collection).buffer(0)


def is_geopandas(collection: geometry_collection) -> bool:
    return isinstance(collection, geopandas.GeoSeries | geopandas.GeoDataFrame)


def geom_test(
    collection1: geometry_collection,
    collection2: geometry_collection,
    tolerance: float = 1e-1,
) -> bool:
    """Testing helper -- geometry verification."""

    if not is_geopandas(collection1):
        collection1 = geopandas.GeoSeries(collection1)

    if not is_geopandas(collection2):
        collection2 = geopandas.GeoSeries(collection2)

    assert shapely.equals_exact(
        collection1.geometry.normalize(),
        collection2.geometry.normalize(),
        tolerance=tolerance,
    ).all()


def pytest_configure(config):  # noqa: ARG001
    """PyTest session attributes, methods, etc."""

    pytest.polygonize = polygonize
    pytest.geom_test = geom_test
