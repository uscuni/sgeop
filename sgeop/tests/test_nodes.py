import pathlib

import geopandas.testing
import numpy
import pytest
import shapely
from pandas.testing import assert_series_equal

import sgeop





'''
def test_split():
    
    crs = "EPSG:3857"
    
    known = geopandas.GeoDataFrame(
        {"_status": ["changed", "changed"]},
        geometry=[shapely.LineString(((2,2), (5,5))), shapely.LineString(((5,5), (8,8)))],
        crs=crs
    )

    split_points = [shapely.Point(5,5)]

    cleaned_roads = geopandas.GeoSeries(
        [shapely.LineString(((2,2), (8,8)))], crs=crs
    )

    observed = sgeop.nodes.split(split_points, cleaned_roads, crs)


    assert isinstance(observed, geopandas.GeoDataFrame)
    assert observed.crs == known.crs == cleaned_roads.crs == crs


    geopandas.testing.assert_geodataframe_equal(observed, known)
'''

split_list = [shapely.Point(5,5)]
split_array = numpy.array(split_list)

line_2_8 = shapely.LineString(((2,2), (8,8)))
line_2_5 = shapely.LineString(((2,2), (5,5)))
line_5_8 = shapely.LineString(((5,5), (8,8)))

crs = "EPSG:3857"
input_series_a = geopandas.GeoSeries([line_2_8], crs=crs)
input_frame_a = geopandas.GeoDataFrame(geometry=[line_2_8], crs=crs)

known_a = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed"]}, geometry=[line_2_5, line_5_8], crs=crs,
)


@pytest.mark.parametrize(
    "cleaned_roads,split_points,known",
    (
        [
            input_series_a,
            split_list,
            known_a
        ],
        [
            input_series_a,
            split_array,
            known_a
        ],
        [
            input_frame_a,
            split_list,
            known_a
        ],
        [
            input_frame_a,
            split_array,
            known_a
        ],
    ),
    ids=["one", "two", "three", "four"]
)
def test_split(cleaned_roads, split_points, known):

    observed = sgeop.nodes.split(split_points, cleaned_roads, crs)

    assert isinstance(observed, geopandas.GeoDataFrame)
    assert observed.crs == known.crs == cleaned_roads.crs == crs

    geopandas.testing.assert_geodataframe_equal(observed, known)