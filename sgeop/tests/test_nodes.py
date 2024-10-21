import itertools

import geopandas.testing
import numpy
import pandas
import pytest
import shapely

import sgeop

crs = "EPSG:3857"

point_1 = shapely.Point(1, 1)
point_2 = shapely.Point(2, 2)
point_3 = shapely.Point(3, 3)
point_4 = shapely.Point(4, 4)
point_5 = shapely.Point(5, 5)
point_6 = shapely.Point(6, 6)
point_7 = shapely.Point(7, 7)
point_8 = shapely.Point(8, 8)
point_9 = shapely.Point(9, 9)
point_24 = shapely.Point(2, 4)
point_42 = shapely.Point(4, 2)

split_list_2 = [point_2]
split_array_2 = numpy.array(split_list_2)
split_series_2 = geopandas.GeoSeries(split_array_2)

split_list_3 = [point_3]
split_array_3 = numpy.array(split_list_3)
split_series_3 = geopandas.GeoSeries(split_array_3)

split_list_2_3 = split_list_2 + split_list_3
split_array_2_3 = numpy.array(split_list_2_3)
split_series_2_3 = geopandas.GeoSeries(split_array_2_3)

split_list_2_8 = split_list_2 + [point_8]
split_array_2_8 = numpy.array(split_list_2_8)
split_series_2_8 = geopandas.GeoSeries(split_array_2_8)

split_list_2_3_7_8 = split_list_2_3 + [point_7, point_8]
split_array_2_3_7_8 = numpy.array(split_list_2_3_7_8)
split_series_2_3_7_8 = geopandas.GeoSeries(split_array_2_3_7_8)

line_1_4 = shapely.LineString((point_1, point_4))
line_1_2 = shapely.LineString((point_1, point_2))
line_1_3 = shapely.LineString((point_1, point_3))
line_2_3 = shapely.LineString((point_2, point_3))
line_2_4 = shapely.LineString((point_2, point_4))
line_3_4 = shapely.LineString((point_3, point_4))
line_6_9 = shapely.LineString((point_6, point_9))
line_6_7 = shapely.LineString((point_6, point_7))
line_6_8 = shapely.LineString((point_6, point_8))
line_7_8 = shapely.LineString((point_7, point_8))
line_8_9 = shapely.LineString((point_8, point_9))
line_24_42 = shapely.LineString((point_24, point_42))
line_24_3 = shapely.LineString((point_24, point_3))
line_3_42 = shapely.LineString((point_3, point_42))

cases = range(1, 9)
types = ["list", "array", "series"]

# case 1: 1 road input -- not split
cleaned_roads_1 = geopandas.GeoDataFrame(geometry=[line_1_2], crs=crs)
known_1 = cleaned_roads_1.copy()

# case 2: 1 road input -- split once
cleaned_roads_2 = geopandas.GeoDataFrame(geometry=[line_1_4], crs=crs)
known_2 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed"]},
    geometry=[line_1_2, line_2_4],
    crs=crs,
)

# case 3: 1 road input -- split twice
cleaned_roads_3 = geopandas.GeoDataFrame(geometry=[line_1_4], crs=crs)
known_3 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed"]},
    geometry=[line_1_2, line_2_3, line_3_4],
    crs=crs,
)

# case 4: 2 roads input -- neither roads split
cleaned_roads_4 = geopandas.GeoDataFrame(geometry=[line_1_2, line_2_4], crs=crs)
known_4 = cleaned_roads_4.copy()

# case 5: 2 roads input -- 1 road split once
cleaned_roads_5 = geopandas.GeoDataFrame(geometry=[line_1_4, line_6_9], crs=crs)
known_5 = geopandas.GeoDataFrame(
    {"_status": [numpy.nan, "changed", "changed"]},
    geometry=[line_6_9, line_1_2, line_2_4],
    crs=crs,
)

# case 6: 2 roads input -- 2 roads split once (unique splits)
cleaned_roads_6 = cleaned_roads_5.copy()
known_6 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed", "changed"]},
    geometry=[line_1_2, line_2_4, line_6_8, line_8_9],
    crs=crs,
)

# case 7: 2 roads input -- 2 roads split twice (unique splits)
cleaned_roads_7 = cleaned_roads_5.copy()
known_7 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed", "changed", "changed", "changed"]},
    geometry=[line_1_2, line_2_3, line_3_4, line_6_7, line_7_8, line_8_9],
    crs=crs,
)

# case 8: 2 roads input (perpendicular)-- 2 roads split once (intersection)
cleaned_roads_8 = geopandas.GeoDataFrame(geometry=[line_1_4, line_24_42], crs=crs)
known_8 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed", "changed"]},
    geometry=[line_1_3, line_3_4, line_24_3, line_3_42],
    crs=crs,
)


@pytest.mark.parametrize(
    "split_points,cleaned_roads,known",
    (
        [split_list_2, cleaned_roads_1, known_1],  # case 1
        [split_array_2, cleaned_roads_1, known_1],
        [split_series_2, cleaned_roads_1, known_1],
        [split_list_2, cleaned_roads_2, known_2],  # case 2
        [split_array_2, cleaned_roads_2, known_2],
        [split_series_2, cleaned_roads_2, known_2],
        [split_list_2_3, cleaned_roads_3, known_3],  # case 3
        [split_array_2_3, cleaned_roads_3, known_3],
        [split_series_2_3, cleaned_roads_3, known_3],
        [split_list_2, cleaned_roads_4, known_4],  # case 4
        [split_array_2, cleaned_roads_4, known_4],
        [split_series_2, cleaned_roads_4, known_4],
        [split_list_2, cleaned_roads_5, known_5],  # case 5
        [split_array_2, cleaned_roads_5, known_5],
        [split_series_2, cleaned_roads_5, known_5],
        [split_list_2_8, cleaned_roads_6, known_6],  # case 6
        [split_array_2_8, cleaned_roads_6, known_6],
        [split_series_2_8, cleaned_roads_6, known_6],
        [split_list_2_3_7_8, cleaned_roads_7, known_7],  # case 7
        [split_array_2_3_7_8, cleaned_roads_7, known_7],
        [split_series_2_3_7_8, cleaned_roads_7, known_7],
        [split_list_3, cleaned_roads_8, known_8],  # case 8
        [split_array_3, cleaned_roads_8, known_8],
        [split_series_3, cleaned_roads_8, known_8],
    ),
    ids=[f"case{c}-{t}" for c, t in list(itertools.product(cases, types))],
)
def test_split(split_points, cleaned_roads, known):
    observed = sgeop.nodes.split(split_points, cleaned_roads, crs)
    assert isinstance(observed, geopandas.GeoDataFrame)
    assert observed.crs == known.crs == cleaned_roads.crs == crs
    pytest.geom_test(observed.geometry, known.geometry)
    if "_status" in observed.columns:
        pandas.testing.assert_series_equal(observed["_status"], known["_status"])


point_20001 = shapely.Point(2.0001, 2.0001)
point_21 = shapely.Point(2.1, 2.1)

line_1_20001 = shapely.LineString((point_1, point_20001))
line_20001_4 = shapely.LineString((point_20001, point_4))
line_1_21 = shapely.LineString((point_1, point_21))
line_21_4 = shapely.LineString((point_21, point_4))
line_1_6 = shapely.LineString((point_1, point_6))


@pytest.mark.parametrize(
    "edge,split_point,tol,known",
    (
        [line_1_4, point_2, 0.0000001, numpy.array([line_1_2, line_2_4])],
        [line_1_4, point_20001, 0.0000001, numpy.array([line_1_20001, line_20001_4])],
        [line_1_4, point_21, 0.0001, numpy.array([line_1_21, line_21_4])],
        [line_1_4, point_6, 0.1, numpy.array([line_1_4])],
        [line_1_4, point_6, 3, numpy.array([line_1_6])],
    ),
    ids=["exact", "precise", "relaxed", "ignore", "extend"],
)
def test_snap_n_split(edge, split_point, tol, known):
    observed = sgeop.nodes._snap_n_split(edge, split_point, tol)
    numpy.testing.assert_array_equal(observed, known)
