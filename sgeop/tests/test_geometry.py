import geopandas.testing
import numpy
import pandas
import pytest
import shapely

import sgeop


class TestIsWithin:
    def setup_method(self):
        self.polygon = shapely.Polygon(((0, 0), (10, 0), (10, 10), (0, 10), (0, 0)))

    def test_within_fully(self):
        line = shapely.LineString(((2, 2), (8, 8)))

        known = True
        observed = sgeop.geometry._is_within(line, self.polygon)

        assert known == observed

    def test_within_tol(self):
        line = shapely.LineString(((2, 2), (2, 10.0001)))

        known = True
        observed = sgeop.geometry._is_within(line, self.polygon)

        assert known == observed

    def test_not_within_tol(self):
        line = shapely.LineString(((2, 2), (2, 10.001)))

        known = False
        observed = sgeop.geometry._is_within(line, self.polygon)

        assert known == observed

    def test_within_tol_strict(self):
        line = shapely.LineString(((2, 2), (2, 10.0000001)))

        known = True
        observed = sgeop.geometry._is_within(line, self.polygon, rtol=1e-7)

        assert known == observed

    def test_not_within_tol_strict(self):
        line = shapely.LineString(((2, 2), (2, 10.000001)))

        known = False
        observed = sgeop.geometry._is_within(line, self.polygon, rtol=1e-7)

        assert known == observed

    def test_within_tol_relaxed(self):
        line = shapely.LineString(((2, 2), (2, 11)))

        known = True
        observed = sgeop.geometry._is_within(line, self.polygon, rtol=1)

        assert known == observed

    def test_not_within_tol_relaxed(self):
        line = shapely.LineString(((2, 2), (2, 12)))

        known = False
        observed = sgeop.geometry._is_within(line, self.polygon, rtol=1)

        assert known == observed

    def test_not_within(self):
        line = shapely.LineString(((11, 11), (12, 12)))

        known = False
        observed = sgeop.geometry._is_within(line, self.polygon)

        assert known == observed


class TestAngleBetween2Lines:
    def setup_method(self):
        self.line1 = shapely.LineString(((0, 0), (1, 0)))
        self.line2 = shapely.LineString(((1, 0), (1, 1)))
        self.line3 = shapely.LineString(((0, 0), (0, 1)))
        self.line4 = shapely.LineString(((0, 1), (1, 1)))

    def test_q1(self):
        known = 90.0
        observed = sgeop.geometry.angle_between_two_lines(self.line1, self.line3)
        assert observed == known

    def test_q2(self):
        known = 90.0
        observed = sgeop.geometry.angle_between_two_lines(self.line1, self.line2)
        assert observed == known

    def test_q3(self):
        known = 90.0
        observed = sgeop.geometry.angle_between_two_lines(self.line2, self.line4)
        assert observed == known

    def test_q4(self):
        known = 90.0
        observed = sgeop.geometry.angle_between_two_lines(self.line3, self.line4)
        assert observed == known

    def test_indistinct(self):
        known = 0.0
        with pytest.warns(
            UserWarning,
            match="Input lines are identical - must be distinct. Returning 0.0.",
        ):
            observed = sgeop.geometry.angle_between_two_lines(self.line1, self.line1)
        assert observed == known

    def test_not_adjacent(self):
        known = 0.0
        with pytest.warns(
            UserWarning, match="Input lines do not share a vertex. Returning 0.0."
        ):
            observed = sgeop.geometry.angle_between_two_lines(self.line1, self.line4)
        assert observed == known


voronoi_skeleton_params = pytest.mark.parametrize(
    "lines_type,as_poly,buffer",
    [
        (list, False, None),
        (list, True, 0.001),
        (numpy.array, False, 0.01),
        (numpy.array, True, 0.1),
        (pandas.Series, False, 1),
        (pandas.Series, True, 2.0),
        (geopandas.GeoSeries, False, 5),
        (geopandas.GeoSeries, True, 10.314),
    ],
)


class TestVoronoiSkeleton:
    def setup_method(self):
        self.square = [
            shapely.LineString(((0, 0), (1000, 0))),
            shapely.LineString(((1000, 0), (1000, 1000))),
            shapely.LineString(((0, 0), (0, 1000))),
            shapely.LineString(((0, 1000), (1000, 1000))),
        ]
        self.known_square_skeleton_edges = numpy.array(
            [
                shapely.LineString(((1000, 0), (998, 2), (500, 500))),
                shapely.LineString(((0, 0), (2, 2), (500, 500))),
                shapely.LineString(((1000, 1000), (998, 998), (500, 500))),
                shapely.LineString(((0, 1000), (2, 998), (500, 500))),
            ]
        )
        self.known_square_skeleton_splits = [shapely.Point(0, 0)]
        self.known_square_skeleton_splits_snap_to = [
            shapely.Point(1000, 0),
            shapely.Point(0, 0),
            shapely.Point(0, 1000),
            shapely.Point(1000, 1000),
        ]

    @voronoi_skeleton_params
    def test_square(self, lines_type, as_poly, buffer):
        known_edges = self.known_square_skeleton_edges
        known_splits = self.known_square_skeleton_splits

        lines = lines_type(self.square)
        poly = pytest.polygonize(lines) if as_poly else None
        observed_edges, observed_splits = sgeop.geometry.voronoi_skeleton(
            lines,
            poly=poly,
            buffer=buffer,
        )

        pytest.geom_test(observed_edges, known_edges)
        pytest.geom_test(observed_splits, known_splits)

    @voronoi_skeleton_params
    def test_square_snap_to(self, lines_type, as_poly, buffer):
        known_edges = self.known_square_skeleton_edges
        known_splits = self.known_square_skeleton_splits_snap_to

        lines = lines_type(self.square)
        poly = pytest.polygonize(lines) if as_poly else None
        observed_edges, observed_splits = sgeop.geometry.voronoi_skeleton(
            lines,
            poly=poly,
            buffer=buffer,
            snap_to=(
                pytest.polygonize(geopandas.GeoSeries(lines), as_geom=False)
                .extract_unique_points()
                .explode()
            ),
        )

        pytest.geom_test(observed_edges, known_edges)
        pytest.geom_test(observed_splits, known_splits)


line_100_900 = shapely.LineString(((1000, 1000), (1000, 9000)))
line_100_120 = shapely.LineString(((1000, 1020), (1020, 1020)))
lines_100_900_100_120 = shapely.MultiLineString((line_100_900, line_100_120))
line_110_900 = shapely.LineString(((1000, 9000), (1100, 9000)))


def test_remove_sliver():
    known = line_100_900
    observed = sgeop.geometry._remove_sliver(lines_100_900_100_120)
    assert observed == known


def test_as_parts():
    known = numpy.array([line_100_900, line_100_120, line_110_900])
    observed = sgeop.geometry._as_parts(
        numpy.array([lines_100_900_100_120, line_110_900])
    )
    numpy.testing.assert_array_equal(observed, known)


@pytest.mark.parametrize("tolerance", [0.1, 1, 10, 100, 1_000, 10_000, 100_000])
def test_consolidate(tolerance):
    known = numpy.array([line_100_900, line_100_120, line_110_900])
    observed = sgeop.geometry._consolidate(
        numpy.array([line_100_900, line_100_120, line_110_900]), tolerance
    )
    numpy.testing.assert_array_equal(observed, known)
