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


class TestAngelBetween2Lines:
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
