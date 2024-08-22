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
