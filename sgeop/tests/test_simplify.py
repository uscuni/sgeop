import pathlib

import geopandas
import pytest
import shapely
from pandas.testing import assert_series_equal

import sgeop

test_data = pathlib.Path("sgeop", "tests", "data")
full_fua_data = pathlib.Path("data")

ci_artifacts = pathlib.Path("ci_artifacts")


def test_simplify_network_small():
    ac = "apalachicola"
    known = geopandas.read_parquet(test_data / f"{ac}_simplified.parquet")
    observed = sgeop.simplify_network(
        geopandas.read_parquet(test_data / f"{ac}_original.parquet")
    )

    observed.to_parquet(ci_artifacts / ac / "simplified.parquet")

    assert_series_equal(known._status, observed._status)
    assert shapely.equals_exact(
        known.geometry.normalize(), observed.geometry.normalize(), tolerance=1e-1
    ).all()


@pytest.mark.parametrize(
    "aoi,tol",
    [
        ("aleppo_1133", 2e-1),
        ("auckland_869", 2e-1),
        ("bucaramanga_4617", 2e-1),
        ("douala_809", 1e-1),
        ("liege_1656", 2e-1),
        ("slc_4881", 2e-1),
    ],
)
def test_simplify_network_full_fua(aoi, tol):
    known = geopandas.read_parquet(full_fua_data / aoi / "simplified.parquet")
    observed = sgeop.simplify_network(
        geopandas.read_parquet(full_fua_data / aoi / "original.parquet")
    )

    observed.to_parquet(ci_artifacts / aoi / "simplified.parquet")

    assert_series_equal(known._status, observed._status)
    assert shapely.equals_exact(
        known.geometry.normalize(), observed.geometry.normalize(), tolerance=tol
    ).all()
