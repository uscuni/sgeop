import pathlib

import geopandas
import pytest
import shapely
from pandas.testing import assert_series_equal

import sgeop

test_data = pathlib.Path("sgeop", "tests", "data")
full_fua_data = pathlib.Path("data")


def test_simplify_network_small():
    ac = "apalachicola"
    known = geopandas.read_parquet(test_data / f"{ac}_simplified.parquet")
    observed = sgeop.simplify_network(
        geopandas.read_parquet(test_data / f"{ac}_original.parquet")
    )

    assert shapely.equals_exact(
        known.geometry.normalize(), observed.geometry.normalize(), tolerance=1e-1
    ).all()
    assert_series_equal(known._status, observed._status)


@pytest.mark.parametrize(
    "aoi",
    [
        "aleppo_1133",
        "auckland_869",
        "bucaramanga_4617",
        "douala_809",
        "liege_1656",
        "slc_4881",
    ],
)
def test_simplify_network_full_fua(aoi):
    known = geopandas.read_parquet(full_fua_data / aoi / "simplified.parquet")
    observed = sgeop.simplify_network(
        geopandas.read_parquet(full_fua_data / aoi / "original.parquet")
    )

    assert shapely.equals_exact(
        known.geometry.normalize(), observed.geometry.normalize(), tolerance=1e-1
    ).all()
    assert_series_equal(known._status, observed._status)
