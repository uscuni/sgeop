import pathlib

import geopandas
import shapely
from pandas.testing import assert_series_equal

import sgeop

test_data = pathlib.Path("sgeop", "tests", "data")

ac = "apalachicola"


def test_simplify():
    known = geopandas.read_parquet(test_data / f"{ac}_simplified.parquet")
    observed = sgeop.simplify_network(
        geopandas.read_parquet(test_data / f"{ac}_original.parquet")
    )

    assert shapely.equals_exact(
        known.geometry.normalize(), observed.geometry.normalize(), tolerance=1e-1
    ).all()
    assert_series_equal(known._status, observed._status)
