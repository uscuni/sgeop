import pathlib

import geopandas
import pytest
from pandas.testing import assert_series_equal

import sgeop

test_data = pathlib.Path("sgeop", "tests", "data")

ac = "apalachicola"


def test_simplify():
    known = geopandas.read_parquet(test_data / f"{ac}_simplified.parquet")
    observed = sgeop.simplify_network(
        geopandas.read_parquet(test_data / f"{ac}_original.parquet")
    )

    pytest.geom_test(known, observed)
    assert_series_equal(known._status, observed._status)
