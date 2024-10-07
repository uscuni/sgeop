import pathlib

import geopandas.testing
import pytest

import sgeop

test_data = pathlib.Path("sgeop", "tests", "data")

ac = "apalachicola"


def test_simplify():
    known = geopandas.read_parquet(test_data / f"{ac}_simplified.parquet")
    observed = sgeop.simplify_network(
        geopandas.read_parquet(test_data / f"{ac}_original.parquet")
    )

    pytest.geom_test(observed.geometry, known.geometry, relax=True)
