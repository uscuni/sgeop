import pathlib

import geopandas.testing

import sgeop

test_data = pathlib.Path("sgeop", "tests", "data")

ac = "apalachicola"


def test_simplify():
    known = geopandas.read_parquet(test_data / f"{ac}_simplified.parquet")
    observed = sgeop.simplify_network(
        geopandas.read_parquet(test_data / f"{ac}_original.parquet")
    )

    geopandas.testing.assert_geodataframe_equal(observed, known)
