import pathlib

import geopandas
import numpy
import pytest
from pandas.testing import assert_series_equal

import sgeop

test_data = pathlib.Path("sgeop", "tests", "data")
full_fua_data = pathlib.Path("data")

ci_artifacts = pathlib.Path("ci_artifacts")


def test_simplify_network_small():
    ac = "apalachicola"
    known = geopandas.read_parquet(test_data / f"{ac}_simplified.parquet")
    known_length = 78089.0

    observed = sgeop.simplify_network(
        geopandas.read_parquet(test_data / f"{ac}_original.parquet")
    )
    observed_length = observed.geometry.length.sum()

    # storing GH artifacts
    artifact_dir = ci_artifacts / ac
    artifact_dir.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(artifact_dir / "simplified.parquet")

    assert pytest.approx(observed_length, rel=0.00001) == known_length
    assert observed.index.dtype == numpy.dtype("int64")

    assert observed.shape == known.shape
    assert_series_equal(known._status, observed._status)

    # see gh#.....
    known = known.drop(index=746)
    observed = observed.drop(index=746)

    pytest.geom_test(known, observed, tolerance=1.5)


"""
@pytest.mark.parametrize(
    "aoi,tol,known_length",
    [
        ("aleppo_1133", 2e-1, 4_361_625),
        ("auckland_869", 2e-1, 1_268_048),
        ("bucaramanga_4617", 2e-1, 1_681_011),
        ("douala_809", 1e-1, 2_961_364),
        ("liege_1656", 2e-1, 2_350_782),
        ("slc_4881", 2e-1, 1_762_456),
    ],
)
def test_simplify_network_full_fua(aoi, tol, known_length):
    known = geopandas.read_parquet(full_fua_data / aoi / "simplified.parquet")
    observed = sgeop.simplify_network(
        geopandas.read_parquet(full_fua_data / aoi / "original.parquet")
    )
    observed_length = observed.geometry.length.sum()

    # storing GH artifacts
    artifact_dir = ci_artifacts / aoi
    artifact_dir.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(artifact_dir / "simplified.parquet")

    assert pytest.approx(observed_length, rel=0.0001) == known_length
    assert observed.index.dtype == numpy.dtype("int64")

    if pytest.ubuntu and pytest.env_type != "oldest":
        assert_series_equal(known._status, observed._status)
        pytest.geom_test(known, observed, tolerance=tol)
"""
