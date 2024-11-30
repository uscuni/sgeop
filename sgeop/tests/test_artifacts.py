import pathlib

import geopandas
import pytest

import sgeop


def test_get_artifacts_error():
    path = pathlib.Path("sgeop", "tests", "data", "apalachicola_original.parquet")
    with pytest.raises(  # noqa: SIM117
        ValueError,
        match=(
            "No threshold for artifact detection found. Pass explicit "
            "`threshold` or `threshold_fallback` to provide the value directly."
        ),
    ):

        sgeop.artifacts.get_artifacts(geopandas.read_parquet(path).iloc[:3])
