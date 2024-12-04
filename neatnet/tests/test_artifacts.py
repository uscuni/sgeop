import pathlib

import geopandas
import pytest

import neatnet


def test_get_artifacts_error():
    path = pathlib.Path("neatnet", "tests", "data", "apalachicola_original.parquet")
    with pytest.raises(  # noqa: SIM117
        ValueError,
        match=(
            "No threshold for artifact detection found. Pass explicit "
            "`threshold` or `threshold_fallback` to provide the value directly."
        ),
    ):
        with pytest.warns(
            UserWarning,
            match=(
                "Input roads could not not be polygonized. "
                "Identification of face artifacts not possible."
            ),
        ):
            neatnet.artifacts.get_artifacts(geopandas.read_parquet(path).iloc[:3])
