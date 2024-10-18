import io

import geopandas.testing
import momepy
import pandas
import pytest
import shapely

import sgeop


@pytest.fixture
def roads() -> geopandas.GeoDataFrame:
    """Toy set of 'roads' for testing only."""
    inita = 1
    final = 8
    grid = list(range(inita, final))
    vert_points = list(zip(grid[:-1], grid[1:], strict=True))
    hori_points = [(j, i) for i, j in vert_points]
    vert_lines = [
        shapely.LineString(i)
        for i in list(zip(hori_points[:-1], vert_points[1:], strict=True))
    ]
    hori_lines = [
        shapely.LineString(i)
        for i in list(zip(vert_points[:-1], hori_points[1:], strict=True))
    ]
    return geopandas.GeoDataFrame(
        geometry=(
            vert_lines
            + hori_lines
            + [
                shapely.LineString(((4, 5), (3, 6))),
                shapely.LineString(((3, 6), (4, 4))),
                shapely.LineString(((6, 3), (5, 4))),
                shapely.LineString(((1, 1), (3, 3))),
                shapely.LineString(((5, 5), (6, 6))),
                shapely.LineString(((6, 7), (7, 7))),
                shapely.LineString(((7, 6), (7, 7))),
                shapely.LineString(((1, 7), (2, 7))),
                shapely.LineString(((2, 7), (2, 8))),
                shapely.LineString(((2, 8), (1, 8))),
                shapely.LineString(((1, 8), (1, 7))),
            ]
        )
    )


def test_continuity(roads):
    observed_continuity, observed_coins = sgeop.continuity.continuity(roads)

    assert isinstance(observed_continuity, geopandas.GeoDataFrame)
    known_continuity = (
        geopandas.GeoDataFrame(
            pandas.read_csv(
                io.StringIO(
                    "geometry	coins_group	coins_end	coins_len	coins_count\n"
                    "LINESTRING (2 1, 2 3)	0	True	9.650281539872886	5\n"
                    "LINESTRING (3 2, 3 4)	1	False	7.414213562373095	4\n"
                    "LINESTRING (4 3, 4 5)	0	False	9.650281539872886	5\n"
                    "LINESTRING (5 4, 5 6)	2	True	10.0	6\n"
                    "LINESTRING (6 5, 6 7)	2	False	10.0	6\n"
                    "LINESTRING (1 2, 3 2)	1	True	7.414213562373095	4\n"
                    "LINESTRING (2 3, 4 3)	0	False	9.650281539872886	5\n"
                    "LINESTRING (3 4, 5 4)	1	False	7.414213562373095	4\n"
                    "LINESTRING (4 5, 6 5)	2	True	10.0	6\n"
                    "LINESTRING (5 6, 7 6)	2	False	10.0	6\n"
                    "LINESTRING (4 5, 3 6)	0	False	9.650281539872886	5\n"
                    "LINESTRING (3 6, 4 4)	0	True	9.650281539872886	5\n"
                    "LINESTRING (6 3, 5 4)	1	True	7.414213562373095	4\n"
                    "LINESTRING (1 1, 3 3)	3	True	2.8284271247461903	1\n"
                    "LINESTRING (5 5, 6 6)	4	True	1.4142135623730951	1\n"
                    "LINESTRING (6 7, 7 7)	2	False	10.0	6\n"
                    "LINESTRING (7 6, 7 7)	2	False	10.0	6\n"
                    "LINESTRING (1 7, 2 7)	5	False	4.0	4\n"
                    "LINESTRING (2 7, 2 8)	5	False	4.0	4\n"
                    "LINESTRING (2 8, 1 8)	5	False	4.0	4\n"
                    "LINESTRING (1 8, 1 7)	5	False	4.0	4\n"
                ),
                sep="\t",
            )
        )
        .pipe(lambda df: df.assign(**{"geometry": shapely.from_wkt(df["geometry"])}))
        .set_geometry("geometry")
    )
    geopandas.testing.assert_geodataframe_equal(observed_continuity, known_continuity)

    assert isinstance(observed_coins, momepy.COINS)
    assert observed_coins.already_merged
    assert observed_coins.merging_list == [
        [0, 2, 6, 10, 11],
        [1, 5, 7, 12],
        [3, 4, 8, 9, 15, 16],
        [13],
        [14],
        [17, 18, 19, 20],
    ]
    assert len(observed_coins.angle_pairs) == 40


def test_get_stroke_info(roads):
    known_strokes = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    known_c_ = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    known_e_ = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    known_s_ = [0, 0, 0, 0, 0, 1, 1, 0, 1]

    observed = sgeop.continuity.get_stroke_info(
        sgeop.artifacts.get_artifacts(roads, threshold=1)[0],
        sgeop.continuity.continuity(roads.copy())[0],
    )

    observed_strokes = observed[0]
    observed_c_ = observed[1]
    observed_e_ = observed[2]
    observed_s_ = observed[3]

    assert observed_strokes == known_strokes
    assert observed_c_ == known_c_
    assert observed_e_ == known_e_
    assert observed_s_ == known_s_
