import geopandas
import momepy


def continuity(
    roads: geopandas.GeoDataFrame, angle_threshold: float = 120
) -> geopandas.GeoDataFrame:
    """Assign COINS-based information to roads.

    Parameters
    ----------
    roads :  geopandas.GeoDataFrame
        Road network.

    Returns
    -------
    geopandas.GeoDataFrame
        The input ``roads`` with additional columns where the original
        index may be reset (see ``dedup`` keyword argument).
    """
    roads = roads.copy()

    # Measure continuity of street network
    coins = momepy.COINS(roads, angle_threshold=angle_threshold, flow_mode=True)

    # Assing continuity group
    group, end = coins.stroke_attribute(True)
    roads["coins_group"] = group
    roads["coins_end"] = end

    # Assign length of each continuity group and a number of segments within the group.
    coins_grouped = roads.length.groupby(roads.coins_group)
    roads["coins_len"] = coins_grouped.sum()[roads.coins_group].values
    roads["coins_count"] = coins_grouped.size()[roads.coins_group].values

    return roads, coins


def get_stroke_info(artifacts, roads):
    """Generate information about strokes within the artifacts

    Resulting lists can be assigned as columns to ``artifacts``.

    Parameters
    ----------
    artifacts : GeoDataFrame | GeoSeries
        Polygons representing the artifacts
    roads : GeoDataFrame | GeoSeries
        LineStrings representing the road network

    Returns
    -------
    stroke_count : list
    C_count : list
    E_count : list
    S_count : list
    """
    strokes = []
    c_ = []
    e_ = []
    s_ = []
    for geom in artifacts.geometry:
        singles = 0
        ends = 0
        edges = roads.iloc[roads.sindex.query(geom, predicate="covers")]
        if (  # roundabout special case
            edges.coins_group.nunique() == 1
            and edges.shape[0] == edges.coins_count.iloc[0]
        ):
            singles = 1
            mains = 0
        else:
            all_ends = edges[edges.coins_end]
            mains = edges[
                ~edges.coins_group.isin(all_ends.coins_group)
            ].coins_group.nunique()

            visited = []
            for coins_count, group in zip(
                all_ends.coins_count, all_ends.coins_group, strict=True
            ):
                if (group not in visited) and (
                    coins_count == (edges.coins_group == group).sum()
                ):
                    singles += 1
                    visited.append(group)
                elif group not in visited:
                    ends += 1
                    # do not add to visited as they may be disjoint within the artifact
        strokes.append(edges.coins_group.nunique())
        c_.append(mains)
        e_.append(ends)
        s_.append(singles)
    return strokes, c_, e_, s_
