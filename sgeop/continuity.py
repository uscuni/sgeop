import geopandas
import momepy


def continuity(
    roads: geopandas.GeoDataFrame, angle_threshold: float = 120
) -> tuple[geopandas.GeoDataFrame, momepy.COINS]:
    """Assign COINS-based information to roads.

    Parameters
    ----------
    roads :  geopandas.GeoDataFrame
        Road network.
    angle_threshold : float = 120
        See the ``angle_threshold`` keyword argument in ``momepy.COINS()``.

    Returns
    -------
    roads : geopandas.GeoDataFrame
        The first eleThe input ``roads`` with additional columns where the original
        index may be reset (see ``dedup`` keyword argument).
    coins : momepy.COINS
        **This is not used in production.**

    Notes
    -----
    The returned ``coins`` object is not used in production, but is
    very helpful in testing & debugging. See gh:sgeop#49.
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


def get_stroke_info(
    artifacts: geopandas.GeoSeries | geopandas.GeoDataFrame,
    roads: geopandas.GeoSeries | geopandas.GeoDataFrame,
) -> tuple[list[int]]:
    """Generate information about strokes within ``artifacts`` and the
    resulting lists can be assigned as columns to ``artifacts``.

    Parameters
    ----------
    artifacts : GeoSeries | GeoDataFrame
        Polygons representing the artifacts.
    roads : GeoSeries | GeoDataFrame
        LineStrings representing the road network.

    Returns
    -------
    stroke_count : list[int]
        ...
    C_count : list[int]
        ...
    E_count : list[int]
        ...
    S_count : list[int]
        ...
    """
    strokes = []
    c_ = []
    e_ = []
    s_ = []
    for geom in artifacts.geometry:
        singles = 0
        ends = 0
        edges = roads.iloc[roads.sindex.query(geom, predicate="covers")]
        ecg = edges.coins_group
        if ecg.nunique() == 1 and edges.shape[0] == edges.coins_count.iloc[0]:
            # roundabout special case
            singles = 1
            mains = 0
        else:
            all_ends = edges[edges.coins_end]
            ae_cg = all_ends.coins_group
            mains = edges[~ecg.isin(ae_cg)].coins_group.nunique()
            visited = []
            for coins_count, group in zip(all_ends.coins_count, ae_cg, strict=True):
                if (group not in visited) and (coins_count == (ecg == group).sum()):
                    singles += 1
                    visited.append(group)
                elif group not in visited:
                    ends += 1
                    # do not add to visited as they may be disjoint within the artifact
        strokes.append(ecg.nunique())
        c_.append(mains)
        e_.append(ends)
        s_.append(singles)
    return strokes, c_, e_, s_
