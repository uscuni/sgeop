import logging
import warnings

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
from libpysal import graph
from scipy import sparse

from .artifacts import (
    get_artifacts,
    n1_g1_identical,
    nx_gx,
    nx_gx_cluster,
    nx_gx_identical,
)
from .continuity import continuity, get_stroke_info
from .nodes import (
    _status,
    consolidate_nodes,
    fix_topology,
    induce_nodes,
    remove_false_nodes,
    split,
)

logger = logging.getLogger(__name__)


def simplify_singletons(
    artifacts: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    max_segment_length: float | int = 1,
    compute_coins: bool = True,
    min_dangle_length: float | int = 10,
    eps: float = 1e-4,
    clip_limit: float | int = 2,
    simplification_factor: float | int = 2,
    consolidation_tolerance: float | int = 10,
) -> gpd.GeoDataFrame:
    """Simplification of singleton face artifacts – the first simplification step in
    the procedure detailed in ``simplify.simplify_loop()``.

    This process extracts nodes from network edges before computing and labeling
    face artifacts with a ``{C, E, S}`` typology through ``momepy.COINS`` via the
    constituent road geometries.

    Next, the artifacts' constituent line geometries are either dropped or added in
    the following order of typologies:
        1. 1 node and 1 continuity group
        2. more than 1 node and 1 or more identical continuity groups
        3. 2 or more nodes and 2 or more continuity groups

    Non-planar geometries are ignored.

    Parameters
    ----------
    artifacts : geopandas.GeoDataFrame
        Face artifact polygons.
    roads : geopandas.GeoDataFrame
        Preprocessed road network data.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
        Used in multiple internal geometric operations.
    compute_coins : bool = True
        Flag for computing and labeling artifacts with a ``{C, E, S}`` typology through
        ``momepy.COINS`` via the constituent road geometries.
    min_dangle_length : float | int = 10
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    eps : float = 1e-4
        Tolerance epsilon used in multiple internal geometric operations.
    clip_limit : float | int = 2
        Following generation of the Voronoi linework, we clip to fit inside the
        polygon. To ensure we get a space to make proper topological connections
        from the linework to the actual points on the edge of the polygon, we clip
        using a polygon with a negative buffer of ``clip_limit`` or the radius of
        maximum inscribed circle, whichever is smaller.
    simplification_factor : float | int = 2
        The factor by which singles, pairs, and clusters are simplified. The
        ``max_segment_length`` is multiplied by this factor to get the
        simplification epsilon.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation when generating Voronoi skeletons.

    Returns
    -------
    geopandas.GeoDataFrame
        The road network line data following singletons.
    """

    # Get nodes from the network.
    nodes = momepy.nx_to_gdf(momepy.node_degree(momepy.gdf_to_nx(roads)), lines=False)

    # Link nodes to artifacts
    node_idx, artifact_idx = artifacts.sindex.query(
        nodes.geometry, predicate="dwithin", distance=eps
    )
    intersects = sparse.coo_array(
        ([True] * len(node_idx), (node_idx, artifact_idx)),
        shape=(len(nodes), len(artifacts)),
        dtype=np.bool_,
    )

    # Compute number of nodes per artifact
    artifacts["node_count"] = intersects.sum(axis=0)

    # Compute number of stroke groups per artifact
    if compute_coins:
        roads, _ = continuity(roads)
    strokes, c_, e_, s_ = get_stroke_info(artifacts, roads)

    artifacts["stroke_count"] = strokes
    artifacts["C"] = c_
    artifacts["E"] = e_
    artifacts["S"] = s_

    # Filter artifacts caused by non-planar intersections. (TODO: Note that this is not
    # perfect and some 3CC artifacts were non-planar but not captured here).
    artifacts["non_planar"] = artifacts["stroke_count"] > artifacts["node_count"]
    a_idx, r_idx = roads.sindex.query(artifacts.geometry.boundary, predicate="overlaps")
    artifacts.iloc[np.unique(a_idx), artifacts.columns.get_loc("non_planar")] = True

    # Count intersititial nodes (primes).
    artifacts["interstitial_nodes"] = artifacts.node_count - artifacts[
        ["C", "E", "S"]
    ].sum(axis=1)

    # Define the type label.
    ces_type = []
    for x in artifacts[["node_count", "C", "E", "S"]].itertuples():
        ces_type.append(f"{x.node_count}{'C' * x.C}{'E' * x.E}{'S' * x.S}")
    artifacts["ces_type"] = ces_type

    # collect changes
    to_drop = []
    to_add = []
    split_points = []

    planar = artifacts[~artifacts.non_planar].copy()
    planar["buffered"] = planar.buffer(eps)
    if artifacts.non_planar.any():
        logger.debug(f"IGNORING {artifacts.non_planar.sum()} non planar artifacts")

    for artifact in planar.itertuples():
        # get edges relevant for an artifact
        edges = roads.iloc[roads.sindex.query(artifact.buffered, predicate="covers")]

        try:
            if (artifact.node_count == 1) and (artifact.stroke_count == 1):
                logger.debug("FUNCTION n1_g1_identical")
                n1_g1_identical(
                    edges,
                    to_drop=to_drop,
                    to_add=to_add,
                    geom=artifact.geometry,
                    max_segment_length=max_segment_length,
                    clip_limit=clip_limit,
                )

            elif (artifact.node_count > 1) and (len(set(artifact.ces_type[1:])) == 1):
                logger.debug("FUNCTION nx_gx_identical")
                nx_gx_identical(
                    edges,
                    geom=artifact.geometry,
                    to_add=to_add,
                    to_drop=to_drop,
                    nodes=nodes,
                    angle=75,
                    max_segment_length=max_segment_length,
                    clip_limit=clip_limit,
                    consolidation_tolerance=consolidation_tolerance,
                )

            elif (artifact.node_count > 1) and (len(artifact.ces_type) > 2):
                logger.debug("FUNCTION nx_gx")
                nx_gx(
                    edges,
                    artifact=artifact,
                    to_drop=to_drop,
                    to_add=to_add,
                    split_points=split_points,
                    nodes=nodes,
                    max_segment_length=max_segment_length,
                    clip_limit=clip_limit,
                    min_dangle_length=min_dangle_length,
                    consolidation_tolerance=consolidation_tolerance,
                )
            else:
                logger.debug("NON PLANAR")
        except Exception as e:
            warnings.warn(
                f"An error occured at location {artifact.geometry.centroid}. "
                f"The artifact has not been simplified. The original message:\n{e}",
                UserWarning,
                stacklevel=2,
            )

    cleaned_roads = roads.drop(to_drop)
    # split lines on new nodes
    cleaned_roads = split(split_points, cleaned_roads, roads.crs)

    if to_add:
        # create new roads with fixed geometry. Note that to_add and to_drop lists shall
        # be global and this step should happen only once, not for every artifact
        new = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries(to_add).line_merge(), crs=roads.crs
        ).explode()
        new = new[~new.normalize().duplicated()].copy()
        new["_status"] = "new"
        new.geometry = new.simplify(max_segment_length * simplification_factor)
        new_roads = pd.concat(
            [cleaned_roads, new],
            ignore_index=True,
        )
        new_roads = remove_false_nodes(
            new_roads[~(new_roads.is_empty | new_roads.geometry.isna())],
            aggfunc={"_status": _status},
        )

        return new_roads
    else:
        return cleaned_roads


def simplify_pairs(
    artifacts,
    roads,
    max_segment_length=1,
    min_dangle_length=20,
    clip_limit: int = 2,
    simplification_factor=2,
    consolidation_tolerance=10,
):
    """

    Parameters
    ----------

    clip_limit : int = 2
        Following generation of the Voronoi linework in ``geometry.voronoi_skeleton()``,
        we clip to fit inside the polygon. To ensure we get a space to make proper
        topological connections from the linework to the actual points on the edge of
        the polygon, we clip using a polygon with a negative buffer of ``clip_limit``
        or the radius of maximum inscribed circle, whichever is smaller.

    Returns
    -------

    """

    # Get nodes from the network.
    nodes = momepy.nx_to_gdf(momepy.node_degree(momepy.gdf_to_nx(roads)), lines=False)

    # Link nodes to artifacts
    node_idx, artifact_idx = artifacts.sindex.query(
        nodes.buffer(0.1), predicate="intersects"
    )
    intersects = sparse.coo_array(
        ([True] * len(node_idx), (node_idx, artifact_idx)),
        shape=(len(nodes), len(artifacts)),
        dtype=np.bool_,
    )

    # Compute number of nodes per artifact
    artifacts["node_count"] = intersects.sum(axis=0)

    # Compute number of stroke groups per artifact
    roads, _ = continuity(roads)
    strokes, c_, e_, s_ = get_stroke_info(artifacts, roads)

    artifacts["stroke_count"] = strokes
    artifacts["C"] = c_
    artifacts["E"] = e_
    artifacts["S"] = s_

    # Filter artifacts caused by non-planar intersections.
    artifacts["non_planar"] = artifacts["stroke_count"] > artifacts["node_count"]
    a_idx, _ = roads.sindex.query(artifacts.geometry.boundary, predicate="overlaps")
    artifacts.loc[artifacts.index[np.unique(a_idx)], "non_planar"] = True

    artifacts["non_planar_cluster"] = artifacts.apply(
        lambda x: sum(artifacts.loc[artifacts["comp"] == x.comp]["non_planar"]), axis=1
    )
    np_clusters = artifacts[artifacts.non_planar_cluster > 0]
    artifacts_planar = artifacts[artifacts.non_planar_cluster == 0]

    artifacts_w_info = artifacts.merge(
        artifacts_planar.groupby("comp")[artifacts_planar.columns].apply(
            get_solution, roads=roads
        ),
        left_on="comp",
        right_index=True,
    )
    artifacts_under_np = np_clusters[np_clusters.non_planar_cluster == 2].dissolve(
        "comp", as_index=False
    )

    if not artifacts_w_info.empty:
        to_drop = (
            artifacts_w_info.drop_duplicates("comp")
            .query("solution == 'drop_interline'")
            .drop_id
        )

        roads_cleaned = remove_false_nodes(
            roads.drop(to_drop.dropna().values),
            aggfunc={
                "coins_group": "first",
                "coins_end": lambda x: x.any(),
                "_status": _status,
            },
        )
        merged_pairs = artifacts_w_info.query("solution == 'drop_interline'").dissolve(
            "comp", as_index=False
        )

        sorted_by_node_count = artifacts_w_info.sort_values(
            "node_count", ascending=False
        )
        first = sorted_by_node_count.query("solution == 'iterate'").drop_duplicates(
            "comp", keep="first"
        )
        second = sorted_by_node_count.query("solution == 'iterate'").drop_duplicates(
            "comp", keep="last"
        )

        first = pd.concat(
            [first, np_clusters[~np_clusters.non_planar]], ignore_index=True
        )

        for_skeleton = artifacts_w_info.query("solution == 'skeleton'")
    else:
        merged_pairs = pd.DataFrame()
        first = pd.DataFrame()
        second = pd.DataFrame()
        for_skeleton = pd.DataFrame()
        roads_cleaned = roads[
            ["coins_group", "coins_end", "_status", roads.geometry.name]
        ]

    coins_count = (
        roads_cleaned.groupby("coins_group", as_index=False)
        .geometry.count()
        .rename(columns={"geometry": "coins_count"})
    )
    roads_cleaned = roads_cleaned.merge(coins_count, on="coins_group", how="left")

    if not artifacts_under_np.empty:
        for_skeleton = pd.concat([for_skeleton, artifacts_under_np])

    if not merged_pairs.empty or not first.empty:
        roads_cleaned = simplify_singletons(
            pd.concat([merged_pairs, first]),
            roads_cleaned,
            max_segment_length=max_segment_length,
            clip_limit=clip_limit,
            compute_coins=False,
            min_dangle_length=min_dangle_length,
            simplification_factor=simplification_factor,
            consolidation_tolerance=consolidation_tolerance,
        )
        if not second.empty:
            roads_cleaned = simplify_singletons(
                second,
                roads_cleaned,
                max_segment_length=max_segment_length,
                clip_limit=clip_limit,
                compute_coins=True,
                min_dangle_length=min_dangle_length,
                simplification_factor=simplification_factor,
                consolidation_tolerance=consolidation_tolerance,
            )
    if not for_skeleton.empty:
        roads_cleaned = simplify_clusters(
            for_skeleton,
            roads_cleaned,
            max_segment_length=max_segment_length,
            simplification_factor=simplification_factor,
            min_dangle_length=min_dangle_length,
            consolidation_tolerance=consolidation_tolerance,
        )
    return roads_cleaned


def simplify_clusters(
    artifacts,
    roads,
    max_segment_length=1,
    eps=1e-4,
    simplification_factor=2,
    min_dangle_length=20,
    consolidation_tolerance=10,
):
    # Get nodes from the network.
    nodes = momepy.nx_to_gdf(momepy.node_degree(momepy.gdf_to_nx(roads)), lines=False)

    # collect changes
    to_drop = []
    to_add = []

    for _, artifact in artifacts.groupby("comp"):
        # get artifact cluster polygon
        cluster_geom = artifact.union_all()
        # get edges relevant for an artifact
        edges = roads.iloc[
            roads.sindex.query(cluster_geom, predicate="intersects")
        ].copy()

        nx_gx_cluster(
            edges=edges,
            cluster_geom=cluster_geom,
            nodes=nodes,
            to_drop=to_drop,
            to_add=to_add,
            eps=eps,
            max_segment_length=max_segment_length,
            min_dangle_length=min_dangle_length,
            consolidation_tolerance=consolidation_tolerance,
        )

    cleaned_roads = roads.drop(to_drop)

    # create new roads with fixed geometry. Note that to_add and to_drop lists shall be
    # global and this step should happen only once, not for every artifact
    new = gpd.GeoDataFrame(geometry=to_add, crs=roads.crs)
    new["_status"] = "new"
    new["geometry"] = new.line_merge().simplify(
        max_segment_length * simplification_factor
    )
    new_roads = pd.concat(
        [
            cleaned_roads,
            new,
        ],
        ignore_index=True,
    ).explode()
    new_roads = remove_false_nodes(
        new_roads[~new_roads.is_empty], aggfunc={"_status": _status}
    ).drop_duplicates("geometry")

    return new_roads


def get_type(edges, shared_edge):
    if (  # roundabout special case
        edges.coins_group.nunique() == 1 and edges.shape[0] == edges.coins_count.iloc[0]
    ):
        return "S"

    all_ends = edges[edges.coins_end]
    mains = edges[~edges.coins_group.isin(all_ends.coins_group)]
    shared = edges.loc[shared_edge]
    if shared_edge in mains.index:
        return "C"
    if shared.coins_count == (edges.coins_group == shared.coins_group).sum():
        return "S"
    return "E"


def get_solution(group, roads):
    cluster_geom = group.union_all()

    roads_a = roads.iloc[
        roads.sindex.query(group.geometry.iloc[0], predicate="intersects")
    ]
    roads_b = roads.iloc[
        roads.sindex.query(group.geometry.iloc[1], predicate="intersects")
    ]
    covers_a = roads_a.iloc[
        roads_a.sindex.query(group.geometry.iloc[0], predicate="covers")
    ]
    covers_b = roads_b.iloc[
        roads_b.sindex.query(group.geometry.iloc[1], predicate="covers")
    ]
    # find the road segment that is contained within the cluster geometry
    shared = roads.index[roads.sindex.query(cluster_geom, predicate="contains")]
    if shared.empty or covers_a.empty or covers_b.empty:
        return pd.Series({"solution": "non_planar", "drop_id": None})

    shared = shared.item()

    if (np.invert(roads_b.index.isin(covers_a.index)).sum() == 1) or (
        np.invert(roads_a.index.isin(covers_b.index)).sum() == 1
    ):
        return pd.Series({"solution": "drop_interline", "drop_id": shared})

    seen_by_a = get_type(
        covers_a,
        shared,
    )
    seen_by_b = get_type(
        covers_b,
        shared,
    )

    if seen_by_a == "C" and seen_by_b == "C":
        return pd.Series({"solution": "iterate", "drop_id": shared})
    if seen_by_a == seen_by_b:
        return pd.Series({"solution": "drop_interline", "drop_id": shared})
    return pd.Series({"solution": "skeleton", "drop_id": shared})


def simplify_network(
    roads: gpd.GeoDataFrame,
    *,
    max_segment_length: float | int = 1,
    min_dangle_length: float | int = 20,
    clip_limit: float | int = 2,
    simplification_factor: float | int = 2,
    consolidation_tolerance: float | int = 10,
    artifact_threshold: None | float | int = None,
    artifact_threshold_fallback: None | float | int = None,
    area_threshold_blocks: float | int = 1e5,
    isoareal_threshold_blocks: float | int = 0.5,
    area_threshold_circles: float | int = 5e4,
    isoareal_threshold_circles_enclosed: float | int = 0.75,
    isoperimetric_threshold_circles_touching: float | int = 0.9,
    eps: float = 1e-4,
    exclusion_mask: None | gpd.GeoSeries = None,
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    """Top-level workflow for simplifying networks. The input raw road network data is
    first preprocessed (topological corrections & node consolidation) before two
    iterations of artifact detection and simplification.

    Each iteration of the simplification procedure which includes (1.) the removal
    of false nodes; (2.) face artifact classification; and (3.) the line-based
    simplification of face artifacts in the order of single artifacts, pairs of
    artifacts, clusters of artifacts.

    For further information on face artifact detection and extraction
    see :cite:`fleischmann2023`.

    Parameters
    ----------
    roads : geopandas.GeoDataFrame
        Raw road network data.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
        Used in multiple internal geometric operations.
    min_dangle_length : float | int
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    clip_limit : float | int = 2
        Following generation of the Voronoi linework, we clip to fit inside the
        polygon. To ensure we get a space to make proper topological connections
        from the linework to the actual points on the edge of the polygon, we clip
        using a polygon with a negative buffer of ``clip_limit`` or the radius of
        maximum inscribed circle, whichever is smaller.
    simplification_factor : float | int = 2
        The factor by which singles, pairs, and clusters are simplified. The
        ``max_segment_length`` is multiplied by this factor to get the
        simplification epsilon.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation when generating Voronoi skeletons.
    artifact_threshold : None | float | int = None
        When ``artifact_threshold`` is passed, the computed value from
        ``momepy.FaceArtifacts.threshold`` is not used in favor of the
        given value. This is useful for small networks where artifact
        detection may fail or become unreliable.
    artifact_threshold_fallback : None | float | int = None
        If artifact threshold detection fails, this value is used as a fallback.
    area_threshold_blocks : float | int = 1e5
        Areal theshold for block detection.
    isoareal_threshold_blocks : float | int = 0.5
        Isoareal theshold for block detection.
        See ``esda.shape.isoareal_quotient``.
    area_threshold_circles : float | int = 5e4
        Areal theshold for circle detection.
    isoareal_threshold_circles_enclosed : float | int = 0.75
        Isoareal theshold for enclosed circle detection.
        See ``esda.shape.isoareal_quotient``.
    isoperimetric_threshold_circles_touching : float | int = 0.9
        Isoperimetric theshold for enclosed circle touching.
        See ``esda.shape.isoperimetric_quotient``.
    eps : float = 1e-4
        Tolerance epsilon used in multiple internal geometric operations.
    exclusion_mask : None | geopandas.GeoSeries = None
        Polygons used to determine face artifacts to exclude from returned output.
    predicate : str = 'intersects'
        The spatial predicate used to exclude face artifacts from returned output.

    Returns
    -------
    geopandas.GeoDataFrame
        The final, simplified road network line data.
    """

    ################################################################################
    raw_roads = roads.copy()
    ################################################################################

    roads = fix_topology(roads, eps=eps)
    # Merge nearby nodes (up to double of distance used in skeleton).
    roads = consolidate_nodes(roads, tolerance=max_segment_length * 2.1)

    # Identify artifacts
    artifacts, threshold = get_artifacts(
        roads,
        threshold=artifact_threshold,
        threshold_fallback=artifact_threshold_fallback,
        area_threshold_blocks=area_threshold_blocks,
        isoareal_threshold_blocks=isoareal_threshold_blocks,
        area_threshold_circles=area_threshold_circles,
        isoareal_threshold_circles_enclosed=isoareal_threshold_circles_enclosed,
        isoperimetric_threshold_circles_touching=isoperimetric_threshold_circles_touching,
        exclusion_mask=exclusion_mask,
        predicate=predicate,
    )

    ################################################################################
    if (
        artifacts.empty
        and gpd.testing.assert_geoseries_equal(roads.geometry, raw_roads.geometry)
    ):
        STOP
        return roads.reset_index(drop=True)
    ################################################################################

    # Loop 1
    new_roads = simplify_loop(
        roads,
        artifacts,
        max_segment_length=max_segment_length,
        min_dangle_length=min_dangle_length,
        clip_limit=clip_limit,
        simplification_factor=simplification_factor,
        consolidation_tolerance=consolidation_tolerance,
        eps=eps,
    )

    # this is potentially fixing some minor erroneous edges coming from Voronoi
    new_roads = induce_nodes(new_roads, eps=eps)
    new_roads = new_roads[~new_roads.geometry.normalize().duplicated()].copy()

    # Identify artifacts based on the first loop network
    artifacts, _ = get_artifacts(
        new_roads,
        threshold=threshold,
        threshold_fallback=artifact_threshold_fallback,
        area_threshold_blocks=area_threshold_blocks,
        isoareal_threshold_blocks=isoareal_threshold_blocks,
        area_threshold_circles=area_threshold_circles,
        isoareal_threshold_circles_enclosed=isoareal_threshold_circles_enclosed,
        isoperimetric_threshold_circles_touching=isoperimetric_threshold_circles_touching,
        exclusion_mask=exclusion_mask,
        predicate=predicate,
    )
    if artifacts.empty:
        return new_roads.reset_index(drop=True)

    # Loop 2
    final_roads = simplify_loop(
        new_roads,
        artifacts,
        max_segment_length=max_segment_length,
        min_dangle_length=min_dangle_length,
        clip_limit=clip_limit,
        simplification_factor=simplification_factor,
        consolidation_tolerance=consolidation_tolerance,
        eps=eps,
    )

    # this is potentially fixing some minor erroneous edges coming from Voronoi
    final_roads = induce_nodes(final_roads, eps=eps)
    final_roads = final_roads[~final_roads.geometry.normalize().duplicated()].copy()

    return final_roads


def simplify_loop(
    roads: gpd.GeoDataFrame,
    artifacts: gpd.GeoDataFrame,
    max_segment_length: float | int = 1,
    min_dangle_length: float | int = 20,
    clip_limit: float | int = 2,
    simplification_factor: float | int = 2,
    consolidation_tolerance: float | int = 10,
    eps: float = 1e-4,
) -> gpd.GeoDataFrame:
    """Perform an iteration of the simplification procedure which includes:
        1. Removal of false nodes
        2. Artifact classification
        3. Simplifying artifacts:
            - Single artifacts
            - Pairs of artifacts
            - Clusters of artifacts

    Parameters
    ----------
    roads : geopandas.GeoDataFrame
        Raw road network data.
    artifacts : geopandas.GeoDataFrame
        Face artifact polygons.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
        Used in multiple internal geometric operations.
    min_dangle_length : float | int = 20
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    clip_limit : float | int = 2
        Following generation of the Voronoi linework, we clip to fit inside the
        polygon. To ensure we get a space to make proper topological connections
        from the linework to the actual points on the edge of the polygon, we clip
        using a polygon with a negative buffer of ``clip_limit`` or the radius of
        maximum inscribed circle, whichever is smaller.
    simplification_factor : float | int = 2
        The factor by which singles, pairs, and clusters are simplified. The
        ``max_segment_length`` is multiplied by this factor to get the
        simplification epsilon.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation when generating Voronoi skeletons.
    eps : float = 1e-4
        Tolerance epsilon used in multiple internal geometric operations.

    Returns
    -------
    geopandas.GeoDataFrame
        The road network line data following 1 iteration of simplification.
    """

    # Remove edges fully within the artifact (dangles).
    _, r_idx = roads.sindex.query(artifacts.geometry, predicate="contains")
    roads = remove_false_nodes(roads.drop(roads.index[r_idx]))  # drop could cause new

    # Filter singleton artifacts
    rook = graph.Graph.build_contiguity(artifacts, rook=True)

    # keep only those artifacts which occur as isolates, i.e. are not part of a larger
    # intersection
    singles = artifacts.loc[artifacts.index.intersection(rook.isolates)].copy()

    # Filter doubles
    artifacts["comp"] = rook.component_labels
    counts = artifacts["comp"].value_counts()
    doubles = artifacts.loc[artifacts["comp"].isin(counts[counts == 2].index)].copy()

    # Filter clusters
    clusters = artifacts.loc[artifacts["comp"].isin(counts[counts > 2].index)].copy()

    if not singles.empty:
        roads = simplify_singletons(
            singles,
            roads,
            max_segment_length=max_segment_length,
            simplification_factor=simplification_factor,
            consolidation_tolerance=consolidation_tolerance,
        )
    if not doubles.empty:
        roads = simplify_pairs(
            doubles,
            roads,
            max_segment_length=max_segment_length,
            min_dangle_length=min_dangle_length,
            clip_limit=clip_limit,
            simplification_factor=simplification_factor,
            consolidation_tolerance=consolidation_tolerance,
        )
    if not clusters.empty:
        roads = simplify_clusters(
            clusters,
            roads,
            max_segment_length=max_segment_length,
            simplification_factor=simplification_factor,
            eps=eps,
            min_dangle_length=min_dangle_length,
            consolidation_tolerance=consolidation_tolerance,
        )

    return roads
