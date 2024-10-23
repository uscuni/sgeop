import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import shapely
from scipy import sparse


def split(
    split_points: list | np.ndarray | gpd.GeoSeries,
    cleaned_roads: gpd.GeoDataFrame,
    crs: str | pyproj.CRS,
    eps: float = 1e-4,
) -> gpd.GeoSeries | gpd.GeoDataFrame:
    """Split lines on new nodes.

    Parameters
    ----------
    split_points : list | numpy.ndarray
        Points to split the ``cleaned_roads``.
    cleaned_roads : geopandas.GeoDataFrame
        Line geometries to be split with ``split_points``.
    crs : str | pyproj.CRS
        Anything accepted by ``pyproj.CRS``.
    eps : float
        Tolerance epsilon for point snapping.

    Returns
    -------
    geopandas.GeoSeries | geopandas.GeoDataFrame
        Resultant split line geometries.
    """
    split_points = gpd.GeoSeries(split_points, crs=crs)
    for split in split_points.drop_duplicates():
        _, ix = cleaned_roads.sindex.nearest(split, max_distance=eps)
        edge = cleaned_roads.geometry.iloc[ix]
        if edge.shape[0] == 1:
            lines_split = _snap_n_split(edge.item(), split, eps)
            if lines_split.shape[0] > 1:
                gdf_split = gpd.GeoDataFrame(geometry=lines_split, crs=crs)
                gdf_split["_status"] = "changed"
                cleaned_roads = pd.concat(
                    [cleaned_roads.drop(edge.index[0]), gdf_split],
                    ignore_index=True,
                )
        else:
            to_be_dropped = []
            to_be_added = []
            for i, e in edge.items():
                lines_split = _snap_n_split(e, split, eps)
                if lines_split.shape[0] > 1:
                    to_be_dropped.append(i)
                    to_be_added.append(lines_split)

            if to_be_added:
                gdf_split = gpd.GeoDataFrame(
                    geometry=np.concatenate(to_be_added), crs=crs
                )
                gdf_split["_status"] = "changed"
                cleaned_roads = pd.concat(
                    [cleaned_roads.drop(to_be_dropped), gdf_split],
                    ignore_index=True,
                )

    return cleaned_roads.reset_index(drop=True)


def _snap_n_split(e: shapely.LineString, s: shapely.Point, tol: float) -> np.ndarray:
    """Snap point to edge and return lines to split."""
    snapped = shapely.snap(e, s, tolerance=tol)
    _lines_split = shapely.get_parts(shapely.ops.split(snapped, s))
    return _lines_split[~shapely.is_empty(_lines_split)]


def _status(x: pd.Series) -> str:
    """Determine the status of edge line(s)."""
    if len(x) == 1:
        return x.iloc[0]
    if "new" in x:
        return "new"
    return "changed"


def get_components(
    edgelines: list | np.ndarray | gpd.GeoSeries,
    ignore: None | gpd.GeoSeries = None,
) -> np.ndarray:
    """Associate edges with connected component labels and return.

    Parameters
    ----------
    edgelines : list | np.ndarray | gpd.GeoSeries
        Collection of line objects.
    ignore : None | gpd.GeoSeries = None
        Nodes to ignore when labeling components.

    Returns
    -------
    np.ndarray
        Edge connected component labels.

    Notes
    -----
    See [https://github.com/uscuni/sgeop/issues/56] for detailed explanation of output.
    """
    edgelines = np.array(edgelines)
    start_points = shapely.get_point(edgelines, 0)
    end_points = shapely.get_point(edgelines, -1)
    points = shapely.points(
        np.unique(
            shapely.get_coordinates(np.concatenate([start_points, end_points])), axis=0
        )
    )
    if ignore is not None:
        mask = np.isin(points, ignore)
        points = points[~mask]
    # query LineString geometry to identify points intersecting 2 geometries
    inp, res = shapely.STRtree(shapely.boundary(edgelines)).query(
        points, predicate="intersects"
    )
    unique, counts = np.unique(inp, return_counts=True)
    mask = np.isin(inp, unique[counts == 2])
    merge_res = res[mask]
    merge_inp = inp[mask]
    closed = np.arange(len(edgelines))[shapely.is_closed(edgelines)]
    mask = np.isin(merge_res, closed) | np.isin(merge_inp, closed)
    merge_res = merge_res[~mask]
    merge_inp = merge_inp[~mask]
    g = nx.Graph(list(zip((merge_inp * -1) - 1, merge_res, strict=True)))
    components = {
        i: {v for v in k if v > -1} for i, k in enumerate(nx.connected_components(g))
    }
    component_labels = {value: key for key in components for value in components[key]}
    labels = pd.Series(component_labels, index=range(len(edgelines)))

    max_label = len(edgelines) - 1 if pd.isna(labels.max()) else int(labels.max())
    filling = pd.Series(range(max_label + 1, max_label + len(edgelines) + 1))
    labels = labels.fillna(filling)

    return labels.values


def weld_edges(
    edgelines: list | np.ndarray | gpd.GeoSeries,
    ignore: None | gpd.GeoSeries = None,
) -> list:
    """Combine lines sharing an endpoint (if only 2 lines share that point).
    Lightweight version of ``remove_false_nodes()``.

    Parameters
    ----------
    edgelines : list | np.ndarray | gpd.GeoSeries
        Collection of line objects.
    ignore : None | gpd.GeoSeries = None
        Nodes to ignore when welding components.

    Returns
    -------
    list | np.ndarray | gpd.GeoSeries
        Resultant welded ``edgelines`` if more than 1 passed in, otherwise
        the original ``edgelines`` object.
    """
    if len(edgelines) < 2:
        return edgelines
    labels = get_components(edgelines, ignore=ignore)
    return (
        gpd.GeoSeries(edgelines)
        .groupby(labels)
        .agg(lambda x: shapely.line_merge(shapely.GeometryCollection(x.values)))
    ).tolist()


def remove_false_nodes(gdf, aggfunc="first", **kwargs):
    """Reimplementation of momepy.remove_false_nodes that preserves attributes

    Parameters
    ----------
    gdf : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if len(gdf) < 2:
        return gdf

    if isinstance(gdf, gpd.GeoSeries):
        gdf = gdf.to_frame("geometry")

    labels = get_components(gdf.geometry)

    # Process non-spatial component
    data = gdf.drop(labels=gdf.geometry.name, axis=1)
    aggregated_data = data.groupby(by=labels).agg(aggfunc, **kwargs)
    aggregated_data.columns = aggregated_data.columns.to_flat_index()

    # Process spatial component
    def merge_geometries(block):
        merged_geom = shapely.line_merge(shapely.GeometryCollection(block.values))
        return merged_geom

    g = gdf.groupby(group_keys=False, by=labels)[gdf.geometry.name].agg(
        merge_geometries
    )
    aggregated_geometry = gpd.GeoDataFrame(g, geometry=gdf.geometry.name, crs=gdf.crs)
    # Recombine
    aggregated = aggregated_geometry.join(aggregated_data)

    nodes = momepy.nx_to_gdf(
        momepy.node_degree(momepy.gdf_to_nx(aggregated[[aggregated.geometry.name]])),
        lines=False,
    )
    loop_mask = aggregated.is_ring
    loops = aggregated[loop_mask]

    fixed_loops = []
    fixed_index = []
    node_ix, loop_ix = loops.sindex.query(nodes.geometry, predicate="intersects")
    for ix in np.unique(loop_ix):
        loop_geom = loops.geometry.iloc[ix]
        target_nodes = nodes.geometry.iloc[node_ix[loop_ix == ix]]
        if len(target_nodes) == 2:
            node_coords = shapely.get_coordinates(target_nodes)
            coords = np.array(loop_geom.coords)
            new_start = (
                node_coords[0]
                if (node_coords[0] != coords[0]).all()
                else node_coords[1]
            )
            new_start_idx = np.where(coords == new_start)[0][0]
            rolled_coords = np.roll(coords[:-1], -new_start_idx, axis=0)
            new_sequence = np.append(rolled_coords, rolled_coords[[0]], axis=0)
            fixed_loops.append(shapely.LineString(new_sequence))
            fixed_index.append(ix)

    aggregated.loc[loops.index[fixed_index], aggregated.geometry.name] = fixed_loops
    return aggregated


def fix_topology(roads, eps=1e-4, **kwargs):
    """Fix road network topology

    This ensures correct topology of the network by:

    1.  adding potentially missing nodes
    on intersections of individual LineString endpoints with the remaining network. The
    idea behind is that if a line ends on an intersection with another, there should be
    a node on both of them.

    2. removing nodes of degree 2 that have no meaning in the network
    used within our framework.

    3. removing duplicated geometries (irrespective of orientation).
    """
    roads = roads[~roads.geometry.normalize().duplicated()].copy()
    roads_w_nodes = induce_nodes(roads, eps=eps)
    return remove_false_nodes(roads_w_nodes, **kwargs)


def induce_nodes(roads, eps=1e-4):
    """
    adding potentially missing nodes
    on intersections of individual LineString endpoints with the remaining network. The
    idea behind is that if a line ends on an intersection with another, there should be
    a node on both of them.
    """
    nodes_w_degree = momepy.nx_to_gdf(
        momepy.node_degree(momepy.gdf_to_nx(roads)), lines=False
    )
    nodes_ix, roads_ix = roads.sindex.query(
        nodes_w_degree.geometry, predicate="dwithin", distance=1e-4
    )
    intersects = sparse.coo_array(
        ([True] * len(nodes_ix), (nodes_ix, roads_ix)),
        shape=(len(nodes_w_degree), len(roads)),
        dtype=np.bool_,
    )
    nodes_w_degree["expected_degree"] = intersects.sum(axis=1)
    nodes_to_induce = nodes_w_degree[
        nodes_w_degree.degree != nodes_w_degree.expected_degree
    ]
    return split(nodes_to_induce.geometry, roads, roads.crs, eps=eps)


def consolidate_nodes(gdf, tolerance=2, preserve_ends=False):
    """Return geometry with consolidated nodes.

    Replace clusters of nodes with a single node (weighted centroid
    of a cluster) and snap linestring geometry to it. Cluster is
    defined using DBSCAN on coordinates with ``tolerance``==``eps`.

    Does not preserve any attributes, function is purely geometric.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with LineStrings (usually representing street network)
    tolerance : float
        The maximum distance between two nodes for one to be considered
        as in the neighborhood of the other. Nodes within tolerance are
        considered a part of a single cluster and will be consolidated.
    preserve_ends : bool
        If True, nodes of a degree 1 will be excluded from the consolidation

    Returns
    -------
    GeoSeries
    """
    # TODO: this should not dumbly merge all nodes within the cluster to a single
    # TODO: centroid but iteratively - do the two nearest and add other only if the
    # TODO: distance is still below the tolerance

    # TODO: make it work on GeoDataFrames preserving attributes
    from sklearn.cluster import DBSCAN

    if isinstance(gdf, gpd.GeoSeries):
        gdf = gdf.to_frame("geometry")
    elif isinstance(gdf, np.ndarray):
        gdf = gpd.GeoDataFrame(geometry=gdf)

    nodes = momepy.nx_to_gdf(momepy.node_degree(momepy.gdf_to_nx(gdf)), lines=False)

    if preserve_ends:
        ends = nodes[nodes.degree == 1].buffer(
            1
        )  # keep at least one meter of original geometry around each end
        nodes = nodes[nodes.degree > 1].copy()

        # if all we have are ends, return the original
        if nodes.empty:
            gdf["_status"] = "original"
            return gdf

    # get clusters of nodes which should be consolidated
    db = DBSCAN(eps=tolerance, min_samples=2).fit(nodes.get_coordinates())
    nodes["lab"] = db.labels_
    change = nodes[nodes.lab > -1]

    # no change needed, return the original
    if change.empty:
        gdf["_status"] = "original"
        return gdf

    gdf = gdf.copy()
    # get geometry
    geom = gdf.geometry.copy()
    status = pd.Series("original", index=geom.index)

    # loop over clusters, cut out geometry within tolerance / 2 and replace it
    # with spider-like geometry to the weighted centroid of a cluster
    spiders = []
    midpoints = []

    clusters = change.dissolve(change.lab)
    cookies = clusters.buffer(
        tolerance / 2
    ).convex_hull  # TODO: not optimal but avoids some MultiLineStrings but not all
    if preserve_ends:
        cookies = cookies.to_frame().overlay(ends.to_frame(), how="difference")

    for cluster, cookie in zip(clusters.geometry, cookies.geometry, strict=True):
        inds = geom.sindex.query(cookie, predicate="intersects")
        pts = geom.iloc[inds].intersection(cookie.boundary).get_coordinates()
        pts = shapely.get_coordinates(geom.iloc[inds].intersection(cookie.boundary))
        if pts.shape[0] > 0:
            geom.iloc[inds] = geom.iloc[inds].difference(
                cookie
            )  # TODO: this may result in MultiLineString - we need to avoid that
            # TODO: It is temporarily fixed by that explode in return
            status.iloc[inds] = "snapped"
            midpoint = np.mean(shapely.get_coordinates(cluster), axis=0)
            midpoints.append(midpoint)
            mids = np.array(
                [
                    midpoint,
                ]
                * len(pts)
            )
            spider = shapely.linestrings(
                np.array([pts[:, 0], mids[:, 0]]).T,
                y=np.array([pts[:, 1], mids[:, 1]]).T,
            )
            spiders.append(spider)

    gdf = gdf.set_geometry(geom)
    gdf["_status"] = status

    if spiders:
        # combine geometries
        gdf = pd.concat(
            [
                gdf,
                gpd.GeoDataFrame(geometry=np.hstack(spiders), crs=geom.crs),
            ]
        )

    return remove_false_nodes(
        gdf[~gdf.geometry.is_empty].explode(),
        aggfunc={"_status": _status},
    )
