"""
viz/mapper.py
-------------
Mapper graph helper for direct frame-level clouds.
"""
import kmapper as km
import sklearn
import numpy as np


def build_mapper_graph(cloud_direct, out_file="mapper_graph.html", title="Song Mapper Graph"):
    """Build and save a Mapper graph for a direct frame cloud."""
    mapper = km.KeplerMapper(verbose=1)

    lens = mapper.fit_transform(cloud_direct, projection=sklearn.decomposition.PCA(n_components=2))
    clusterer = sklearn.cluster.DBSCAN(eps=0.5, min_samples=3)
    graph = mapper.map(
        lens,
        cloud_direct,
        cover=km.Cover(n_cubes=10, perc_overlap=0.3),
        clusterer=clusterer
    )

    mapper.visualize(
        graph,
        path_html=out_file,
        title=title,
        custom_tooltips=np.arange(len(cloud_direct))
    )
    return graph
