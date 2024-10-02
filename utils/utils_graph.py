import multiprocessing
import numpy as np
from node2vec import Node2Vec
from sklearn.neighbors import NearestNeighbors
import json
import networkx as nx
from utils import ontology

try:
    import importlib.resources as importlib_resources
except ModuleNotFoundError:
    import importlib_resources


class GraphEmb:
    def __init__(
        self,
        embed_size: int = 128,
        num_walks: int = 10,
        walk_length: int = 10,
        workers=1,
        **kwargs
    ):
        self.embed_size = embed_size
        self.workers = workers if workers != -1 else multiprocessing.cpu_count()
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.node2vec_kwargs = kwargs
        self.node2vec = None
        self.code_hierarchy, self.codes = self.comorb_tree()

    def comorb_tree(self):
        with importlib_resources.open_text(ontology, f"icd10cm_tree.json") as f:
            hierarchy = json.loads(f.read())
        return (
            nx.readwrite.json_graph.tree_graph(hierarchy["tree"]),
            hierarchy["codes"],
        )

    def train_embedding(self, **kwargs):
        self.node2vec = Node2Vec(
            self.code_hierarchy,
            dimensions=self.embed_size,
            workers=self.workers,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            quiet=True,
            **self.node2vec_kwargs
        ).fit(window=4, min_count=1, **kwargs)
        self.nn = NearestNeighbors(n_neighbors=1)
        self.nn.fit(self.to_vec(self.codes))

    def to_vec(self, icd_codes):
        return np.stack([
            self.node2vec.wv.get_vector(icd_code)
            for icd_code in icd_codes
        ])

