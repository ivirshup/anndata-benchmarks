# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from memory_profiler import memory_usage
import numpy as np

from .utils import (
    get_anndata_memsize,
    sedate,
    gen_indexer, gen_adata,
    take_view, take_repeated_view
)


class SubsetMemorySuite:
    """
    Parameterized suite for testing memory usage of subsetting.
    """

    params = (
        [100, 1000],
        [100, 1000],
        ["X-csr", "X-dense", "obs,var"],
        ["obs", "var"],
        ["intarray", "boolarray", "slice"]
    )
    param_names = ["n_obs", "n_var", "attr_set", "subset_dim", "index_kind"]

    def setup(self, n_obs, n_var, attr_set, subset_dim, index_kind):
        adata = gen_adata(n_obs, n_var, attr_set)
        self.adata = adata

    def track_subset_memratio(self, n_obs, n_var, attr_set, subset_dim, index_kind):
        nviews = 5
        base_size = get_anndata_memsize(self.adata)
        mem_recording = memory_usage(
            (
                sedate(take_view, .005),
                (self.adata,),
                {"dim": subset_dim, "index_kind": index_kind, "nviews": nviews}
            ),
            interval=.001
        )
        return ((np.max(mem_recording) - np.min(mem_recording)) / nviews) / base_size

    def track_repeated_subset_memratio(self, n_obs, n_var, attr_set, subset_dim, index_kind):
        nviews = 5
        base_size = get_anndata_memsize(self.adata)
        mem_recording = memory_usage(
            (
                sedate(take_repeated_view, .005),
                (self.adata,),
                {"dim": subset_dim, "index_kind": index_kind,
                    "nviews": nviews, "ratio": 0.9}
            ),
            interval=.001
        )
        return (np.max(mem_recording) - np.min(mem_recording)) / base_size


# TODO: Add test for 1 cell
# TODO: Test subseting csc
# TODO: Add case for backed
# This is split off from the previous so we don't time index generation, I think this isn't as big an issue for memory usage
class SubsetTimingSuite:
    params = (
        [100, 1000],
        [100, 1000],
        ["X-csr", "X-dense", "obs,var"],
        ["obs", "var"],
        ["intarray", "boolarray", "slice", "strarray"]
    )
    param_names = ["n_obs", "n_var", "attr_set", "subset_dim", "index_kind"]

    def setup(self, n_obs, n_var, attr_set, subset_dim, index_kind):
        adata = gen_adata(n_obs, n_var, attr_set)
        self.adata = adata
        self.subset = gen_indexer(adata, subset_dim, index_kind, ratio=.5)

    def time_subset(self, n_obs, n_var, attr_set, subset_dim, index_kind):
        v = self.adata[self.subset]

    def time_subset_X(self, n_obs, n_var, attr_set, subset_dim, index_kind):
        Xv = self.adata[self.subset].X

    def time_copy_subset(self, n_obs, n_var, attr_set, subset_dim, index_kind):
        c = self.adata[self.subset].copy()
