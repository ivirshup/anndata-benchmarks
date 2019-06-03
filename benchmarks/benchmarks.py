# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from string import ascii_lowercase
from time import sleep

from memory_profiler import memory_usage
import numpy as np
import pandas as pd
from scipy import sparse

# from .helpers import gen_adata

from anndata import AnnData

###############
# Helpers
###############


def get_anndata_memsize(adata):
    # def make_copy(adata):
    #     sleep(0.005)
    #     c = adata.copy()
    #     sleep(0.005)
    # recording = memory_usage((make_copy, (adata,)), interval=0.001)
    recording = memory_usage(
        (sedate(adata.copy, naplength=0.005), (adata,)), interval=0.001)
    diff = recording[-1] - recording[0]
    return diff


def sedate(func, naplength=0.05):
    """Make a function sleepy, so we can sample the start and end state.
    """
    def wrapped_function(*args, **kwargs):
        sleep(naplength)
        val = func(*args, **kwargs)
        sleep(naplength)
        return val
    return wrapped_function

# TODO: Factor out the time it takes to generate these


def gen_indexer(adata, dim, index_kind, ratio):
    dimnames = ("obs", "var")
    axis = dimnames.index(dim)
    subset = [slice(None), slice(None)]
    axis_size = adata.shape[axis]
    if index_kind == "slice":
        subset[axis] = slice(0, int(np.round(axis_size * ratio)))
    elif index_kind == "intarray":
        subset[axis] = np.random.choice(
            np.arange(axis_size), int(np.round(axis_size * ratio)), replace=False)
        subset[axis].sort()
    elif index_kind == "boolarray":
        pos = np.random.choice(
            np.arange(axis_size), int(np.round(axis_size * ratio)), replace=False
        )
        a = np.zeros(axis_size, dtype=bool)
        a[pos] = True
        subset[axis] = a
    else:
        raise ValueError()
    return tuple(subset)


def take_view(adata, *, dim, index_kind, ratio=.5, nviews=100):
    subset = gen_indexer(adata, dim, index_kind, ratio)
    views = []
    for i in range(nviews):
        views.append(adata[subset])


def take_repeated_view(adata, *, dim, index_kind, ratio=.9, nviews=10):
    v = adata
    views = []
    for i in range(nviews):
        subset = gen_indexer(v, dim, index_kind, ratio)
        v = v[subset]
        views.append(v)


def gen_adata(n_obs, n_var, attr_set):
    if "X-csr" in attr_set:
        X = sparse.random(n_obs, n_var, density=0.1, format="csr")
    elif "X-dense" in attr_set:
        X = sparse.random(n_obs, n_var, density=0.1, format="csr")
        X = X.toarray()
    else:
        # TODO: Theres probably a better way to do this
        X = sparse.random(n_obs, n_var, density=0, format="csr")
    adata = AnnData(X)
    if "obs,var" in attr_set:
        adata.obs = pd.DataFrame(
            {k: np.random.randint(0, 100, n_obs) for k in ascii_lowercase},
            index=["cell{}".format(i) for i in range(n_obs)]
        )
        adata.var = pd.DataFrame(
            {k: np.random.randint(0, 100, n_var) for k in ascii_lowercase},
            index=["gene{}".format(i) for i in range(n_var)]
        )
    return adata


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


# This is split off from the previous so we don't time index generation, I think this isn't as big an issue for memory usage
class SubsetTimingSuite:
    params = (
        [100], #[100, 1000],
        [100], #[100, 1000],
        ["X-csr", "X-dense", "obs,var"],
        ["obs", "var"],
        ["intarray", "boolarray", "slice"]
    )
    param_names = ["n_obs", "n_var", "attr_set", "subset_dim", "index_kind"]

    def setup(self, n_obs, n_var, attr_set, subset_dim, index_kind):
        adata = gen_adata(n_obs, n_var, attr_set)
        self.adata = adata
        self.subset = gen_indexer(adata, subset_dim, index_kind, ratio=.5)

    def time_subset(self, n_obs, n_var, attr_set, subset_dim, index_kind):
        v = self.adata[self.subset]
