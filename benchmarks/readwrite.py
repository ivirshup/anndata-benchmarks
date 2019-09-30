"""
This module will benchmark io of AnnData objects

Things to test:

* Read time, write time
* Peak memory during io
* File sizes

Parameterized by:

* What method is being used
* What data is being included
* Size of data being used

Also interesting:

* io for views
* io for backed objects
* Reading dense as sparse, writing sparse as dense
"""
from itertools import product
import tempfile
from pathlib import Path
import pickle

from memory_profiler import memory_usage
import numpy as np
import pandas as pd
from scipy import sparse

from .utils import (
    get_anndata_memsize,
    sedate,
    get_peak_mem
)
from . import datasets

import anndata

PBMC_REDUCED_PATH = "/Users/isaac/github/scanpy/scanpy/datasets/10x_pbmc68k_reduced.h5ad"
PBMC_3K_PATH = "/Users/isaac/data/pbmc3k_raw.h5ad"
BM_43K_CSR_PATH = Path(__file__).parent.parent / "datasets/BM2_43k-cells.h5ad"
BM_43K_CSC_PATH = Path(__file__).parent.parent / "datasets/BM2_43k-cells_CSC.h5ad"


class H5ADReadSuite:
    params = (
        [PBMC_REDUCED_PATH, PBMC_3K_PATH]
    )
    param_names = ["input_path"]

    def setup(self, input_path):
        self.filepath = input_path

    def time_read_full(self, input_path):
        anndata.read_h5ad(self.filepath)

    def peakmem_read_full(self, input_path):
        anndata.read_h5ad(self.filepath)

    def mem_readfull_object(self, input_path):
        return anndata.read_h5ad(self.filepath)

    def track_read_full_memratio(self, input_path):
        mem_recording = memory_usage(
            (
                sedate(anndata.read_h5ad, .005),
                (self.filepath,)
            ),
            interval=.001
        )
        adata = anndata.read_h5ad(self.filepath)
        base_size = mem_recording[-1] - mem_recording[0]
        print(np.max(mem_recording) - np.min(mem_recording))
        print(base_size)
        return (np.max(mem_recording) - np.min(mem_recording)) / base_size

    def peakmem_read_backed(self, input_path):
        anndata.read_h5ad(self.filepath, backed="r")

    def mem_read_backed_object(self, input_path):
        return anndata.read_h5ad(self.filepath, backed="r")


class H5ADWriteSuite:
    params = (
        [PBMC_REDUCED_PATH, PBMC_3K_PATH]
    )
    param_names = ["input_path"]

    def setup(self, input_path):
        mem_recording, adata = memory_usage(
            (
                sedate(anndata.read_h5ad, .005),
                (input_path,)
            ),
            retval=True,
            interval=.001
        )
        self.adata = adata
        self.base_size = mem_recording[-1] - mem_recording[0]
        self.tmpdir = tempfile.TemporaryDirectory()
        self.writepth = Path(self.tmpdir.name) / "out.h5ad"

    def teardown(self, input_path):
        self.tmpdir.cleanup()

    def time_write_full(self, input_path):
        self.adata.write_h5ad(self.writepth, compression=None)

    def peakmem_write_full(self, input_path):
        self.adata.write_h5ad(self.writepth)

    def track_peakmem_write_full(self, input_path):
        return get_peak_mem((sedate(self.adata.write_h5ad), (self.writepth,)))

    def time_write_compressed(self, input_path):
        self.adata.write_h5ad(self.writepth, compression="gzip")

    def peakmem_write_compressed(self, input_path):
        self.adata.write_h5ad(self.writepth, compression="gzip")

    def track_peakmem_write_compressed(self, input_path):
        return get_peak_mem((sedate(self.adata.write_h5ad), (self.writepth,), {"compression": "gzip"}))


class H5ADBackedWriteSuite(H5ADWriteSuite):
    params = (
        [PBMC_REDUCED_PATH, PBMC_3K_PATH]
    )
    param_names = ["input_path"]

    def setup(self, input_path):
        mem_recording, adata = memory_usage(
            (
                sedate(anndata.read_h5ad, .005),
                (input_path,),
                {"backed": "r"}
            ),
            retval=True,
            interval=.001
        )
        self.adata = adata
        self.base_size = mem_recording[-1] - mem_recording[0]
        self.tmpdir = tempfile.TemporaryDirectory()
        self.writepth = Path(self.tmpdir.name) / "out.h5ad"


class WriteSparseAsDense:
    timeout = 300

    params = (
        datasets.list_available(),
        ["csc", "csr"],
        [False, "r"],
    )
    param_names = ["dataset", "mtx_format", "backed"]

    def setup_cache(self):
        base_path = Path(".")
        fmt_map = {
            "csc": sparse.csc_matrix,
            "csr": sparse.csr_matrix
        }
        cache_map = {}
        for dataset in self.params[0]:
            orig = dataset.load()
            for mtx_format in self.params[1]:
                cur = orig.copy()
                cur_path = base_path / f"{dataset.name}_{mtx_format}.h5ad"
                cur.X = fmt_map[mtx_format](orig.X)
                cur.write_h5ad(cur_path)
                cache_map[(dataset.name, mtx_format)] = cur_path
        pd.to_pickle(cache_map, "cache_map.pkl")

    def setup(self, dataset, mtx_format, backed):
        cache_map = pd.read_pickle("cache_map.pkl")
        file_path = cache_map[(dataset.name, mtx_format)]
        self.adata = anndata.read_h5ad(file_path, backed=backed)

    def peakmem_write(self, dataset, mtx_format, backed):
        self.adata.write_h5ad("./dense.h5ad", force_dense=True)

    def time_write(self, dataset, mtx_format, backed):
        self.adata.write_h5ad("./dense.h5ad", force_dense=True)


class ReadBackedSparse:
    timeout = 300

    params = (
        datasets.list_available(),
        ["csr", "csc"]
    )
    param_names = ["dataset", "mtx_fmt"]

    def setup_cache(self):
        cur_path = Path(".")
        local_files = {}
        rows = {}
        cols = {}
        fmt_map = {
            "csc": sparse.csc_matrix,
            "csr": sparse.csr_matrix
        }
        for dset, mtx_fmt in product(*self.params):
            ad_pth = cur_path / f"{dset.name}_{mtx_fmt}.h5ad"
            adata = dset.load()
            justX = anndata.AnnData(
                X=fmt_map[mtx_fmt](adata.X),
                obs=pd.DataFrame([], index=adata.obs_names),
                var=pd.DataFrame([], index=adata.var_names),
            )
            justX.write(ad_pth)
            local_files[(dset.name, mtx_fmt)] = ad_pth
            rs = np.random.RandomState(seed=42)
            rows[(dset.name, mtx_fmt)] = rs.choice(adata.shape[0], 10, replace=False)
            cols[(dset.name, mtx_fmt)] = rs.choice(adata.shape[1], 10, replace=False)
        with open("settings.pkl", "wb") as f:
            pickle.dump((local_files, rows, cols), f)

    def setup(self, dset, mtx_fmt):
        with open("settings.pkl", "rb") as f:
            local_files, rows, cols = pickle.load(f)
        self.local_file = local_files[(dset.name, mtx_fmt)]
        self.rows = rows[(dset.name, mtx_fmt)]
        self.cols = cols[(dset.name, mtx_fmt)]
        self.adata = anndata.read_h5ad(self.local_file, backed="r")

    def time_read_full_row(self, dset, mtx_fmt):
        for row in self.rows:
            result = self.adata[row, :].X

    def peakmem_read_full_row(self, dset, mtx_fmt):
        for row in self.rows:
            result = self.adata[row, :].X

    def time_read_full_col(self, dset, mtx_fmt):
        for col in self.cols:
            result = self.adata[:, col].X

    def peakmem_read_full_col(self, dset, mtx_fmt):
        for col in self.cols:
            result = self.adata[:, col].X

    def time_read_many_rows(self, dset, mtx_fmt):
        result = self.adata[self.rows, :].X

    def peakmem_read_many_rows(self, dset, mtx_fmt):
        result = self.adata[self.rows, :].X

    def time_read_many_rows_slice(self, dset, mtx_fmt):
        start = self.rows[0]
        result = self.adata[start:start+100:2, :].X

    def peakmem_read_many_rows_slice(self, dset, mtx_fmt):
        start = self.rows[0]
        result = self.adata[start:start+100:2, :].X

    def time_read_many_cols(self, dset, mtx_fmt):
        result = self.adata[:, self.cols].X

    def peakmem_read_many_cols(self, dset, mtx_fmt):
        result = self.adata[:, self.cols].X

    def time_read_many_cols_slice(self, dset, mtx_fmt):
        start = self.cols[0]
        result = self.adata[:, start:start+100:2].X

    def peakmem_read_many_cols_slice(self, dset, mtx_fmt):
        start = self.cols[0]
        result = self.adata[:, start:start+100:2].X
