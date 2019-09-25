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
import tempfile
from pathlib import Path
import pickle

from memory_profiler import memory_usage
import numpy as np

from .utils import (
    get_anndata_memsize,
    sedate,
    get_peak_mem
)

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
        [PBMC_3K_PATH, BM_43K_CSR_PATH, BM_43K_CSC_PATH],
        [False, "r"],
    )
    param_names = ["input_path", "backed"]

    def setup(self, input_path, backed):
        self.adata = anndata.read_h5ad(input_path, backed=backed)

    def peakmem_write(self, input_path, backed):
        self.adata.write_h5ad("./dense.h5ad", force_dense=True)

    def time_write(self, input_path, backed):
        self.adata.write_h5ad("./dense.h5ad", force_dense=True)


class ReadBackedSparse:
    timeout = 300

    params = (
        [PBMC_3K_PATH, BM_43K_CSR_PATH, BM_43K_CSC_PATH],
    )
    param_names = ["input_path"]

    def setup_cache(self):
        cur_path = Path(".")
        local_files = {}
        rows = {}
        cols = {}
        orig_paths = self.params[0]
        for orig in orig_paths:
            new = cur_path / Path(orig).stem
            adata = anndata.read_h5ad(orig)
            justX = anndata.AnnData(X=adata.X)
            justX.write_h5ad(new)
            local_files[orig] = new
            rs = np.random.RandomState(seed=42)
            rows[orig] = rs.choice(adata.shape[0], 100, replace=False)
            cols[orig] = rs.choice(adata.shape[1], 100, replace=False)
        with open("settings.pkl", "wb") as f:
            pickle.dump((local_files, rows, cols), f)

    def setup(self, input_path):
        with open("settings.pkl", "rb") as f:
            local_files, rows, cols = pickle.load(f)
        self.local_file = local_files[input_path]
        self.rows = rows[input_path]
        self.cols = cols[input_path]
        self.adata = anndata.read_h5ad(self.local_file, backed="r")

    def time_read_full_row(self, input_path):
        for row in self.rows:
            result = self.adata[row, :].X

    def peakmem_read_full_row(self, input_path):
        for row in self.rows:
            result = self.adata[row, :].X

    def time_read_full_col(self, input_path):
        for col in self.cols:
            result = self.adata[:, col].X

    def peakmem_read_full_col(self, input_path):
        for col in self.cols:
            result = self.adata[:, col].X

    def time_read_many_rows(self, input_path):
        result = self.adata[self.rows, :].X

    def peakmem_read_many_rows(self, input_path):
        result = self.adata[self.rows, :].X

    def time_read_many_rows_slice(self, input_path):
        start = self.rows[0]
        result = self.adata[start:start+100:2, :].X

    def peakmem_read_many_rows_slice(self, input_path):
        start = self.rows[0]
        result = self.adata[start:start+100:2, :].X

    def time_read_many_cols(self, input_path):
        result = self.adata[:, self.cols].X

    def peakmem_read_many_cols(self, input_path):
        result = self.adata[:, self.cols].X

    def time_read_many_cols_slice(self, input_path):
        start = self.cols[0]
        result = self.adata[:, start:start+100:2].X

    def peakmem_read_many_cols_slice(self, input_path):
        start = self.cols[0]
        result = self.adata[:, start:start+100:2].X
