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
"""
import tempfile
from pathlib import Path

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
