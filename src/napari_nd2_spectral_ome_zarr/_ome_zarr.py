from __future__ import annotations

from pathlib import Path

import dask.array as da
import numcodecs
import numpy as np
import zarr
from ome_zarr.writer import write_multiscale
from skimage.transform import downscale_local_mean


def _build_multiscales(data_tczyx: np.ndarray) -> list[da.Array]:
    multiscales: list[da.Array] = []
    current = np.asarray(data_tczyx)
    multiscales.append(
        da.from_array(current, chunks=(1, min(current.shape[1], 16), 1, min(current.shape[3], 256), min(current.shape[4], 256)))
    )

    for factor in (2, 4):
        if min(current.shape[-2:]) < factor:
            break
        reduced = downscale_local_mean(data_tczyx, factors=(1, 1, 1, factor, factor))
        multiscales.append(
            da.from_array(
                reduced.astype(data_tczyx.dtype, copy=False),
                chunks=(1, min(reduced.shape[1], 16), 1, min(reduced.shape[3], 256), min(reduced.shape[4], 256)),
            )
        )
    return multiscales


def export_dataset_to_ome_zarr(
    data_tczyx: np.ndarray,
    output_path: str,
    metadata: dict,
) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    store = zarr.DirectoryStore(str(output))
    group = zarr.group(store=store, overwrite=True)
    write_multiscale(
        _build_multiscales(data_tczyx),
        group,
        axes=["t", "c", "z", "y", "x"],
        storage_options={"compressor": numcodecs.Blosc(cname="zstd", clevel=5)},
    )
    group.attrs.update(metadata)
    return str(output)
