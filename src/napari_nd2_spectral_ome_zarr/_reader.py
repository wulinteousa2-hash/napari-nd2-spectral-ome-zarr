from __future__ import annotations

import os
from pathlib import Path

import dask.array as da
import numpy as np
import numcodecs
import zarr

from ._nd2 import get_first_2d_spectral_plane, load_nd2_dataset
from ._spectral import render_visible_truecolor

numcodecs.blosc.set_nthreads(max(1, min(8, os.cpu_count() or 1)))


def napari_get_reader(path):
    paths = path if isinstance(path, list) else [path]
    if not all(_is_supported_path(str(item)) for item in paths):
        return None
    return _reader


def _reader(path):
    input_path = path[0] if isinstance(path, list) else path
    return build_layer_data(str(input_path))


def _is_supported_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(".nd2") or lowered.endswith(".zarr")


def _read_ome_zarr(path: str) -> tuple[da.Array, np.ndarray, dict, da.Array]:
    group = zarr.open_group(path, mode="r")
    if "multiscales" in group.attrs:
        datasets = group.attrs["multiscales"][0]["datasets"]
        array_paths = [dataset["path"] for dataset in datasets]
    else:
        array_paths = ["0"]

    full_res = da.from_zarr(group[array_paths[0]])
    preview_array = da.from_zarr(group[array_paths[-1]])

    wavelengths = np.asarray(group.attrs.get("wavelengths_nm", np.linspace(400.0, 740.0, full_res.shape[1])), dtype=np.float32)
    metadata = dict(group.attrs)
    metadata["path"] = path
    metadata.setdefault("axes", ["T", "C", "Z", "Y", "X"])
    metadata.setdefault("is_spectral", bool(full_res.shape[1] > 4))
    return full_res, wavelengths, metadata, preview_array


def inspect_ome_zarr(path: str) -> dict:
    data_tczyx, wavelengths_nm, metadata, preview_array = _read_ome_zarr(path)
    return {
        "path": path,
        "name": Path(path).name,
        "axes": list(metadata.get("axes", ["T", "C", "Z", "Y", "X"])),
        "shape": tuple(int(value) for value in data_tczyx.shape),
        "preview_shape": tuple(int(value) for value in preview_array.shape),
        "wavelength_count": int(wavelengths_nm.size),
        "wavelength_min_nm": float(wavelengths_nm.min()) if wavelengths_nm.size else None,
        "wavelength_max_nm": float(wavelengths_nm.max()) if wavelengths_nm.size else None,
        "is_spectral": bool(metadata.get("is_spectral", False)),
    }


def _build_layer_payload(
    file_stem: str,
    spectral_plane,
    data_tczyx,
    wavelengths_nm: np.ndarray,
    metadata: dict,
    use_gpu: bool,
    include_visible_layer: bool = True,
    include_truecolor_layer: bool = True,
    include_raw_layer: bool = False,
) -> list:
    layer_data = []
    shared_metadata = {
        "source_path": metadata.get("path"),
        "dataset_metadata": metadata,
        "wavelengths_nm": wavelengths_nm.tolist(),
        "spectral_cube": spectral_plane,
    }

    if include_raw_layer:
        layer_data.append(
            (
                spectral_plane,
                {
                    "name": f"{file_stem} spectral",
                    "metadata": shared_metadata,
                    "channel_axis": 0,
                },
                "image",
            )
        )

    if metadata["is_spectral"] and (include_visible_layer or include_truecolor_layer):
        merged, rgb = render_visible_truecolor(spectral_plane, wavelengths_nm, use_gpu=use_gpu)
        if include_visible_layer:
            layer_data.append(
                (
                    merged,
                    {
                        "name": f"{file_stem} visible sum",
                        "metadata": shared_metadata,
                        "colormap": "gray",
                    },
                    "image",
                )
            )
        if include_truecolor_layer:
            layer_data.append(
                (
                    rgb,
                    {
                        "name": f"{file_stem} truecolor",
                        "metadata": shared_metadata,
                        "rgb": True,
                    },
                    "image",
                )
            )

    return layer_data


def build_layer_data(
    input_path: str,
    use_gpu: bool = False,
    include_visible_layer: bool = True,
    include_truecolor_layer: bool = True,
    include_raw_layer: bool = False,
    zarr_use_preview: bool = True,
):
    file_stem = Path(input_path).stem
    if input_path.lower().endswith(".nd2"):
        dataset = load_nd2_dataset(input_path)
        return _build_layer_payload(
            file_stem=file_stem,
            spectral_plane=get_first_2d_spectral_plane(dataset),
            data_tczyx=dataset.data_tczyx,
            wavelengths_nm=dataset.wavelengths_nm,
            metadata=dataset.metadata,
            use_gpu=use_gpu,
            include_visible_layer=include_visible_layer,
            include_truecolor_layer=include_truecolor_layer,
            include_raw_layer=include_raw_layer,
        )

    if input_path.lower().endswith(".zarr"):
        data_tczyx, wavelengths_nm, metadata, preview_data = _read_ome_zarr(input_path)
        spectral_plane = data_tczyx[0, :, 0, :, :]
        render_plane = np.asarray(preview_data[0, :, 0, :, :]) if zarr_use_preview else np.asarray(spectral_plane)
        layer_data = _build_layer_payload(
            file_stem=file_stem,
            spectral_plane=spectral_plane,
            data_tczyx=data_tczyx,
            wavelengths_nm=wavelengths_nm,
            metadata=metadata,
            use_gpu=False,
            include_visible_layer=False,
            include_truecolor_layer=False,
            include_raw_layer=include_raw_layer,
        )
        if metadata["is_spectral"] and (include_visible_layer or include_truecolor_layer):
            merged, rgb = render_visible_truecolor(render_plane, wavelengths_nm, use_gpu=use_gpu)
            shared_metadata = {
                "source_path": metadata.get("path"),
                "dataset_metadata": metadata,
                "wavelengths_nm": wavelengths_nm.tolist(),
                "spectral_cube": spectral_plane,
            }
            suffix = " preview" if zarr_use_preview else ""
            if include_visible_layer:
                layer_data.append(
                    (
                        merged,
                        {
                            "name": f"{file_stem} visible sum{suffix}",
                            "metadata": shared_metadata,
                            "colormap": "gray",
                        },
                        "image",
                    )
                )
            if include_truecolor_layer:
                layer_data.append(
                    (
                        rgb,
                        {
                            "name": f"{file_stem} truecolor{suffix}",
                            "metadata": shared_metadata,
                            "rgb": True,
                        },
                        "image",
                    )
                )
        return layer_data

    raise ValueError(f"Unsupported path: {input_path}")
