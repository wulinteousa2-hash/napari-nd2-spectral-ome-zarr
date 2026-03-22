from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nd2
import numpy as np


@dataclass
class Nd2Dataset:
    path: str
    data_tczyx: np.ndarray
    wavelengths_nm: np.ndarray
    metadata: dict


def _open_nd2(path: str) -> nd2.ND2File:
    try:
        handle = nd2.ND2File(path)
        _ = handle.events()
        return handle
    except AttributeError:
        return nd2.ND2File(path, legacy=True)


def _canonical_axis_name(axis_name: str) -> str:
    return {
        "Time": "T",
        "Channel": "C",
        "Z": "Z",
        "Y": "Y",
        "X": "X",
    }.get(axis_name, axis_name)


def _normalize_to_tczyx(data: np.ndarray, sizes: dict) -> np.ndarray:
    axes = [_canonical_axis_name(axis) for axis in sizes.keys()]
    normalized = np.asarray(data)

    unsupported = [axis for axis in axes if axis not in {"T", "C", "Z", "Y", "X"}]
    for axis in unsupported:
        axis_index = axes.index(axis)
        if normalized.shape[axis_index] != 1:
            raise ValueError(f"Unsupported ND2 axis {axis!r} with size {normalized.shape[axis_index]}")
    for axis in reversed(unsupported):
        axis_index = axes.index(axis)
        normalized = np.take(normalized, indices=0, axis=axis_index)
        axes.pop(axis_index)

    for axis in ["T", "C", "Z", "Y", "X"]:
        if axis not in axes:
            normalized = np.expand_dims(normalized, axis=0)
            axes.insert(0, axis)

    permutation = [axes.index(axis) for axis in ["T", "C", "Z", "Y", "X"]]
    return np.transpose(normalized, axes=permutation)


def _extract_wavelengths(handle: nd2.ND2File, channel_count: int) -> np.ndarray:
    metadata_channels = getattr(handle.metadata, "channels", []) or []
    emission = []
    for channel in metadata_channels:
        wavelength = getattr(getattr(channel, "channel", None), "emissionLambdaNm", None)
        if wavelength:
            emission.append(float(wavelength))
    if len(emission) == channel_count:
        return np.asarray(emission, dtype=np.float32)

    description = getattr(handle, "text_info", {}).get("description", "")
    match = re.search(r"{Si Grating Resolution}: ([\\d.]+)", description)
    step = float(match.group(1)) if match else None
    if emission and step:
        start = min(emission)
        return np.asarray([start + idx * step for idx in range(channel_count)], dtype=np.float32)

    return np.linspace(400.0, 740.0, channel_count, dtype=np.float32)


def load_nd2_dataset(path: str) -> Nd2Dataset:
    input_path = str(Path(path))
    with _open_nd2(input_path) as handle:
        raw = handle.asarray()
        data_tczyx = _normalize_to_tczyx(raw, handle.sizes)
        wavelengths_nm = _extract_wavelengths(handle, data_tczyx.shape[1])
        voxel = handle.voxel_size() if hasattr(handle, "voxel_size") else None
        metadata = {
            "path": input_path,
            "axes": ["T", "C", "Z", "Y", "X"],
            "shape": tuple(int(v) for v in data_tczyx.shape),
            "wavelengths_nm": wavelengths_nm.tolist(),
            "voxel_size_um": {
                "x": getattr(voxel, "x", None),
                "y": getattr(voxel, "y", None),
                "z": getattr(voxel, "z", None),
            },
            "is_spectral": bool(data_tczyx.shape[1] > 4),
        }
    return Nd2Dataset(
        path=input_path,
        data_tczyx=data_tczyx,
        wavelengths_nm=wavelengths_nm,
        metadata=metadata,
    )


def get_first_2d_spectral_plane(dataset: Nd2Dataset) -> np.ndarray:
    plane = dataset.data_tczyx[0, :, 0, :, :]
    if plane.ndim != 3:
        raise ValueError(f"Expected 2D spectral plane with shape (C, Y, X), got {plane.shape}")
    return plane
