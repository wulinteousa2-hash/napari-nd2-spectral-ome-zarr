from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from importlib import resources
from importlib.util import find_spec

import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import uniform_filter
from skimage import exposure, img_as_ubyte


def _load_cie_csv():
    wavelengths = []
    x_vals = []
    y_vals = []
    z_vals = []
    with resources.files(__package__).joinpath("resources/CIE1931_CMF.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            wavelengths.append(float(row["WL (nm)"]))
            x_vals.append(float(row["X"]))
            y_vals.append(float(row["Y"]))
            z_vals.append(float(row["Z"]))
    return (
        np.asarray(wavelengths, dtype=np.float32),
        np.asarray(x_vals, dtype=np.float32),
        np.asarray(y_vals, dtype=np.float32),
        np.asarray(z_vals, dtype=np.float32),
    )


_CIE_WAVELENGTHS, _CIE_X, _CIE_Y, _CIE_Z = _load_cie_csv()


def approximate_cie_xyz(wavelengths_nm: np.ndarray) -> np.ndarray:
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float32)
    x_bar = np.interp(wavelengths_nm, _CIE_WAVELENGTHS, _CIE_X)
    y_bar = np.interp(wavelengths_nm, _CIE_WAVELENGTHS, _CIE_Y)
    z_bar = np.interp(wavelengths_nm, _CIE_WAVELENGTHS, _CIE_Z)
    return np.stack([x_bar, y_bar, z_bar], axis=1)


def gpu_available() -> bool:
    return find_spec("cupy") is not None


def get_gpu_status_text(use_gpu: bool) -> str:
    if not gpu_available():
        return "GPU: OFF (CuPy not installed)"
    return "GPU: ON" if use_gpu else "GPU: OFF"


def _xyz_to_srgb(xyz, xp):
    xyz = xp.asarray(xyz, dtype=xp.float32)
    xyz_to_rgb = xp.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ],
        dtype=xp.float32,
    )
    rgb_linear = xyz @ xyz_to_rgb.T
    rgb_linear = xp.clip(rgb_linear, 0.0, None)
    threshold = 0.0031308
    srgb = xp.where(
        rgb_linear <= threshold,
        12.92 * rgb_linear,
        1.055 * xp.power(rgb_linear, 1 / 2.4) - 0.055,
    )
    return xp.clip(srgb, 0.0, 1.0)


def cie_to_rgb(xyz: np.ndarray) -> np.ndarray:
    rgb = _xyz_to_srgb(np.asarray(xyz, dtype=np.float32), np)
    return (rgb * 255).astype(np.uint8)


def _auto_clean_truecolor_background(
    spectral_cube: np.ndarray,
    *,
    strength: str = "med",
) -> tuple[np.ndarray, np.ndarray]:
    presets = {
        "low": {"bg_cut": 10.0, "bg_pct": 30.0, "black": 1.0, "full": 28.0},
        "med": {"bg_cut": 18.0, "bg_pct": 40.0, "black": 2.0, "full": 36.0},
        "high": {"bg_cut": 28.0, "bg_pct": 50.0, "black": 4.0, "full": 46.0},
    }
    config = presets.get(strength.lower(), presets["med"])
    cube = np.asarray(spectral_cube, dtype=np.float32)
    visible = cube.mean(axis=0)
    low = float(np.percentile(visible, 1.0))
    high = float(np.percentile(visible, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return cube, np.ones(visible.shape, dtype=np.float32)

    background_threshold = float(np.percentile(visible, config["bg_cut"]))
    background_mask = visible <= background_threshold
    if int(background_mask.sum()) >= max(16, cube.shape[1] * cube.shape[2] // 200):
        background_spectrum = np.percentile(cube[:, background_mask], config["bg_pct"], axis=1).astype(np.float32)
        cube = cube - background_spectrum[:, None, None]
        cube = np.clip(cube, 0.0, None)

    cleaned_visible = cube.mean(axis=0)
    black = float(np.percentile(cleaned_visible, config["black"]))
    full = float(np.percentile(cleaned_visible, config["full"]))
    if not np.isfinite(black) or not np.isfinite(full) or full <= black:
        return cube, np.ones(visible.shape, dtype=np.float32)

    alpha = np.clip((cleaned_visible - black) / (full - black + 1e-8), 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    alpha = uniform_filter(alpha.astype(np.float32), size=3, mode="nearest")
    return cube, np.clip(alpha, 0.0, 1.0).astype(np.float32)


def estimate_truecolor_rgb(
    spectral_cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    gamma: float = 1.4,
    use_gpu: bool = False,
    auto_clean_background: bool = False,
    clean_background_strength: str = "med",
) -> np.ndarray:
    alpha_mask = None
    if auto_clean_background:
        spectral_cube, alpha_mask = _auto_clean_truecolor_background(
            spectral_cube,
            strength=clean_background_strength,
        )

    xp = np
    if use_gpu and gpu_available():
        import cupy as cp

        xp = cp

    spectral_cube = xp.asarray(spectral_cube, dtype=xp.float32)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float32)
    if spectral_cube.ndim != 3:
        raise ValueError(f"Expected spectral cube with shape (C, Y, X), got {spectral_cube.shape}")
    if spectral_cube.shape[0] != wavelengths_nm.shape[0]:
        raise ValueError("Wavelength count does not match spectral channel count")

    cube = spectral_cube - spectral_cube.min()
    if alpha_mask is not None and bool(np.any(alpha_mask > 0.1)):
        alpha_mask_xp = xp.asarray(alpha_mask)
        foreground_mask_xp = alpha_mask_xp > 0.1
        cube /= xp.max(cube[:, foreground_mask_xp]) + 1e-8
    else:
        alpha_mask_xp = None
        foreground_mask_xp = None
        cube /= cube.max() + 1e-8

    xyz_weights = xp.asarray(approximate_cie_xyz(wavelengths_nm), dtype=xp.float32)
    xyz = xp.tensordot(cube.transpose(1, 2, 0), xyz_weights, axes=([2], [0]))
    if foreground_mask_xp is not None:
        xyz /= xp.max(xyz[foreground_mask_xp], axis=0, keepdims=True) + 1e-8
    else:
        xyz /= xp.max(xyz, axis=(0, 1), keepdims=True) + 1e-8

    rgb = _xyz_to_srgb(xyz, xp)
    if gamma != 1.0:
        rgb = xp.power(xp.clip(rgb, 0.0, 1.0), gamma)
    if alpha_mask_xp is not None:
        rgb = rgb * alpha_mask_xp[..., None]

    if xp is not np:
        rgb = xp.asnumpy(rgb)
    return (rgb * 255).astype(np.uint8)


def summed_visible_image(spectral_cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(spectral_cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected spectral cube with shape (C, Y, X), got {cube.shape}")
    return cube.mean(axis=0)


def _render_truecolor_cpu(
    spectral_cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    auto_clean_background: bool = False,
    clean_background_strength: str = "med",
) -> np.ndarray:
    return estimate_truecolor_rgb(
        spectral_cube,
        wavelengths_nm,
        use_gpu=False,
        auto_clean_background=auto_clean_background,
        clean_background_strength=clean_background_strength,
    )


def _render_visible_cpu(spectral_cube: np.ndarray) -> np.ndarray:
    return summed_visible_image(spectral_cube)


def render_visible_truecolor(
    spectral_cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    use_gpu: bool = False,
    max_workers: int | None = None,
    auto_clean_background: bool = False,
    clean_background_strength: str = "med",
) -> tuple[np.ndarray, np.ndarray]:
    spectral_cube = np.asarray(spectral_cube, dtype=np.float32)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float32)

    if use_gpu and gpu_available():
        visible = summed_visible_image(spectral_cube)
        truecolor = estimate_truecolor_rgb(
            spectral_cube,
            wavelengths_nm,
            use_gpu=True,
            auto_clean_background=auto_clean_background,
            clean_background_strength=clean_background_strength,
        )
        return visible, truecolor

    worker_count = max_workers or max(2, min(4, (os.cpu_count() or 4) // 2))
    if worker_count <= 1:
        return (
            summed_visible_image(spectral_cube),
            estimate_truecolor_rgb(
                spectral_cube,
                wavelengths_nm,
                use_gpu=False,
                auto_clean_background=auto_clean_background,
                clean_background_strength=clean_background_strength,
            ),
        )

    with ProcessPoolExecutor(max_workers=min(2, worker_count)) as executor:
        visible_future = executor.submit(_render_visible_cpu, spectral_cube)
        truecolor_future = executor.submit(
            _render_truecolor_cpu,
            spectral_cube,
            wavelengths_nm,
            auto_clean_background,
            clean_background_strength,
        )
        visible = visible_future.result()
        truecolor = truecolor_future.result()
    return visible, truecolor


def save_pseudocolor_config(path: str, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def load_pseudocolor_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _kernel_average_numpy(spectral_cube: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return np.asarray(spectral_cube, dtype=np.float32)
    return uniform_filter(np.asarray(spectral_cube, dtype=np.float32), size=(1, kernel_size, kernel_size), mode="nearest")


def _resample_reference(reference: np.ndarray, src_wavelengths: np.ndarray, dst_wavelengths: np.ndarray) -> np.ndarray:
    return np.interp(dst_wavelengths, src_wavelengths, reference).astype(np.float32)


def _build_rainbow_lookup(lookup_size: int = 256) -> np.ndarray:
    lookup = np.zeros((lookup_size, 3), dtype=np.uint8)
    for i in range(lookup_size):
        v = i / (lookup_size - 1)
        wavelength_val = 380 + v * 400
        X = np.interp(wavelength_val, _CIE_WAVELENGTHS, _CIE_X)
        Y = np.interp(wavelength_val, _CIE_WAVELENGTHS, _CIE_Y)
        Z = np.interp(wavelength_val, _CIE_WAVELENGTHS, _CIE_Z)
        lookup[i] = cie_to_rgb(np.array([X, Y, Z], dtype=np.float32))
    return lookup


_RAINBOW_LOOKUP = _build_rainbow_lookup()


def generate_pseudocolor_image(
    spectral_cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    reference_spectrum: np.ndarray,
    shift: float,
    gamma: float = 1.2,
    kernel_size: int = 3,
    index_method: str = "correlation",
    use_bg_subtraction: bool = False,
    bg_spectrum: np.ndarray | None = None,
    bg_similarity_threshold: float = 0.9,
    auto_contrast: bool = False,
    adaptive_eq: bool = False,
    adaptive_clip: float = 0.03,
    use_gpu: bool = False,
    reference_wavelengths_nm: np.ndarray | None = None,
) -> np.ndarray:
    del use_gpu
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float32)
    spectral_cube = np.asarray(spectral_cube, dtype=np.float32)
    if spectral_cube.ndim != 3:
        raise ValueError("Expected spectral cube with shape (C, Y, X).")
    if spectral_cube.shape[0] != wavelengths_nm.shape[0]:
        raise ValueError("Wavelength count does not match spectral cube channels.")

    if reference_wavelengths_nm is None:
        reference_wavelengths_nm = wavelengths_nm
    else:
        reference_wavelengths_nm = np.asarray(reference_wavelengths_nm, dtype=np.float32)

    mean_spectrum = _resample_reference(np.asarray(reference_spectrum, dtype=np.float32), reference_wavelengths_nm, wavelengths_nm)
    left_ref = np.interp(wavelengths_nm, wavelengths_nm + (-shift), mean_spectrum, left=mean_spectrum[0], right=mean_spectrum[-1]).astype(np.float32)
    right_ref = np.interp(wavelengths_nm, wavelengths_nm + shift, mean_spectrum, left=mean_spectrum[0], right=mean_spectrum[-1]).astype(np.float32)

    cube = _kernel_average_numpy(spectral_cube, kernel_size)
    channels, height, width = cube.shape
    flat = cube.reshape(channels, -1)

    if index_method == "distance":
        d_left = np.linalg.norm(flat - left_ref[:, None], axis=0)
        d_right = np.linalg.norm(flat - right_ref[:, None], axis=0)
        indices = 1.0 - (d_left / (d_left + d_right + 1e-8))
    elif index_method == "ratio":
        indices = np.dot(right_ref, flat) / (np.dot(left_ref, flat) + 1e-8)
    elif index_method == "correlation":
        norm_pix = np.linalg.norm(flat, axis=0) + 1e-8
        corr_left = np.dot(left_ref, flat) / (norm_pix * (np.linalg.norm(left_ref) + 1e-8))
        corr_right = np.dot(right_ref, flat) / (norm_pix * (np.linalg.norm(right_ref) + 1e-8))
        indices = (corr_right - corr_left) / (corr_right + corr_left + 1e-8)
    else:
        raise ValueError(f"Unknown index_method: {index_method}")

    indices = indices.reshape(height, width).astype(np.float32)
    raw_intensity = cube.sum(axis=0)
    i_min, i_max = np.percentile(raw_intensity, [1, 99])
    intensity_map = np.clip((raw_intensity - i_min) / (i_max - i_min + 1e-8), 0.0, 1.0)

    idx_min, idx_max = float(indices.min()), float(indices.max())
    denom = (idx_max - idx_min) if idx_max > idx_min else 1e-8
    indices = (indices - idx_min) / denom
    indices = exposure.adjust_gamma(indices, gamma)

    pseudo_rgb = _RAINBOW_LOOKUP[(indices * (_RAINBOW_LOOKUP.shape[0] - 1)).astype(np.uint8)]
    hsv = mcolors.rgb_to_hsv(pseudo_rgb.astype(np.float32) / 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * intensity_map, 0.0, 1.0)
    pseudo_rgb = (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)

    if use_bg_subtraction and bg_spectrum is not None:
        bg_ref = _resample_reference(np.asarray(bg_spectrum, dtype=np.float32), reference_wavelengths_nm, wavelengths_nm)
        similarity = np.dot(bg_ref, flat) / ((np.linalg.norm(flat, axis=0) + 1e-8) * (np.linalg.norm(bg_ref) + 1e-8))
        similarity = ((similarity + 1.0) / 2.0).reshape(height, width)
        pseudo_rgb[similarity >= bg_similarity_threshold] = [0, 0, 0]

    if auto_contrast:
        pseudo_rgb = img_as_ubyte(exposure.rescale_intensity(pseudo_rgb, in_range="image", out_range=(0, 1)))
    if adaptive_eq:
        pseudo_rgb = img_as_ubyte(exposure.equalize_adapthist(pseudo_rgb, clip_limit=adaptive_clip))
    return pseudo_rgb


def generate_pseudocolor_pair_image(
    spectral_cube: np.ndarray,
    wavelengths_nm: np.ndarray,
    left_reference: np.ndarray,
    right_reference: np.ndarray,
    kernel_size: int = 3,
    gamma: float = 1.2,
    index_method: str = "correlation",
    use_bg_subtraction: bool = False,
    bg_spectrum: np.ndarray | None = None,
    bg_similarity_threshold: float = 0.9,
    auto_contrast: bool = False,
    adaptive_eq: bool = False,
    adaptive_clip: float = 0.03,
    reference_wavelengths_nm: np.ndarray | None = None,
) -> np.ndarray:
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=np.float32)
    spectral_cube = np.asarray(spectral_cube, dtype=np.float32)
    if reference_wavelengths_nm is None:
        reference_wavelengths_nm = wavelengths_nm
    else:
        reference_wavelengths_nm = np.asarray(reference_wavelengths_nm, dtype=np.float32)

    left_ref = _resample_reference(np.asarray(left_reference, dtype=np.float32), reference_wavelengths_nm, wavelengths_nm)
    right_ref = _resample_reference(np.asarray(right_reference, dtype=np.float32), reference_wavelengths_nm, wavelengths_nm)

    cube = _kernel_average_numpy(spectral_cube, kernel_size)
    channels, height, width = cube.shape
    flat = cube.reshape(channels, -1)

    if index_method == "distance":
        d_left = np.linalg.norm(flat - left_ref[:, None], axis=0)
        d_right = np.linalg.norm(flat - right_ref[:, None], axis=0)
        indices = 1.0 - (d_left / (d_left + d_right + 1e-8))
    elif index_method == "ratio":
        indices = np.dot(right_ref, flat) / (np.dot(left_ref, flat) + 1e-8)
    elif index_method == "correlation":
        norm_pix = np.linalg.norm(flat, axis=0) + 1e-8
        corr_left = np.dot(left_ref, flat) / (norm_pix * (np.linalg.norm(left_ref) + 1e-8))
        corr_right = np.dot(right_ref, flat) / (norm_pix * (np.linalg.norm(right_ref) + 1e-8))
        indices = (corr_right - corr_left) / (corr_right + corr_left + 1e-8)
    else:
        raise ValueError(f"Unknown index_method: {index_method}")

    indices = indices.reshape(height, width).astype(np.float32)
    raw_intensity = cube.sum(axis=0)
    i_min, i_max = np.percentile(raw_intensity, [1, 99])
    intensity_map = np.clip((raw_intensity - i_min) / (i_max - i_min + 1e-8), 0.0, 1.0)

    idx_min, idx_max = float(indices.min()), float(indices.max())
    denom = (idx_max - idx_min) if idx_max > idx_min else 1e-8
    indices = (indices - idx_min) / denom
    indices = exposure.adjust_gamma(indices, gamma)

    pseudo_rgb = _RAINBOW_LOOKUP[(indices * (_RAINBOW_LOOKUP.shape[0] - 1)).astype(np.uint8)]
    hsv = mcolors.rgb_to_hsv(pseudo_rgb.astype(np.float32) / 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * intensity_map, 0.0, 1.0)
    pseudo_rgb = (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)

    if use_bg_subtraction and bg_spectrum is not None:
        bg_ref = _resample_reference(np.asarray(bg_spectrum, dtype=np.float32), reference_wavelengths_nm, wavelengths_nm)
        similarity = np.dot(bg_ref, flat) / ((np.linalg.norm(flat, axis=0) + 1e-8) * (np.linalg.norm(bg_ref) + 1e-8))
        similarity = ((similarity + 1.0) / 2.0).reshape(height, width)
        pseudo_rgb[similarity >= bg_similarity_threshold] = [0, 0, 0]

    if auto_contrast:
        pseudo_rgb = img_as_ubyte(exposure.rescale_intensity(pseudo_rgb, in_range="image", out_range=(0, 1)))
    if adaptive_eq:
        pseudo_rgb = img_as_ubyte(exposure.equalize_adapthist(pseudo_rgb, clip_limit=adaptive_clip))
    return pseudo_rgb


def pseudocolor_config(
    reference_spectrum: np.ndarray,
    wavelengths_nm: np.ndarray,
    shift: float,
    gamma: float,
    kernel_size: int,
    index_method: str = "correlation",
    use_bg_subtraction: bool = False,
    bg_spectrum: np.ndarray | None = None,
    bg_similarity_threshold: float = 0.9,
    auto_contrast: bool = False,
    adaptive_eq: bool = False,
    adaptive_clip: float = 0.03,
) -> dict:
    return {
        "mode": "auto_shift",
        "reference_spectrum": np.asarray(reference_spectrum, dtype=np.float32).tolist(),
        "wavelengths_nm": np.asarray(wavelengths_nm, dtype=np.float32).tolist(),
        "shift": float(shift),
        "gamma": float(gamma),
        "kernel_size": int(kernel_size),
        "index_method": index_method,
        "use_bg_subtraction": bool(use_bg_subtraction),
        "bg_spectrum": None if bg_spectrum is None else np.asarray(bg_spectrum, dtype=np.float32).tolist(),
        "bg_similarity_threshold": float(bg_similarity_threshold),
        "auto_contrast": bool(auto_contrast),
        "adaptive_eq": bool(adaptive_eq),
        "adaptive_clip": float(adaptive_clip),
        "version": 2,
    }


def pseudocolor_pair_config(
    left_reference: np.ndarray,
    right_reference: np.ndarray,
    wavelengths_nm: np.ndarray,
    gamma: float,
    kernel_size: int,
    index_method: str = "correlation",
    use_bg_subtraction: bool = False,
    bg_spectrum: np.ndarray | None = None,
    bg_similarity_threshold: float = 0.9,
    auto_contrast: bool = False,
    adaptive_eq: bool = False,
    adaptive_clip: float = 0.03,
) -> dict:
    return {
        "mode": "roi_pair",
        "left_reference": np.asarray(left_reference, dtype=np.float32).tolist(),
        "right_reference": np.asarray(right_reference, dtype=np.float32).tolist(),
        "wavelengths_nm": np.asarray(wavelengths_nm, dtype=np.float32).tolist(),
        "gamma": float(gamma),
        "kernel_size": int(kernel_size),
        "index_method": index_method,
        "use_bg_subtraction": bool(use_bg_subtraction),
        "bg_spectrum": None if bg_spectrum is None else np.asarray(bg_spectrum, dtype=np.float32).tolist(),
        "bg_similarity_threshold": float(bg_similarity_threshold),
        "auto_contrast": bool(auto_contrast),
        "adaptive_eq": bool(adaptive_eq),
        "adaptive_clip": float(adaptive_clip),
        "version": 2,
    }
