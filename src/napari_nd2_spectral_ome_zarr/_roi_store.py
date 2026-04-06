from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class RoiSpectrumDataset:
    dataset_id: str
    name: str
    source_layer_name: str
    mode: str
    wavelengths_nm: np.ndarray
    roi_labels: list[str]
    roi_areas_px: np.ndarray
    roi_spectra: np.ndarray
    pooled_spectrum: np.ndarray | None
    trace_kind: str = "mixed"
    trace_label: str = ""
    measurement_kind: str = "spectral_mean"
    analysis_level: str = "image"
    accepted: bool = True
    use_for_analysis: bool = True
    roi_count: int = 0
    roi_class: str = ""
    kernel_size: int = 0
    split_nm: float = 0.0
    n_kernels: int = 0
    n_excluded_kernels: int = 0
    mean_ratio: float = 0.0
    median_ratio: float = 0.0
    sd_ratio: float = 0.0
    mean_intensity: float = 0.0
    animal_id: str = ""
    group_label: str = ""
    genotype: str = ""
    sex: str = ""
    age: str = ""
    region: str = ""
    batch: str = ""
    blind_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class RoiSpectrumStore:
    def __init__(self):
        self._datasets: list[RoiSpectrumDataset] = []
        self._counter = 0
        self._listeners: list[callable] = []

    def _clone_dataset(self, dataset: RoiSpectrumDataset, **overrides) -> RoiSpectrumDataset:
        payload = {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "source_layer_name": dataset.source_layer_name,
            "mode": dataset.mode,
            "wavelengths_nm": np.asarray(dataset.wavelengths_nm, dtype=np.float32).copy(),
            "roi_labels": list(dataset.roi_labels),
            "roi_areas_px": np.asarray(dataset.roi_areas_px, dtype=np.float32).copy(),
            "roi_spectra": np.asarray(dataset.roi_spectra, dtype=np.float32).copy(),
            "pooled_spectrum": None if dataset.pooled_spectrum is None else np.asarray(dataset.pooled_spectrum, dtype=np.float32).copy(),
            "trace_kind": dataset.trace_kind,
            "trace_label": dataset.trace_label,
            "measurement_kind": dataset.measurement_kind,
            "analysis_level": dataset.analysis_level,
            "accepted": dataset.accepted,
            "use_for_analysis": dataset.use_for_analysis,
            "roi_count": int(dataset.roi_count),
            "roi_class": dataset.roi_class,
            "kernel_size": int(dataset.kernel_size),
            "split_nm": float(dataset.split_nm),
            "n_kernels": int(dataset.n_kernels),
            "n_excluded_kernels": int(dataset.n_excluded_kernels),
            "mean_ratio": float(dataset.mean_ratio),
            "median_ratio": float(dataset.median_ratio),
            "sd_ratio": float(dataset.sd_ratio),
            "mean_intensity": float(dataset.mean_intensity),
            "animal_id": dataset.animal_id,
            "group_label": dataset.group_label,
            "genotype": dataset.genotype,
            "sex": dataset.sex,
            "age": dataset.age,
            "region": dataset.region,
            "batch": dataset.batch,
            "blind_id": dataset.blind_id,
            "created_at": dataset.created_at,
        }
        payload.update(overrides)
        return RoiSpectrumDataset(**payload)

    def subscribe(self, listener):
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener):
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self):
        for listener in list(self._listeners):
            listener()

    def add_dataset(
        self,
        *,
        name: str | None = None,
        source_layer_name: str,
        mode: str,
        wavelengths_nm: np.ndarray,
        roi_labels: list[str],
        roi_areas_px: np.ndarray,
        roi_spectra: np.ndarray,
        pooled_spectrum: np.ndarray | None,
        trace_kind: str = "mixed",
        trace_label: str = "",
        measurement_kind: str = "spectral_mean",
        analysis_level: str = "image",
        accepted: bool = True,
        use_for_analysis: bool = True,
        roi_class: str = "",
        kernel_size: int = 0,
        split_nm: float = 0.0,
        n_kernels: int = 0,
        n_excluded_kernels: int = 0,
        mean_ratio: float = 0.0,
        median_ratio: float = 0.0,
        sd_ratio: float = 0.0,
        mean_intensity: float = 0.0,
    ) -> RoiSpectrumDataset:
        self._counter += 1
        dataset = RoiSpectrumDataset(
            dataset_id=f"roi_dataset_{self._counter}",
            name=name or f"{source_layer_name} ROI set {self._counter}",
            source_layer_name=source_layer_name,
            mode=mode,
            wavelengths_nm=np.asarray(wavelengths_nm, dtype=np.float32).copy(),
            roi_labels=list(roi_labels),
            roi_areas_px=np.asarray(roi_areas_px, dtype=np.float32).copy(),
            roi_spectra=np.asarray(roi_spectra, dtype=np.float32).copy(),
            pooled_spectrum=None if pooled_spectrum is None else np.asarray(pooled_spectrum, dtype=np.float32).copy(),
            trace_kind=trace_kind,
            trace_label=trace_label,
            measurement_kind=measurement_kind,
            analysis_level=analysis_level,
            accepted=accepted,
            use_for_analysis=use_for_analysis,
            roi_count=len(roi_labels),
            roi_class=roi_class,
            kernel_size=int(kernel_size),
            split_nm=float(split_nm),
            n_kernels=int(n_kernels),
            n_excluded_kernels=int(n_excluded_kernels),
            mean_ratio=float(mean_ratio),
            median_ratio=float(median_ratio),
            sd_ratio=float(sd_ratio),
            mean_intensity=float(mean_intensity),
        )
        self._datasets.append(dataset)
        self._notify_listeners()
        return dataset

    def add_or_replace_dataset(
        self,
        *,
        name: str | None = None,
        source_layer_name: str,
        mode: str,
        wavelengths_nm: np.ndarray,
        roi_labels: list[str],
        roi_areas_px: np.ndarray,
        roi_spectra: np.ndarray,
        pooled_spectrum: np.ndarray | None,
        trace_kind: str = "mixed",
        trace_label: str = "",
        measurement_kind: str = "spectral_mean",
        analysis_level: str = "image",
        accepted: bool = True,
        use_for_analysis: bool = True,
        roi_class: str = "",
        kernel_size: int = 0,
        split_nm: float = 0.0,
        n_kernels: int = 0,
        n_excluded_kernels: int = 0,
        mean_ratio: float = 0.0,
        median_ratio: float = 0.0,
        sd_ratio: float = 0.0,
        mean_intensity: float = 0.0,
    ) -> RoiSpectrumDataset:
        for index, dataset in enumerate(self._datasets):
            if dataset.source_layer_name != source_layer_name:
                continue
            if dataset.measurement_kind != measurement_kind:
                continue
            if dataset.analysis_level != analysis_level:
                continue
            if dataset.trace_label != trace_label:
                continue
            updated_dataset = self._clone_dataset(
                dataset,
                name=name or dataset.name,
                source_layer_name=source_layer_name,
                mode=mode,
                wavelengths_nm=np.asarray(wavelengths_nm, dtype=np.float32).copy(),
                roi_labels=list(roi_labels),
                roi_areas_px=np.asarray(roi_areas_px, dtype=np.float32).copy(),
                roi_spectra=np.asarray(roi_spectra, dtype=np.float32).copy(),
                pooled_spectrum=None if pooled_spectrum is None else np.asarray(pooled_spectrum, dtype=np.float32).copy(),
                trace_kind=trace_kind,
                trace_label=trace_label,
                measurement_kind=measurement_kind,
                analysis_level=analysis_level,
                accepted=accepted,
                use_for_analysis=use_for_analysis,
                roi_count=len(roi_labels),
                roi_class=roi_class or dataset.roi_class,
                kernel_size=int(kernel_size),
                split_nm=float(split_nm),
                n_kernels=int(n_kernels),
                n_excluded_kernels=int(n_excluded_kernels),
                mean_ratio=float(mean_ratio),
                median_ratio=float(median_ratio),
                sd_ratio=float(sd_ratio),
                mean_intensity=float(mean_intensity),
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            self._datasets[index] = updated_dataset
            self._notify_listeners()
            return updated_dataset

        return self.add_dataset(
            name=name,
            source_layer_name=source_layer_name,
            mode=mode,
            wavelengths_nm=wavelengths_nm,
            roi_labels=roi_labels,
            roi_areas_px=roi_areas_px,
            roi_spectra=roi_spectra,
            pooled_spectrum=pooled_spectrum,
            trace_kind=trace_kind,
            trace_label=trace_label,
            measurement_kind=measurement_kind,
            analysis_level=analysis_level,
            accepted=accepted,
            use_for_analysis=use_for_analysis,
            roi_class=roi_class,
            kernel_size=kernel_size,
            split_nm=split_nm,
            n_kernels=n_kernels,
            n_excluded_kernels=n_excluded_kernels,
            mean_ratio=mean_ratio,
            median_ratio=median_ratio,
            sd_ratio=sd_ratio,
            mean_intensity=mean_intensity,
        )

    def list_datasets(self) -> list[RoiSpectrumDataset]:
        return list(self._datasets)

    def get_dataset(self, index: int) -> RoiSpectrumDataset:
        return self._datasets[index]

    def count(self) -> int:
        return len(self._datasets)

    def update_metadata(self, index: int, **metadata):
        dataset = self.get_dataset(index)
        for key, value in metadata.items():
            if hasattr(dataset, key):
                setattr(dataset, key, value)
        self._notify_listeners()

    def remove_dataset(self, index: int):
        del self._datasets[index]
        self._notify_listeners()

    def remove_datasets(self, indices: list[int]):
        for index in sorted(set(indices), reverse=True):
            del self._datasets[index]
        self._notify_listeners()

    def remove_dataset_rows(
        self,
        index: int,
        *,
        roi_indices: list[int] | None = None,
        remove_pooled: bool = False,
    ):
        dataset = self.get_dataset(index)
        roi_index_set = {int(value) for value in (roi_indices or [])}
        keep_indices = [row_index for row_index in range(len(dataset.roi_labels)) if row_index not in roi_index_set]

        roi_labels = [dataset.roi_labels[row_index] for row_index in keep_indices]
        roi_areas_px = np.asarray(dataset.roi_areas_px[keep_indices], dtype=np.float32)
        roi_spectra = np.asarray(dataset.roi_spectra[keep_indices], dtype=np.float32)

        pooled_spectrum = None if remove_pooled else dataset.pooled_spectrum
        if pooled_spectrum is not None and roi_spectra.size > 0:
            pooled_spectrum = np.mean(roi_spectra, axis=0)
        elif roi_spectra.size == 0 and remove_pooled:
            pooled_spectrum = None

        if not roi_labels and pooled_spectrum is None:
            self.remove_dataset(index)
            return

        self._datasets[index] = self._clone_dataset(
            dataset,
            wavelengths_nm=np.asarray(dataset.wavelengths_nm, dtype=np.float32).copy(),
            roi_labels=roi_labels,
            roi_areas_px=roi_areas_px.copy(),
            roi_spectra=roi_spectra.copy(),
            pooled_spectrum=None if pooled_spectrum is None else np.asarray(pooled_spectrum, dtype=np.float32).copy(),
            roi_count=len(roi_labels),
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self._notify_listeners()

    def export_dataset_csv(self, index: int, output_path: str) -> Path:
        dataset = self.get_dataset(index)
        return _write_dataset_csv(dataset, output_path)

    def export_all_csv(self, output_dir: str) -> list[Path]:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        paths = []
        for dataset in self._datasets:
            safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in dataset.name)
            paths.append(_write_dataset_csv(dataset, output_root / f"{safe_name}.csv"))
        return paths


def _write_dataset_csv(dataset: RoiSpectrumDataset, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset_id",
                "dataset_name",
                "source_layer",
                "created_at",
                "plot_mode",
                "trace_kind",
                "trace_label",
                "measurement_kind",
                "analysis_level",
                "accepted",
                "use_for_analysis",
                "roi_count",
                "roi_class",
                "kernel_size",
                "split_nm",
                "n_kernels",
                "n_excluded_kernels",
                "mean_ratio",
                "median_ratio",
                "sd_ratio",
                "mean_intensity",
                "animal_id",
                "group_label",
                "genotype",
                "sex",
                "age",
                "region",
                "batch",
                "blind_id",
                "spectrum_group",
                "roi_label",
                "roi_area_px",
                "wavelength_nm",
                "intensity",
            ]
        )
        for roi_label, roi_area_px, spectrum in zip(dataset.roi_labels, dataset.roi_areas_px, dataset.roi_spectra, strict=False):
            for wavelength_nm, intensity in zip(dataset.wavelengths_nm, spectrum, strict=False):
                writer.writerow(
                    [
                        dataset.dataset_id,
                        dataset.name,
                        dataset.source_layer_name,
                        dataset.created_at,
                        dataset.mode,
                        dataset.trace_kind,
                        dataset.trace_label,
                        dataset.measurement_kind,
                        dataset.analysis_level,
                        dataset.accepted,
                        dataset.use_for_analysis,
                        dataset.roi_count,
                        dataset.roi_class,
                        dataset.kernel_size,
                        dataset.split_nm,
                        dataset.n_kernels,
                        dataset.n_excluded_kernels,
                        dataset.mean_ratio,
                        dataset.median_ratio,
                        dataset.sd_ratio,
                        dataset.mean_intensity,
                        dataset.animal_id,
                        dataset.group_label,
                        dataset.genotype,
                        dataset.sex,
                        dataset.age,
                        dataset.region,
                        dataset.batch,
                        dataset.blind_id,
                        "roi",
                        roi_label,
                        float(roi_area_px),
                        float(wavelength_nm),
                        float(intensity),
                    ]
                )
        if dataset.pooled_spectrum is not None:
            for wavelength_nm, intensity in zip(dataset.wavelengths_nm, dataset.pooled_spectrum, strict=False):
                writer.writerow(
                    [
                        dataset.dataset_id,
                        dataset.name,
                        dataset.source_layer_name,
                        dataset.created_at,
                        dataset.mode,
                        dataset.trace_kind,
                        dataset.trace_label or "Pooled",
                        dataset.measurement_kind,
                        dataset.analysis_level,
                        dataset.accepted,
                        dataset.use_for_analysis,
                        dataset.roi_count,
                        dataset.roi_class,
                        dataset.kernel_size,
                        dataset.split_nm,
                        dataset.n_kernels,
                        dataset.n_excluded_kernels,
                        dataset.mean_ratio,
                        dataset.median_ratio,
                        dataset.sd_ratio,
                        dataset.mean_intensity,
                        dataset.animal_id,
                        dataset.group_label,
                        dataset.genotype,
                        dataset.sex,
                        dataset.age,
                        dataset.region,
                        dataset.batch,
                        dataset.blind_id,
                        "pooled",
                        "Pooled",
                        "",
                        float(wavelength_nm),
                        float(intensity),
                    ]
                )
    return path


ROI_SPECTRUM_STORE = RoiSpectrumStore()
