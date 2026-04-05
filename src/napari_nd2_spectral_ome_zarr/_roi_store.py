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
        source_layer_name: str,
        mode: str,
        wavelengths_nm: np.ndarray,
        roi_labels: list[str],
        roi_areas_px: np.ndarray,
        roi_spectra: np.ndarray,
        pooled_spectrum: np.ndarray | None,
    ) -> RoiSpectrumDataset:
        self._counter += 1
        dataset = RoiSpectrumDataset(
            dataset_id=f"roi_dataset_{self._counter}",
            name=f"{source_layer_name} ROI set {self._counter}",
            source_layer_name=source_layer_name,
            mode=mode,
            wavelengths_nm=np.asarray(wavelengths_nm, dtype=np.float32).copy(),
            roi_labels=list(roi_labels),
            roi_areas_px=np.asarray(roi_areas_px, dtype=np.float32).copy(),
            roi_spectra=np.asarray(roi_spectra, dtype=np.float32).copy(),
            pooled_spectrum=None if pooled_spectrum is None else np.asarray(pooled_spectrum, dtype=np.float32).copy(),
        )
        self._datasets.append(dataset)
        self._notify_listeners()
        return dataset

    def add_or_replace_dataset(
        self,
        *,
        source_layer_name: str,
        mode: str,
        wavelengths_nm: np.ndarray,
        roi_labels: list[str],
        roi_areas_px: np.ndarray,
        roi_spectra: np.ndarray,
        pooled_spectrum: np.ndarray | None,
    ) -> RoiSpectrumDataset:
        for index, dataset in enumerate(self._datasets):
            if dataset.source_layer_name != source_layer_name:
                continue
            updated_dataset = RoiSpectrumDataset(
                dataset_id=dataset.dataset_id,
                name=dataset.name,
                source_layer_name=source_layer_name,
                mode=mode,
                wavelengths_nm=np.asarray(wavelengths_nm, dtype=np.float32).copy(),
                roi_labels=list(roi_labels),
                roi_areas_px=np.asarray(roi_areas_px, dtype=np.float32).copy(),
                roi_spectra=np.asarray(roi_spectra, dtype=np.float32).copy(),
                pooled_spectrum=None if pooled_spectrum is None else np.asarray(pooled_spectrum, dtype=np.float32).copy(),
                animal_id=dataset.animal_id,
                group_label=dataset.group_label,
                genotype=dataset.genotype,
                sex=dataset.sex,
                age=dataset.age,
                region=dataset.region,
                batch=dataset.batch,
                blind_id=dataset.blind_id,
                created_at=datetime.now().isoformat(timespec="seconds"),
            )
            self._datasets[index] = updated_dataset
            self._notify_listeners()
            return updated_dataset

        return self.add_dataset(
            source_layer_name=source_layer_name,
            mode=mode,
            wavelengths_nm=wavelengths_nm,
            roi_labels=roi_labels,
            roi_areas_px=roi_areas_px,
            roi_spectra=roi_spectra,
            pooled_spectrum=pooled_spectrum,
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

        self._datasets[index] = RoiSpectrumDataset(
            dataset_id=dataset.dataset_id,
            name=dataset.name,
            source_layer_name=dataset.source_layer_name,
            mode=dataset.mode,
            wavelengths_nm=np.asarray(dataset.wavelengths_nm, dtype=np.float32).copy(),
            roi_labels=roi_labels,
            roi_areas_px=roi_areas_px.copy(),
            roi_spectra=roi_spectra.copy(),
            pooled_spectrum=None if pooled_spectrum is None else np.asarray(pooled_spectrum, dtype=np.float32).copy(),
            animal_id=dataset.animal_id,
            group_label=dataset.group_label,
            genotype=dataset.genotype,
            sex=dataset.sex,
            age=dataset.age,
            region=dataset.region,
            batch=dataset.batch,
            blind_id=dataset.blind_id,
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
