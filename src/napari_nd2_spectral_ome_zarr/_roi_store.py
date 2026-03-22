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

    def add_dataset(
        self,
        *,
        source_layer_name: str,
        mode: str,
        wavelengths_nm: np.ndarray,
        roi_labels: list[str],
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
            roi_spectra=np.asarray(roi_spectra, dtype=np.float32).copy(),
            pooled_spectrum=None if pooled_spectrum is None else np.asarray(pooled_spectrum, dtype=np.float32).copy(),
        )
        self._datasets.append(dataset)
        return dataset

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

    def remove_dataset(self, index: int):
        del self._datasets[index]

    def remove_datasets(self, indices: list[int]):
        for index in sorted(set(indices), reverse=True):
            self.remove_dataset(index)

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
                "wavelength_nm",
                "intensity",
            ]
        )
        for roi_label, spectrum in zip(dataset.roi_labels, dataset.roi_spectra, strict=False):
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
                        float(wavelength_nm),
                        float(intensity),
                    ]
                )
    return path


ROI_SPECTRUM_STORE = RoiSpectrumStore()
