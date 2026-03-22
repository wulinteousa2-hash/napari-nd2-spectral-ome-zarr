from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._qt_utils import float_parent_dock_later
from ._roi_store import ROI_SPECTRUM_STORE


class SpectralAnalysisWidget(QWidget):
    DATASET_COLUMNS = [
        ("use_for_analysis", True),
        ("dataset_id", False),
        ("name", False),
        ("source_layer_name", False),
        ("animal_id", True),
        ("group_label", True),
        ("genotype", True),
        ("sex", True),
        ("age", True),
        ("region", True),
        ("batch", True),
        ("blind_id", True),
        ("roi_count", False),
        ("created_at", False),
    ]

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._roi_rows: list[dict] = []
        self._image_rows: list[dict] = []
        self._animal_rows: list[dict] = []
        self._animal_spectra: list[np.ndarray] = []
        self._updating_dataset_table = False
        self._selected_dataset_ids: set[str] = set()

        self.split_edit = QLineEdit("600")
        self.ratio_mode_combo = QComboBox()
        self.ratio_mode_combo.addItems(["sum_above_over_below", "mean_above_over_below", "log10_sum_ratio"])
        self.level_combo = QComboBox()
        self.level_combo.addItems(["Animal", "Image", "ROI"])
        self.normalize_checkbox = QCheckBox("Normalize before ratio")

        self.refresh_button = QPushButton("Refresh Datasets")
        self.refresh_button.clicked.connect(self._refresh_dataset_table)
        self.compute_button = QPushButton("Compute Spectral Analysis")
        self.compute_button.clicked.connect(self._compute_analysis)
        self.remove_selected_button = QPushButton("Remove Selected Datasets")
        self.remove_selected_button.clicked.connect(self._remove_selected_datasets)
        self.remove_current_button = QPushButton("Remove Current Row")
        self.remove_current_button.clicked.connect(self._remove_current_dataset)
        self.ttest_button = QPushButton("WT vs HNPP Welch t-test")
        self.ttest_button.clicked.connect(self._run_ttest)
        self.anova_factor_combo = QComboBox()
        self.anova_factor_combo.addItems(["group_label", "genotype", "sex", "age", "region", "batch"])
        self.anova_button = QPushButton("One-way ANOVA")
        self.anova_button.clicked.connect(self._run_anova)
        self.blind_k_combo = QComboBox()
        self.blind_k_combo.addItems(["2", "3"])
        self.pca_button = QPushButton("Blind PCA / Clustering")
        self.pca_button.clicked.connect(self._run_blind_analysis)

        self.export_roi_button = QPushButton("Export ROI Table CSV")
        self.export_roi_button.clicked.connect(lambda: self._export_table_csv("roi"))
        self.export_image_button = QPushButton("Export Image Table CSV")
        self.export_image_button.clicked.connect(lambda: self._export_table_csv("image"))
        self.export_animal_button = QPushButton("Export Animal Table CSV")
        self.export_animal_button.clicked.connect(lambda: self._export_table_csv("animal"))

        self.dataset_table = QTableWidget()
        self.dataset_table.cellChanged.connect(self._on_dataset_cell_changed)
        self.dataset_table.setAlternatingRowColors(True)

        self.roi_table = QTableWidget()
        self.image_table = QTableWidget()
        self.animal_table = QTableWidget()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.roi_table, "ROI Table")
        self.tabs.addTab(self.image_table, "Image Table")
        self.tabs.addTab(self.animal_table, "Animal Table")

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.stats_label = QLabel("Add ROI datasets from Spectral Viewer, annotate metadata, then compute analysis.")
        self.stats_label.setWordWrap(True)

        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("Split λ (nm)"))
        top_controls.addWidget(self.split_edit)
        top_controls.addWidget(QLabel("Ratio mode"))
        top_controls.addWidget(self.ratio_mode_combo)
        top_controls.addWidget(QLabel("Stats level"))
        top_controls.addWidget(self.level_combo)
        top_controls.addWidget(self.normalize_checkbox)
        top_controls.addWidget(self.refresh_button)
        top_controls.addWidget(self.compute_button)
        top_controls.addWidget(self.remove_selected_button)
        top_controls.addWidget(self.remove_current_button)

        stats_controls = QHBoxLayout()
        stats_controls.addWidget(self.ttest_button)
        stats_controls.addWidget(QLabel("ANOVA factor"))
        stats_controls.addWidget(self.anova_factor_combo)
        stats_controls.addWidget(self.anova_button)
        stats_controls.addWidget(QLabel("Blind k"))
        stats_controls.addWidget(self.blind_k_combo)
        stats_controls.addWidget(self.pca_button)

        export_controls = QHBoxLayout()
        export_controls.addWidget(self.export_roi_button)
        export_controls.addWidget(self.export_image_button)
        export_controls.addWidget(self.export_animal_button)

        layout = QVBoxLayout()
        layout.addLayout(top_controls)
        layout.addWidget(QLabel("Stored ROI Datasets"))
        layout.addWidget(self.dataset_table)
        layout.addLayout(stats_controls)
        layout.addLayout(export_controls)
        layout.addWidget(self.tabs)
        layout.addWidget(self.canvas)
        layout.addWidget(self.stats_label)
        self.setLayout(layout)

        self._refresh_dataset_table()
        float_parent_dock_later(self)

    def _refresh_dataset_table(self):
        datasets = ROI_SPECTRUM_STORE.list_datasets()
        if not self._selected_dataset_ids:
            self._selected_dataset_ids = {dataset.dataset_id for dataset in datasets}
        else:
            valid_ids = {dataset.dataset_id for dataset in datasets}
            self._selected_dataset_ids &= valid_ids
        self._updating_dataset_table = True
        self.dataset_table.clear()
        self.dataset_table.setColumnCount(len(self.DATASET_COLUMNS))
        self.dataset_table.setHorizontalHeaderLabels([column for column, _editable in self.DATASET_COLUMNS])
        self.dataset_table.setRowCount(len(datasets))
        for row_index, dataset in enumerate(datasets):
            values = {
                "use_for_analysis": dataset.dataset_id in self._selected_dataset_ids,
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "source_layer_name": dataset.source_layer_name,
                "animal_id": dataset.animal_id,
                "group_label": dataset.group_label,
                "genotype": dataset.genotype,
                "sex": dataset.sex,
                "age": dataset.age,
                "region": dataset.region,
                "batch": dataset.batch,
                "blind_id": dataset.blind_id,
                "roi_count": str(len(dataset.roi_labels)),
                "created_at": dataset.created_at,
            }
            for column_index, (column_name, editable) in enumerate(self.DATASET_COLUMNS):
                if column_name == "use_for_analysis":
                    item = QTableWidgetItem()
                    item.setCheckState(Qt.Checked if values[column_name] else Qt.Unchecked)
                else:
                    item = QTableWidgetItem(str(values[column_name]))
                item.setData(Qt.UserRole, row_index)
                if column_name == "use_for_analysis":
                    item.setFlags((item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled) & ~Qt.ItemIsEditable)
                elif not editable:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.dataset_table.setItem(row_index, column_index, item)
        self.dataset_table.resizeColumnsToContents()
        self._updating_dataset_table = False

    def _on_dataset_cell_changed(self, row: int, column: int):
        if self._updating_dataset_table:
            return
        column_name, editable = self.DATASET_COLUMNS[column]
        item = self.dataset_table.item(row, column)
        if item is None:
            return
        dataset_index = item.data(Qt.UserRole)
        dataset = ROI_SPECTRUM_STORE.get_dataset(dataset_index)
        if column_name == "use_for_analysis":
            if item.checkState() == Qt.Checked:
                self._selected_dataset_ids.add(dataset.dataset_id)
            else:
                self._selected_dataset_ids.discard(dataset.dataset_id)
            return
        if not editable:
            return
        ROI_SPECTRUM_STORE.update_metadata(dataset_index, **{column_name: item.text().strip()})

    def _selected_datasets(self):
        datasets = ROI_SPECTRUM_STORE.list_datasets()
        return [dataset for dataset in datasets if dataset.dataset_id in self._selected_dataset_ids]

    def _split_value(self) -> float:
        return float(self.split_edit.text().strip())

    def _normalized(self, spectrum: np.ndarray) -> np.ndarray:
        shifted = spectrum - float(np.min(spectrum))
        return shifted / (float(np.max(shifted)) + 1e-8)

    def _compute_ratio(self, wavelengths_nm: np.ndarray, spectrum: np.ndarray, split_nm: float) -> float:
        data = self._normalized(spectrum) if self.normalize_checkbox.isChecked() else spectrum
        below = data[wavelengths_nm <= split_nm]
        above = data[wavelengths_nm > split_nm]
        if below.size == 0 or above.size == 0:
            raise ValueError("Split wavelength must leave channels on both sides.")
        if self.ratio_mode_combo.currentText() == "mean_above_over_below":
            numerator = float(np.mean(above))
            denominator = float(np.mean(below))
        elif self.ratio_mode_combo.currentText() == "log10_sum_ratio":
            numerator = float(np.sum(above))
            denominator = float(np.sum(below))
            return float(np.log10((numerator + 1e-8) / (denominator + 1e-8)))
        else:
            numerator = float(np.sum(above))
            denominator = float(np.sum(below))
        return numerator / (denominator + 1e-8)

    def _compute_analysis(self):
        datasets = self._selected_datasets()
        if not datasets:
            self.stats_label.setText("No selected ROI datasets available.")
            return
        try:
            split_nm = self._split_value()
        except ValueError:
            self.stats_label.setText("Split wavelength must be numeric.")
            return

        roi_rows = []
        image_rows = []
        animal_groups: dict[str, dict] = {}
        for dataset in datasets:
            roi_ratios = []
            for roi_label, spectrum in zip(dataset.roi_labels, dataset.roi_spectra, strict=False):
                ratio = self._compute_ratio(dataset.wavelengths_nm, np.asarray(spectrum, dtype=np.float32), split_nm)
                roi_ratios.append(ratio)
                roi_rows.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "image_name": dataset.name,
                        "source_layer": dataset.source_layer_name,
                        "animal_id": dataset.animal_id,
                        "group_label": dataset.group_label,
                        "genotype": dataset.genotype,
                        "sex": dataset.sex,
                        "age": dataset.age,
                        "region": dataset.region,
                        "batch": dataset.batch,
                        "blind_id": dataset.blind_id,
                        "roi_label": roi_label,
                        "ratio": ratio,
                    }
                )

            pooled_spectrum = np.mean(np.asarray(dataset.roi_spectra, dtype=np.float32), axis=0)
            image_mean_ratio = float(np.mean(roi_ratios)) if roi_ratios else np.nan
            image_rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "image_name": dataset.name,
                    "source_layer": dataset.source_layer_name,
                    "animal_id": dataset.animal_id,
                    "group_label": dataset.group_label,
                    "genotype": dataset.genotype,
                    "sex": dataset.sex,
                    "age": dataset.age,
                    "region": dataset.region,
                    "batch": dataset.batch,
                    "blind_id": dataset.blind_id,
                    "n_roi": len(roi_ratios),
                    "mean_ratio": image_mean_ratio,
                }
            )

            animal_key = dataset.animal_id.strip() or dataset.dataset_id
            entry = animal_groups.setdefault(
                animal_key,
                {
                    "animal_id": animal_key,
                    "group_label": dataset.group_label,
                    "genotype": dataset.genotype,
                    "sex": dataset.sex,
                    "age": dataset.age,
                    "region": dataset.region,
                    "batch": dataset.batch,
                    "blind_id": dataset.blind_id,
                    "image_names": [],
                    "image_ratios": [],
                    "roi_ratios": [],
                    "spectra": [],
                },
            )
            entry["image_names"].append(dataset.name)
            entry["image_ratios"].append(image_mean_ratio)
            entry["roi_ratios"].extend(roi_ratios)
            entry["spectra"].append(pooled_spectrum)

        animal_rows = []
        animal_spectra = []
        for animal_key, entry in animal_groups.items():
            mean_spectrum = np.mean(np.stack(entry["spectra"], axis=0), axis=0)
            animal_spectra.append(mean_spectrum)
            animal_rows.append(
                {
                    "animal_id": animal_key,
                    "group_label": entry["group_label"],
                    "genotype": entry["genotype"],
                    "sex": entry["sex"],
                    "age": entry["age"],
                    "region": entry["region"],
                    "batch": entry["batch"],
                    "blind_id": entry["blind_id"],
                    "n_images": len(entry["image_names"]),
                    "n_rois": len(entry["roi_ratios"]),
                    "mean_ratio": float(np.mean(entry["image_ratios"])) if entry["image_ratios"] else np.nan,
                    "sd_ratio": float(np.std(entry["image_ratios"], ddof=1)) if len(entry["image_ratios"]) > 1 else 0.0,
                }
            )

        self._roi_rows = roi_rows
        self._image_rows = image_rows
        self._animal_rows = animal_rows
        self._animal_spectra = animal_spectra
        self._populate_table(self.roi_table, roi_rows)
        self._populate_table(self.image_table, image_rows)
        self._populate_table(self.animal_table, animal_rows)
        self.stats_label.setText(
            f"Computed ROI ratios at split {split_nm:.1f} nm for {len(roi_rows)} ROI(s), "
            f"{len(image_rows)} image set(s), and {len(animal_rows)} animal aggregate(s) from {len(datasets)} selected dataset(s)."
        )

    def _remove_selected_datasets(self):
        datasets = ROI_SPECTRUM_STORE.list_datasets()
        indices = [index for index, dataset in enumerate(datasets) if dataset.dataset_id in self._selected_dataset_ids]
        if not indices:
            self.stats_label.setText("No selected datasets to remove.")
            return
        ROI_SPECTRUM_STORE.remove_datasets(indices)
        self._selected_dataset_ids.clear()
        self._roi_rows = []
        self._image_rows = []
        self._animal_rows = []
        self._animal_spectra = []
        self._populate_table(self.roi_table, [])
        self._populate_table(self.image_table, [])
        self._populate_table(self.animal_table, [])
        self._refresh_dataset_table()
        self.stats_label.setText(f"Removed {len(indices)} selected dataset(s) from memory.")

    def _remove_current_dataset(self):
        row = self.dataset_table.currentRow()
        if row < 0:
            self.stats_label.setText("Select a dataset row to remove.")
            return
        item = self.dataset_table.item(row, 1)
        if item is None:
            self.stats_label.setText("Select a valid dataset row to remove.")
            return
        dataset_index = item.data(Qt.UserRole)
        dataset = ROI_SPECTRUM_STORE.get_dataset(dataset_index)
        self._selected_dataset_ids.discard(dataset.dataset_id)
        ROI_SPECTRUM_STORE.remove_dataset(dataset_index)
        self._roi_rows = []
        self._image_rows = []
        self._animal_rows = []
        self._animal_spectra = []
        self._populate_table(self.roi_table, [])
        self._populate_table(self.image_table, [])
        self._populate_table(self.animal_table, [])
        self._refresh_dataset_table()
        self.stats_label.setText(f"Removed dataset {dataset.dataset_id} from memory.")

    def _populate_table(self, table: QTableWidget, rows: list[dict]):
        table.clear()
        if not rows:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        headers = list(rows[0].keys())
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, header in enumerate(headers):
                item = QTableWidgetItem(str(row[header]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row_index, column_index, item)
        table.resizeColumnsToContents()

    def _selected_level_rows(self) -> list[dict]:
        level = self.level_combo.currentText()
        if level == "ROI":
            return self._roi_rows
        if level == "Image":
            return self._image_rows
        return self._animal_rows

    def _run_ttest(self):
        rows = self._selected_level_rows()
        if not rows:
            self.stats_label.setText("Compute analysis before running t-test.")
            return
        wt = [float(row["mean_ratio"] if "mean_ratio" in row else row["ratio"]) for row in rows if str(row.get("group_label", "")).strip().lower() == "wt"]
        hnpp = [float(row["mean_ratio"] if "mean_ratio" in row else row["ratio"]) for row in rows if str(row.get("group_label", "")).strip().lower() == "hnpp"]
        if len(wt) < 2 or len(hnpp) < 2:
            self.stats_label.setText("Need at least two WT and two HNPP values at the selected analysis level.")
            return
        statistic, pvalue = self._welch_ttest_permutation(np.asarray(wt, dtype=np.float32), np.asarray(hnpp, dtype=np.float32))
        self.stats_label.setText(
            f"Welch t-test on {self.level_combo.currentText().lower()} means: "
            f"WT n={len(wt)}, HNPP n={len(hnpp)}, t={statistic:.4f}, permutation p={pvalue:.4g}"
        )

    def _run_anova(self):
        rows = self._selected_level_rows()
        if not rows:
            self.stats_label.setText("Compute analysis before running ANOVA.")
            return
        factor = self.anova_factor_combo.currentText()
        grouped: dict[str, list[float]] = {}
        metric_key = "mean_ratio" if rows and "mean_ratio" in rows[0] else "ratio"
        for row in rows:
            label = str(row.get(factor, "")).strip()
            if not label:
                continue
            grouped.setdefault(label, []).append(float(row[metric_key]))
        valid = [(label, values) for label, values in grouped.items() if len(values) >= 2]
        if len(valid) < 2:
            self.stats_label.setText(f"Need at least two non-empty {factor} groups with >=2 values each for ANOVA.")
            return
        statistic, pvalue = self._anova_permutation([np.asarray(values, dtype=np.float32) for _label, values in valid])
        summary = ", ".join(f"{label}:n={len(values)}" for label, values in valid)
        self.stats_label.setText(
            f"One-way ANOVA by {factor} on {self.level_combo.currentText().lower()} values: "
            f"F={statistic:.4f}, permutation p={pvalue:.4g} ({summary})"
        )

    def _run_blind_analysis(self):
        if not self._animal_rows or not self._animal_spectra:
            self.stats_label.setText("Compute analysis before running blind PCA.")
            return
        spectra = np.stack(self._animal_spectra, axis=0)
        if spectra.shape[0] < 2:
            self.stats_label.setText("Need at least two animal spectra for blind PCA.")
            return
        if self.normalize_checkbox.isChecked():
            spectra = np.stack([self._normalized(spectrum) for spectrum in spectra], axis=0)
        centered = spectra - np.mean(spectra, axis=0, keepdims=True)
        u, singular_values, _vh = np.linalg.svd(centered, full_matrices=False)
        score_dims = min(2, singular_values.shape[0])
        scores = u[:, :score_dims] * singular_values[:score_dims]
        if score_dims == 1:
            scores = np.column_stack([scores[:, 0], np.zeros(scores.shape[0], dtype=np.float32)])
        labels = self._kmeans(scores, k=int(self.blind_k_combo.currentText()))
        separation, pvalue = self._cluster_permutation_pvalue(scores, labels)

        self.figure.clear()
        axis = self.figure.add_subplot(111)
        for cluster_id in sorted(set(labels.tolist())):
            mask = labels == cluster_id
            axis.scatter(scores[mask, 0], scores[mask, 1], label=f"Cluster {cluster_id + 1}")
            for row_index in np.where(mask)[0]:
                blind_id = self._animal_rows[row_index].get("blind_id") or self._animal_rows[row_index].get("animal_id")
                axis.text(scores[row_index, 0], scores[row_index, 1], str(blind_id), fontsize=8)
        total_var = np.sum(singular_values**2) + 1e-8
        axis.set_xlabel(f"PC1 ({(singular_values[0] ** 2 / total_var) * 100:.1f}%)")
        pc2_pct = (singular_values[1] ** 2 / total_var) * 100 if singular_values.shape[0] > 1 else 0.0
        axis.set_ylabel(f"PC2 ({pc2_pct:.1f}%)")
        axis.set_title("Blind PCA clustering")
        axis.legend()
        axis.grid(True, alpha=0.3)
        self.canvas.draw()
        self.stats_label.setText(
            f"Blind PCA clustering complete on animal spectra: separation={separation:.4f}, permutation p={pvalue:.4g}."
        )

    def _kmeans(self, data: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
        if data.shape[0] < k:
            raise ValueError("Not enough samples for the requested number of clusters.")
        centroids = data[:k].copy()
        labels = np.zeros(data.shape[0], dtype=int)
        for _ in range(max_iter):
            distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster_id in range(k):
                mask = labels == cluster_id
                if np.any(mask):
                    centroids[cluster_id] = np.mean(data[mask], axis=0)
        return labels

    def _cluster_separation_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        unique_labels = sorted(set(labels.tolist()))
        if len(unique_labels) < 2:
            return 0.0
        centroids = [np.mean(data[labels == label], axis=0) for label in unique_labels]
        between = 0.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                between += float(np.linalg.norm(centroids[i] - centroids[j]))
        within = 0.0
        for label, centroid in zip(unique_labels, centroids, strict=False):
            within += float(np.mean(np.linalg.norm(data[labels == label] - centroid, axis=1)))
        return between / (within + 1e-8)

    def _cluster_permutation_pvalue(self, data: np.ndarray, labels: np.ndarray, n_perm: int = 500) -> tuple[float, float]:
        observed = self._cluster_separation_score(data, labels)
        exceed = 0
        rng = np.random.default_rng(42)
        for _ in range(n_perm):
            shuffled = rng.permutation(labels)
            if self._cluster_separation_score(data, shuffled) >= observed:
                exceed += 1
        return observed, (exceed + 1) / (n_perm + 1)

    def _welch_t_statistic(self, group1: np.ndarray, group2: np.ndarray) -> float:
        mean_diff = float(np.mean(group1) - np.mean(group2))
        var_term = (float(np.var(group1, ddof=1)) / len(group1)) + (float(np.var(group2, ddof=1)) / len(group2))
        return mean_diff / np.sqrt(var_term + 1e-8)

    def _welch_ttest_permutation(self, group1: np.ndarray, group2: np.ndarray, n_perm: int = 5000) -> tuple[float, float]:
        observed = self._welch_t_statistic(group1, group2)
        pooled = np.concatenate([group1, group2])
        size1 = len(group1)
        exceed = 0
        rng = np.random.default_rng(42)
        for _ in range(n_perm):
            permuted = rng.permutation(pooled)
            perm1 = permuted[:size1]
            perm2 = permuted[size1:]
            if abs(self._welch_t_statistic(perm1, perm2)) >= abs(observed):
                exceed += 1
        return observed, (exceed + 1) / (n_perm + 1)

    def _anova_f_statistic(self, groups: list[np.ndarray]) -> float:
        all_values = np.concatenate(groups)
        grand_mean = float(np.mean(all_values))
        ss_between = 0.0
        ss_within = 0.0
        for group in groups:
            group_mean = float(np.mean(group))
            ss_between += len(group) * ((group_mean - grand_mean) ** 2)
            ss_within += float(np.sum((group - group_mean) ** 2))
        df_between = len(groups) - 1
        df_within = len(all_values) - len(groups)
        ms_between = ss_between / max(df_between, 1)
        ms_within = ss_within / max(df_within, 1)
        return ms_between / (ms_within + 1e-8)

    def _anova_permutation(self, groups: list[np.ndarray], n_perm: int = 5000) -> tuple[float, float]:
        observed = self._anova_f_statistic(groups)
        pooled = np.concatenate(groups)
        sizes = [len(group) for group in groups]
        exceed = 0
        rng = np.random.default_rng(42)
        for _ in range(n_perm):
            permuted = rng.permutation(pooled)
            start = 0
            perm_groups = []
            for size in sizes:
                perm_groups.append(permuted[start:start + size])
                start += size
            if self._anova_f_statistic(perm_groups) >= observed:
                exceed += 1
        return observed, (exceed + 1) / (n_perm + 1)

    def _export_table_csv(self, table_name: str):
        rows = {
            "roi": self._roi_rows,
            "image": self._image_rows,
            "animal": self._animal_rows,
        }[table_name]
        if not rows:
            self.stats_label.setText(f"No {table_name} analysis table available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, f"Export {table_name.title()} Table CSV", f"{table_name}_analysis.csv", "CSV files (*.csv)")
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.stats_label.setText(f"Exported {table_name} analysis table to {Path(path).name}")
