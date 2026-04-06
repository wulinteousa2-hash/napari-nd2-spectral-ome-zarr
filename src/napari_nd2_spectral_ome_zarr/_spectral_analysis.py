from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy import stats
from qtpy.QtCore import Qt
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
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
        ("trace_kind", False),
        ("trace_label", False),
        ("animal_id", True),
        ("group_label", True),
        ("genotype", True),
        ("sex", True),
        ("age", True),
        ("region", True),
        ("batch", True),
        ("blind_id", True),
        ("roi_class", True),
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

        self.split_edit = QLineEdit("600")
        self.ratio_mode_combo = QComboBox()
        self.ratio_mode_combo.addItems(["sum_above_over_below", "mean_above_over_below", "log10_sum_ratio"])
        self.measurement_combo = QComboBox()
        self.measurement_combo.addItems(["spectral_mean", "spatial_ratio"])
        self.level_combo = QComboBox()
        self.level_combo.addItems(["ROI", "Image", "Animal"])
        self.normalize_checkbox = QCheckBox("Normalize before ratio")
        self.significance_edit = QLineEdit("0.05")
        self.confidence_edit = QLineEdit("95")
        self.stats_factor_combo = QComboBox()
        self.stats_factor_combo.addItems(["group_label", "genotype", "sex", "age", "region", "batch", "roi_class"])

        self.refresh_button = QPushButton("Refresh Datasets")
        self.refresh_button.clicked.connect(self._refresh_dataset_table)
        self.compute_button = QPushButton("Compute Spectral Analysis")
        self.compute_button.clicked.connect(self._compute_analysis)
        self.remove_selected_button = QPushButton("Remove Selected Datasets")
        self.remove_selected_button.clicked.connect(self._remove_selected_datasets)
        self.remove_current_button = QPushButton("Remove Current Row")
        self.remove_current_button.clicked.connect(self._remove_current_dataset)
        self.ttest_factor_combo = QComboBox()
        self.ttest_factor_combo.addItems(["group_label", "genotype", "sex", "age", "region", "batch", "roi_class"])
        self.ttest_button = QPushButton("Two-Group Welch t-test")
        self.ttest_button.clicked.connect(self._run_ttest)
        self.anova_factor_combo = QComboBox()
        self.anova_factor_combo.addItems(["group_label", "genotype", "sex", "age", "region", "batch", "roi_class"])
        self.anova_button = QPushButton("One-way ANOVA")
        self.anova_button.clicked.connect(self._run_anova)
        self.blind_k_combo = QComboBox()
        self.blind_k_combo.addItems(["2", "3"])
        self.pca_button = QPushButton("Blind PCA / Clustering")
        self.pca_button.clicked.connect(self._run_blind_analysis)
        self.descriptive_button = QPushButton("Descriptive Statistics")
        self.descriptive_button.clicked.connect(self._run_descriptive_stats)
        self.normality_button = QPushButton("Normality & Equality of Variance")
        self.normality_button.clicked.connect(self._run_normality_and_variance)
        self.correlation_x_combo = QComboBox()
        self.correlation_y_combo = QComboBox()
        self.correlation_button = QPushButton("Correlation Coefficient")
        self.correlation_button.clicked.connect(self._run_correlation)

        self.export_roi_button = QPushButton("Export ROI Table CSV")
        self.export_roi_button.clicked.connect(lambda: self._export_table_csv("roi"))
        self.export_image_button = QPushButton("Export Image Table CSV")
        self.export_image_button.clicked.connect(lambda: self._export_table_csv("image"))
        self.export_animal_button = QPushButton("Export Animal Table CSV")
        self.export_animal_button.clicked.connect(lambda: self._export_table_csv("animal"))
        self.export_report_button = QPushButton("Export Stats Report")
        self.export_report_button.clicked.connect(self._export_stats_report)

        self.dataset_table = QTableWidget()
        self.dataset_table.cellChanged.connect(self._on_dataset_cell_changed)
        self.dataset_table.setAlternatingRowColors(False)

        self.roi_table = QTableWidget()
        self.image_table = QTableWidget()
        self.animal_table = QTableWidget()
        for table in (self.dataset_table, self.roi_table, self.image_table, self.animal_table):
            self._configure_table_palette(table)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.roi_table, "ROI Table")
        self.tabs.addTab(self.image_table, "Image Table")
        self.tabs.addTab(self.animal_table, "Animal Table")

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(300)
        self.stats_report = QTextEdit()
        self.stats_report.setReadOnly(True)
        self.stats_report.setMinimumHeight(260)
        self.stats_report.setPlainText("Add ROI datasets from Spectral Viewer, annotate metadata, then compute analysis.")

        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("Split λ (nm)"))
        top_controls.addWidget(self.split_edit)
        top_controls.addWidget(QLabel("Ratio mode"))
        top_controls.addWidget(self.ratio_mode_combo)
        top_controls.addWidget(QLabel("Measurement"))
        top_controls.addWidget(self.measurement_combo)
        top_controls.addWidget(QLabel("Stats level"))
        top_controls.addWidget(self.level_combo)
        top_controls.addWidget(self.normalize_checkbox)
        top_controls.addWidget(self.refresh_button)
        top_controls.addWidget(self.compute_button)
        top_controls.addWidget(self.remove_selected_button)
        top_controls.addWidget(self.remove_current_button)

        stats_controls = QHBoxLayout()
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
        export_controls.addWidget(self.export_report_button)

        stats_group = QGroupBox("Stats")
        stats_group_layout = QVBoxLayout()

        stats_options = QHBoxLayout()
        stats_options.addWidget(QLabel("Stats factor"))
        stats_options.addWidget(self.stats_factor_combo)
        stats_options.addWidget(QLabel("Significance level"))
        stats_options.addWidget(self.significance_edit)
        stats_options.addWidget(QLabel("Confidence interval %"))
        stats_options.addWidget(self.confidence_edit)
        stats_group_layout.addLayout(stats_options)

        stats_buttons = QHBoxLayout()
        stats_buttons.addWidget(self.descriptive_button)
        stats_buttons.addWidget(self.normality_button)
        stats_buttons.addWidget(self.ttest_button)
        stats_buttons.addWidget(QLabel("t-test factor"))
        stats_buttons.addWidget(self.ttest_factor_combo)
        stats_buttons.addWidget(QLabel("x"))
        stats_buttons.addWidget(self.correlation_x_combo)
        stats_buttons.addWidget(QLabel("y"))
        stats_buttons.addWidget(self.correlation_y_combo)
        stats_buttons.addWidget(self.correlation_button)
        stats_group_layout.addLayout(stats_buttons)
        stats_group.setLayout(stats_group_layout)

        layout = QVBoxLayout()
        layout.addLayout(top_controls)
        layout.addWidget(QLabel("Stored ROI Datasets"))
        layout.addWidget(self.dataset_table)
        layout.addWidget(stats_group)
        layout.addLayout(stats_controls)
        layout.addLayout(export_controls)
        layout.addWidget(self.tabs)
        layout.addWidget(self.stats_report)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        ROI_SPECTRUM_STORE.subscribe(self._on_store_changed)
        self._refresh_dataset_table()
        self._refresh_correlation_field_combos()
        self.level_combo.currentTextChanged.connect(lambda _text: self._refresh_correlation_field_combos())
        self.measurement_combo.currentTextChanged.connect(lambda _text: self._refresh_dataset_table())
        float_parent_dock_later(self)

    def closeEvent(self, event):
        ROI_SPECTRUM_STORE.unsubscribe(self._on_store_changed)
        super().closeEvent(event)

    def _configure_table_palette(self, table: QTableWidget):
        palette = table.palette()
        text_color = palette.color(QPalette.Text)
        base_color = palette.color(QPalette.Base)
        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            palette.setColor(group, QPalette.Text, text_color)
            palette.setColor(group, QPalette.WindowText, text_color)
            palette.setColor(group, QPalette.HighlightedText, text_color)
            palette.setColor(group, QPalette.Base, base_color)
            palette.setColor(group, QPalette.AlternateBase, base_color)
        table.setPalette(palette)

    def _style_table_item(self, table: QTableWidget, item: QTableWidgetItem):
        palette = table.palette()
        item.setForeground(palette.brush(QPalette.Text))
        item.setBackground(palette.brush(QPalette.Base))

    def _on_store_changed(self):
        self._refresh_dataset_table()

    def _refresh_dataset_table(self):
        selected_measurement = self.measurement_combo.currentText()
        datasets = [
            (index, dataset)
            for index, dataset in enumerate(ROI_SPECTRUM_STORE.list_datasets())
            if dataset.measurement_kind == selected_measurement
        ]
        self._updating_dataset_table = True
        self.dataset_table.clear()
        self.dataset_table.setColumnCount(len(self.DATASET_COLUMNS))
        self.dataset_table.setHorizontalHeaderLabels([column for column, _editable in self.DATASET_COLUMNS])
        self.dataset_table.setRowCount(len(datasets))
        for row_index, (store_index, dataset) in enumerate(datasets):
            values = {
                "use_for_analysis": dataset.use_for_analysis,
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "source_layer_name": dataset.source_layer_name,
                "trace_kind": dataset.trace_kind,
                "trace_label": dataset.trace_label,
                "animal_id": dataset.animal_id,
                "group_label": dataset.group_label,
                "genotype": dataset.genotype,
                "sex": dataset.sex,
                "age": dataset.age,
                "region": dataset.region,
                "batch": dataset.batch,
                "blind_id": dataset.blind_id,
                "roi_class": dataset.roi_class,
                "roi_count": str(dataset.roi_count or len(dataset.roi_labels)),
                "created_at": dataset.created_at,
            }
            for column_index, (column_name, editable) in enumerate(self.DATASET_COLUMNS):
                if column_name == "use_for_analysis":
                    item = QTableWidgetItem()
                    item.setCheckState(Qt.Checked if values[column_name] else Qt.Unchecked)
                else:
                    item = QTableWidgetItem(str(values[column_name]))
                    item.setData(Qt.UserRole, store_index)
                if column_name == "use_for_analysis":
                    item.setFlags((item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled) & ~Qt.ItemIsEditable)
                elif not editable:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._style_table_item(self.dataset_table, item)
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
        if column_name == "use_for_analysis":
            ROI_SPECTRUM_STORE.update_metadata(dataset_index, use_for_analysis=item.checkState() == Qt.Checked)
            return
        if not editable:
            return
        ROI_SPECTRUM_STORE.update_metadata(dataset_index, **{column_name: item.text().strip()})

    def _selected_datasets(self):
        selected_measurement = self.measurement_combo.currentText()
        datasets = [dataset for dataset in ROI_SPECTRUM_STORE.list_datasets() if dataset.measurement_kind == selected_measurement]
        return [dataset for dataset in datasets if dataset.use_for_analysis and dataset.accepted]

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
            self.stats_report.setPlainText("No selected ROI datasets available.")
            return
        if self.measurement_combo.currentText() == "spatial_ratio":
            self._compute_spatial_summary_analysis(datasets)
            return
        try:
            split_nm = self._split_value()
        except ValueError:
            self.stats_report.setPlainText("Split wavelength must be numeric.")
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
                        "roi_class": dataset.roi_class,
                        "roi_label": roi_label,
                        "ratio": ratio,
                    }
                )

            pooled_spectrum = (
                np.asarray(dataset.pooled_spectrum, dtype=np.float32)
                if dataset.pooled_spectrum is not None
                else np.mean(np.asarray(dataset.roi_spectra, dtype=np.float32), axis=0)
            )
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
                    "roi_class": dataset.roi_class,
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
                    "roi_class": dataset.roi_class,
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
                    "roi_class": entry["roi_class"],
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
        self._refresh_correlation_field_combos()
        self.stats_report.setPlainText(
            f"Computed ROI ratios at split {split_nm:.1f} nm for {len(roi_rows)} ROI(s), "
            f"{len(image_rows)} image set(s), and {len(animal_rows)} animal aggregate(s) from {len(datasets)} selected dataset(s)."
        )

    def _compute_spatial_summary_analysis(self, datasets):
        roi_rows = []
        image_rows = []
        animal_groups: dict[str, dict] = {}
        for dataset in datasets:
            base_row = {
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
                "roi_class": dataset.roi_class,
                "kernel_size": dataset.kernel_size,
                "split_nm": dataset.split_nm,
                "n_kernels": dataset.n_kernels,
                "mean_ratio": dataset.mean_ratio,
                "median_ratio": dataset.median_ratio,
                "sd_ratio": dataset.sd_ratio,
                "mean_intensity": dataset.mean_intensity,
            }
            if dataset.analysis_level == "roi":
                roi_rows.append(
                    {
                        **base_row,
                        "roi_label": dataset.trace_label or (dataset.roi_labels[0] if dataset.roi_labels else "ROI"),
                        "ratio": dataset.mean_ratio,
                    }
                )
            elif dataset.analysis_level == "image":
                image_rows.append(base_row)
                animal_key = dataset.animal_id.strip() or dataset.source_layer_name
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
                        "roi_class": dataset.roi_class,
                        "image_ratios": [],
                        "image_intensities": [],
                        "n_kernels": 0,
                        "kernel_size": dataset.kernel_size,
                        "split_nm": dataset.split_nm,
                    },
                )
                entry["image_ratios"].append(float(dataset.mean_ratio))
                entry["image_intensities"].append(float(dataset.mean_intensity))
                entry["n_kernels"] += int(dataset.n_kernels)

        animal_rows = []
        for animal_key, entry in animal_groups.items():
            ratios = np.asarray(entry["image_ratios"], dtype=np.float32)
            intensities = np.asarray(entry["image_intensities"], dtype=np.float32)
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
                    "roi_class": entry["roi_class"],
                    "kernel_size": entry["kernel_size"],
                    "split_nm": entry["split_nm"],
                    "n_images": len(ratios),
                    "n_kernels": int(entry["n_kernels"]),
                    "mean_ratio": float(np.mean(ratios)) if ratios.size else np.nan,
                    "median_ratio": float(np.median(ratios)) if ratios.size else np.nan,
                    "sd_ratio": float(np.std(ratios, ddof=1)) if ratios.size > 1 else 0.0,
                    "mean_intensity": float(np.mean(intensities)) if intensities.size else np.nan,
                }
            )

        self._roi_rows = roi_rows
        self._image_rows = image_rows
        self._animal_rows = animal_rows
        self._animal_spectra = []
        self._populate_table(self.roi_table, roi_rows)
        self._populate_table(self.image_table, image_rows)
        self._populate_table(self.animal_table, animal_rows)
        self._refresh_correlation_field_combos()
        self.stats_report.setPlainText(
            f"Loaded spatial-ratio summaries for {len(roi_rows)} ROI row(s), {len(image_rows)} image row(s), and {len(animal_rows)} animal aggregate(s)."
        )

    def _remove_selected_datasets(self):
        datasets = ROI_SPECTRUM_STORE.list_datasets()
        indices = [index for index, dataset in enumerate(datasets) if dataset.use_for_analysis]
        if not indices:
            self.stats_report.setPlainText("No selected datasets to remove.")
            return
        ROI_SPECTRUM_STORE.remove_datasets(indices)
        self._roi_rows = []
        self._image_rows = []
        self._animal_rows = []
        self._animal_spectra = []
        self._populate_table(self.roi_table, [])
        self._populate_table(self.image_table, [])
        self._populate_table(self.animal_table, [])
        self._refresh_dataset_table()
        self.stats_report.setPlainText(f"Removed {len(indices)} selected dataset(s) from memory.")

    def _remove_current_dataset(self):
        row = self.dataset_table.currentRow()
        if row < 0:
            self.stats_report.setPlainText("Select a dataset row to remove.")
            return
        item = self.dataset_table.item(row, 1)
        if item is None:
            self.stats_report.setPlainText("Select a valid dataset row to remove.")
            return
        dataset_index = item.data(Qt.UserRole)
        dataset = ROI_SPECTRUM_STORE.get_dataset(dataset_index)
        ROI_SPECTRUM_STORE.remove_dataset(dataset_index)
        self._roi_rows = []
        self._image_rows = []
        self._animal_rows = []
        self._animal_spectra = []
        self._populate_table(self.roi_table, [])
        self._populate_table(self.image_table, [])
        self._populate_table(self.animal_table, [])
        self._refresh_dataset_table()
        self.stats_report.setPlainText(f"Removed dataset {dataset.dataset_id} from memory.")

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
                self._style_table_item(table, item)
                table.setItem(row_index, column_index, item)
        table.resizeColumnsToContents()

    def _selected_level_rows(self) -> list[dict]:
        level = self.level_combo.currentText()
        if level == "ROI":
            return self._roi_rows
        if level == "Image":
            return self._image_rows
        return self._animal_rows

    def _alpha_value(self) -> float:
        return float(self.significance_edit.text().strip())

    def _confidence_fraction(self) -> float:
        return float(self.confidence_edit.text().strip()) / 100.0

    def _set_stats_text(self, title: str, lines: list[str]):
        self.stats_report.setPlainText("\n".join([title, *lines]))

    def _finalize_analysis_axis(self, axis, *, title: str, xlabel: str, ylabel: str):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        legend = axis.legend(fontsize=8)
        if legend is not None:
            legend.set_draggable(True)
        self.figure.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.16)

    def _metric_key_for_rows(self, rows: list[dict]) -> str:
        return "mean_ratio" if rows and "mean_ratio" in rows[0] else "ratio"

    def _group_metric_values(self, factor: str) -> tuple[list[dict], dict[str, np.ndarray], str]:
        rows = self._selected_level_rows()
        metric_key = self._metric_key_for_rows(rows)
        grouped: dict[str, list[float]] = {}
        for row in rows:
            label = str(row.get(factor, "")).strip()
            value = row.get(metric_key)
            if not label or value is None:
                continue
            try:
                grouped.setdefault(label, []).append(float(value))
            except (TypeError, ValueError):
                continue
        grouped_arrays = {
            label: np.asarray(values, dtype=np.float32)
            for label, values in grouped.items()
            if len(values) > 0
        }
        return rows, grouped_arrays, metric_key

    def _refresh_correlation_field_combos(self):
        rows = self._selected_level_rows()
        numeric_fields: list[str] = []
        if rows:
            sample = rows[0]
            for key in sample.keys():
                values = []
                for row in rows:
                    value = row.get(key)
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        values = []
                        break
                if values:
                    numeric_fields.append(key)
        self.correlation_x_combo.blockSignals(True)
        self.correlation_y_combo.blockSignals(True)
        self.correlation_x_combo.clear()
        self.correlation_y_combo.clear()
        for field in numeric_fields:
            self.correlation_x_combo.addItem(field)
            self.correlation_y_combo.addItem(field)
        if "ratio" in numeric_fields:
            self.correlation_x_combo.setCurrentText("ratio")
        elif "mean_ratio" in numeric_fields:
            self.correlation_x_combo.setCurrentText("mean_ratio")
        if "n_roi" in numeric_fields:
            self.correlation_y_combo.setCurrentText("n_roi")
        elif "n_rois" in numeric_fields:
            self.correlation_y_combo.setCurrentText("n_rois")
        self.correlation_x_combo.blockSignals(False)
        self.correlation_y_combo.blockSignals(False)

    def _export_stats_report(self):
        report_text = self.stats_report.toPlainText().strip()
        if not report_text:
            self.stats_report.setPlainText("No statistics report available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Stats Report",
            "stats_report.txt",
            "Text files (*.txt);;CSV files (*.csv)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() == ".csv":
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for line in report_text.splitlines():
                    writer.writerow([line])
        else:
            output_path.write_text(report_text, encoding="utf-8")
        self.stats_report.setPlainText(report_text + f"\n\nExported report to {output_path.name}")

    def _run_descriptive_stats(self):
        factor = self.stats_factor_combo.currentText()
        _rows, grouped, metric_key = self._group_metric_values(factor)
        if not grouped:
            self.stats_report.setPlainText(f"No numeric {metric_key} values available for descriptive statistics by {factor}.")
            return
        confidence = self._confidence_fraction()
        alpha = 1.0 - confidence
        lines = [f"Factor: {factor}", f"Metric: {metric_key}", f"Confidence interval: {confidence * 100:.1f}%"]
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        labels = []
        plot_values = []
        for label, values in grouped.items():
            mean_val = float(np.mean(values))
            sd_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            sem_val = float(sd_val / np.sqrt(len(values))) if len(values) > 1 else 0.0
            median_val = float(np.median(values))
            q1, q3 = np.percentile(values, [25, 75])
            if len(values) > 1:
                ci_half = float(stats.t.ppf(1.0 - alpha / 2.0, df=len(values) - 1) * sem_val)
            else:
                ci_half = 0.0
            lines.append(
                f"{label}: n={len(values)}, mean={mean_val:.4f}, sd={sd_val:.4f}, sem={sem_val:.4f}, "
                f"median={median_val:.4f}, IQR=({q1:.4f}, {q3:.4f}), CI=({mean_val - ci_half:.4f}, {mean_val + ci_half:.4f})"
            )
            labels.append(label)
            plot_values.append(values)
        axis.boxplot(plot_values, tick_labels=labels)
        self._finalize_analysis_axis(
            axis,
            title="Descriptive statistics",
            xlabel=factor,
            ylabel=metric_key,
        )
        self.canvas.draw()
        self._set_stats_text("Descriptive statistics", lines)

    def _run_normality_and_variance(self):
        factor = self.stats_factor_combo.currentText()
        _rows, grouped, metric_key = self._group_metric_values(factor)
        if len(grouped) < 2:
            self.stats_report.setPlainText(f"Need at least two non-empty {factor} groups for normality and variance checks.")
            return
        alpha = self._alpha_value()
        normal_flags = []
        lines = [f"Factor: {factor}", f"Metric: {metric_key}", f"Significance level: {alpha:.3f}", ""]
        lines.append("--- Shapiro-Wilk normality test (normal if p > alpha) ---")
        valid_groups = []
        for label, values in grouped.items():
            if len(values) < 3:
                lines.append(f"{label}: n={len(values)} is too small for Shapiro-Wilk.")
                continue
            statistic, pvalue = stats.shapiro(values)
            is_normal = pvalue > alpha
            normal_flags.append(is_normal)
            valid_groups.append(values)
            lines.append(f"{label}: W={statistic:.4f}, p={pvalue:.4g}, {'NORMAL' if is_normal else 'NOT normal'}")
        if len(valid_groups) < 2:
            self._set_stats_text("Normality & equality of variance", lines + ["Not enough groups passed minimum size for variance testing."])
            return
        lines.append("")
        lines.append("--- Homogeneity of variances (equal if p > alpha) ---")
        bart_stat, bart_p = stats.bartlett(*valid_groups)
        lev_stat, lev_p = stats.levene(*valid_groups, center="median")
        lines.append(f"Bartlett: statistic={bart_stat:.4f}, p={bart_p:.4g}, {'EQUAL' if bart_p > alpha else 'NOT equal'}")
        lines.append(f"Levene: statistic={lev_stat:.4f}, p={lev_p:.4g}, {'EQUAL' if lev_p > alpha else 'NOT equal'}")
        lines.append("")
        if normal_flags and all(normal_flags) and bart_p > alpha and lev_p > alpha:
            lines.append("Conclusion: groups are approximately normal and have equal variances. Parametric tests are appropriate.")
        else:
            lines.append("Conclusion: use parametric tests only with caution; otherwise nonparametric methods are more appropriate.")
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        labels = list(grouped.keys())
        box_values = [grouped[label] for label in labels]
        axis.boxplot(box_values, tick_labels=labels)
        self._finalize_analysis_axis(
            axis,
            title="Normality and variance check",
            xlabel=factor,
            ylabel=metric_key,
        )
        self.canvas.draw()
        self._set_stats_text("Normality & equality of variance", lines)

    def _run_ttest(self):
        rows = self._selected_level_rows()
        if not rows:
            self.stats_report.setPlainText("Compute analysis before running t-test.")
            return
        metric_key = "mean_ratio" if rows and "mean_ratio" in rows[0] else "ratio"
        factor = self.ttest_factor_combo.currentText()
        grouped: dict[str, list[float]] = {}
        for row in rows:
            label = str(row.get(factor, "")).strip()
            if not label:
                continue
            grouped.setdefault(label, []).append(float(row[metric_key]))
        valid = [(label, values) for label, values in grouped.items() if len(values) >= 2]
        if len(valid) != 2:
            self.stats_report.setPlainText(
                f"Welch t-test requires exactly two non-empty {factor} groups with at least two values each."
            )
            return
        (label_a, values_a), (label_b, values_b) = valid
        statistic, pvalue = self._welch_ttest_permutation(
            np.asarray(values_a, dtype=np.float32),
            np.asarray(values_b, dtype=np.float32),
        )
        self._set_stats_text(
            "Welch t-test",
            [
                f"Level: {self.level_combo.currentText()}",
                f"Factor: {factor}",
                f"Groups: {label_a} (n={len(values_a)}) vs {label_b} (n={len(values_b)})",
                f"t = {statistic:.4f}",
                f"Permutation p = {pvalue:.4g}",
            ],
        )
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.boxplot([values_a, values_b], tick_labels=[label_a, label_b])
        self._finalize_analysis_axis(
            axis,
            title="Two-group comparison",
            xlabel=factor,
            ylabel=metric_key,
        )
        self.canvas.draw()

    def _run_anova(self):
        rows = self._selected_level_rows()
        if not rows:
            self.stats_report.setPlainText("Compute analysis before running ANOVA.")
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
            self.stats_report.setPlainText(f"Need at least two non-empty {factor} groups with >=2 values each for ANOVA.")
            return
        statistic, pvalue = self._anova_permutation([np.asarray(values, dtype=np.float32) for _label, values in valid])
        summary = ", ".join(f"{label}:n={len(values)}" for label, values in valid)
        self._set_stats_text(
            "One-way ANOVA",
            [
                f"Factor: {factor}",
                f"Level: {self.level_combo.currentText()}",
                f"F = {statistic:.4f}",
                f"Permutation p = {pvalue:.4g}",
                f"Groups: {summary}",
            ],
        )

    def _run_blind_analysis(self):
        if not self._animal_rows or not self._animal_spectra:
            self.stats_report.setPlainText("Compute analysis before running blind PCA.")
            return
        spectra = np.stack(self._animal_spectra, axis=0)
        if spectra.shape[0] < 2:
            self.stats_report.setPlainText("Need at least two animal spectra for blind PCA.")
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
        pc1_label = f"PC1 ({(singular_values[0] ** 2 / total_var) * 100:.1f}%)"
        pc2_pct = (singular_values[1] ** 2 / total_var) * 100 if singular_values.shape[0] > 1 else 0.0
        pc2_label = f"PC2 ({pc2_pct:.1f}%)"
        self._finalize_analysis_axis(
            axis,
            title="Blind PCA clustering",
            xlabel=pc1_label,
            ylabel=pc2_label,
        )
        self.canvas.draw()
        self._set_stats_text(
            "Blind PCA / Clustering",
            [
                f"Animals analyzed: {spectra.shape[0]}",
                f"Clusters: {int(self.blind_k_combo.currentText())}",
                f"Separation = {separation:.4f}",
                f"Permutation p = {pvalue:.4g}",
            ],
        )

    def _run_correlation(self):
        rows = self._selected_level_rows()
        if not rows:
            self.stats_report.setPlainText("Compute analysis before running correlation.")
            return
        x_key = self.correlation_x_combo.currentText()
        y_key = self.correlation_y_combo.currentText()
        if not x_key or not y_key:
            self.stats_report.setPlainText("Select two numeric fields for correlation.")
            return
        x_values = []
        y_values = []
        for row in rows:
            try:
                x_values.append(float(row[x_key]))
                y_values.append(float(row[y_key]))
            except (TypeError, ValueError, KeyError):
                continue
        if len(x_values) < 3:
            self.stats_report.setPlainText("Need at least three paired numeric values for correlation.")
            return
        x = np.asarray(x_values, dtype=np.float32)
        y = np.asarray(y_values, dtype=np.float32)
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.scatter(x, y, label=f"n={len(x)}")
        coeffs = np.polyfit(x, y, 1)
        fit_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        fit_y = coeffs[0] * fit_x + coeffs[1]
        axis.plot(fit_x, fit_y, linestyle="--", color="black", label="Linear fit")
        self._finalize_analysis_axis(
            axis,
            title="Correlation",
            xlabel=x_key,
            ylabel=y_key,
        )
        self.canvas.draw()
        self._set_stats_text(
            "Correlation coefficient",
            [
                f"Level: {self.level_combo.currentText()}",
                f"x = {x_key}",
                f"y = {y_key}",
                f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.4g}",
                f"Spearman rho = {spearman_rho:.4f}, p = {spearman_p:.4g}",
            ],
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
            self.stats_report.setPlainText(f"No {table_name} analysis table available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, f"Export {table_name.title()} Table CSV", f"{table_name}_analysis.csv", "CSV files (*.csv)")
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        self.stats_report.setPlainText(f"Exported {table_name} analysis table to {Path(path).name}")
