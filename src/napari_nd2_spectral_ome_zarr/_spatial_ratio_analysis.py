from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.widgets import RectangleSelector
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ._qt_utils import float_parent_dock_later
from ._roi_store import ROI_SPECTRUM_STORE


class SpatialRatioAnalysisWidget(QWidget):
    ROI_LAYER_SUFFIX = " ROI"
    CONTEXT_COLUMNS = ["image", "roi layer", "labels layer", "mode", "roi count", "status"]
    SUMMARY_COLUMNS = ["level", "image", "roi", "kernel", "n_kernels", "mean_ratio", "median_ratio", "sd_ratio", "mean_intensity"]

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._context_rows: list[str] = []
        self._kernel_rows: list[dict] = []
        self._summary_rows: list[dict] = []
        self._selection_overlay_name = "Spatial Ratio Selection"

        self.kernel_combo = QComboBox()
        self.kernel_combo.setEditable(True)
        for value in ("1", "2", "3", "5", "7", "10", "15", "20"):
            self.kernel_combo.addItem(value)
        self.kernel_combo.setCurrentText("3")
        self.min_coverage_edit = QLineEdit("0.5")
        self.split_edit = QLineEdit("600")
        self.ratio_mode_combo = QComboBox()
        self.ratio_mode_combo.addItems(["sum_above_over_below", "mean_above_over_below", "log10_sum_ratio"])
        self.normalize_checkbox = QCheckBox("Normalize before ratio")
        self.blank_subtract_checkbox = QCheckBox("Subtract blank reference")
        self.blank_subtract_checkbox.setChecked(False)
        self.blank_reference_combo = QComboBox()
        self.blank_reference_combo.setMinimumContentsLength(18)

        self.context_table = QTableWidget()
        self.context_table.setColumnCount(len(self.CONTEXT_COLUMNS))
        self.context_table.setHorizontalHeaderLabels(self.CONTEXT_COLUMNS)
        self.context_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.context_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.context_table.itemSelectionChanged.connect(self._focus_selected_context_image)

        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(len(self.SUMMARY_COLUMNS))
        self.summary_table.setHorizontalHeaderLabels(self.SUMMARY_COLUMNS)
        self.summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.summary_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.summary_table.setMinimumHeight(220)

        self.prepare_roi_button = QPushButton("Prepare Selected ROI")
        self.prepare_roi_button.clicked.connect(self._prepare_selected_roi_layer)
        self.bind_labels_button = QPushButton("Bind Active Labels")
        self.bind_labels_button.clicked.connect(self._bind_active_labels_layer)
        self.focus_image_button = QPushButton("Focus Image")
        self.focus_image_button.clicked.connect(self._focus_selected_context_image)
        self.compute_button = QPushButton("Compute Spatial Ratio")
        self.compute_button.clicked.connect(self._compute_spatial_ratio)
        self.clear_selection_button = QPushButton("Clear Scatter Selection")
        self.clear_selection_button.clicked.connect(self._clear_scatter_selection)
        self.export_raw_button = QPushButton("Export Raw Kernel CSV")
        self.export_raw_button.clicked.connect(self._export_raw_kernel_csv)
        self.export_summary_button = QPushButton("Export Summary CSV")
        self.export_summary_button.clicked.connect(self._export_summary_csv)
        self.send_summary_button = QPushButton("Send Summary To Analysis")
        self.send_summary_button.clicked.connect(self._send_summary_to_analysis)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(320)
        self._selector = None

        self.stats_report = QTextEdit()
        self.stats_report.setReadOnly(True)
        self.stats_report.setMinimumHeight(180)
        self.stats_report.setPlainText(
            "Select an image ROI context, choose kernel size and split wavelength, then compute spatial ratio."
        )

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Kernel size"))
        controls.addWidget(self.kernel_combo)
        controls.addWidget(QLabel("Min valid coverage"))
        controls.addWidget(self.min_coverage_edit)
        controls.addWidget(QLabel("Split λ (nm)"))
        controls.addWidget(self.split_edit)
        controls.addWidget(QLabel("Ratio mode"))
        controls.addWidget(self.ratio_mode_combo)
        controls.addWidget(self.normalize_checkbox)
        controls.addWidget(self.blank_subtract_checkbox)
        controls.addWidget(QLabel("Blank image"))
        controls.addWidget(self.blank_reference_combo)
        controls.addWidget(self.compute_button)

        context_buttons = QHBoxLayout()
        context_buttons.addWidget(self.prepare_roi_button)
        context_buttons.addWidget(self.bind_labels_button)
        context_buttons.addWidget(self.focus_image_button)

        export_buttons = QHBoxLayout()
        export_buttons.addWidget(self.clear_selection_button)
        export_buttons.addWidget(self.export_raw_button)
        export_buttons.addWidget(self.export_summary_button)
        export_buttons.addWidget(self.send_summary_button)

        context_group = QGroupBox("Image ROI Context")
        context_layout = QVBoxLayout()
        context_layout.addLayout(context_buttons)
        context_layout.addWidget(self.context_table)
        context_group.setLayout(context_layout)

        summary_group = QGroupBox("Kernel Summary")
        summary_layout = QVBoxLayout()
        summary_layout.addLayout(export_buttons)
        summary_layout.addWidget(self.summary_table)
        summary_group.setLayout(summary_layout)

        layout = QVBoxLayout()
        layout.addWidget(context_group)
        layout.addLayout(controls)
        layout.addWidget(self.canvas)
        layout.addWidget(summary_group)
        layout.addWidget(self.stats_report)
        self.setLayout(layout)

        if hasattr(self.viewer, "layers") and hasattr(self.viewer.layers, "events"):
            self.viewer.layers.events.inserted.connect(self._on_layers_changed)
            self.viewer.layers.events.removed.connect(self._on_layers_changed)
        selection_events = getattr(getattr(self.viewer.layers, "selection", None), "events", None)
        if selection_events is not None and hasattr(selection_events, "active"):
            selection_events.active.connect(lambda _event=None: self._refresh_context_table())
        self.blank_subtract_checkbox.stateChanged.connect(lambda _value=None: self.stats_report.setPlainText("Blank subtraction setting updated. Recompute spatial ratio to apply it."))
        self.blank_reference_combo.currentTextChanged.connect(lambda _text=None: self.stats_report.setPlainText("Blank reference updated. Recompute spatial ratio to apply it."))
        self._refresh_blank_reference_combo()
        self._refresh_context_table()
        float_parent_dock_later(self)

    def _find_layer_by_name(self, name: str):
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        return None

    def _spectral_layers(self):
        spectral_layers = []
        for layer in self.viewer.layers:
            metadata = getattr(layer, "metadata", {})
            cube = metadata.get("spectral_cube")
            wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
            if cube is not None and wavelengths.size > 0:
                spectral_layers.append(layer)
        return spectral_layers

    def _roi_layer_name(self, spectral_layer_name: str) -> str:
        return f"{spectral_layer_name}{self.ROI_LAYER_SUFFIX}"

    def _find_roi_shapes_layer(self, spectral_layer_name: str):
        return self._find_layer_by_name(self._roi_layer_name(spectral_layer_name))

    def _find_bound_labels_layer(self, spectral_layer_name: str):
        for layer in self.viewer.layers:
            if layer.__class__.__name__ != "Labels":
                continue
            if getattr(layer, "metadata", {}).get("source_spectral_layer_name") == spectral_layer_name:
                return layer
        return None

    def _selected_context_source_layer_name(self) -> str:
        selection_model = self.context_table.selectionModel()
        if selection_model is not None:
            selected_rows = selection_model.selectedRows()
            if selected_rows:
                item = self.context_table.item(selected_rows[0].row(), 0)
                if item is not None:
                    return str(item.data(Qt.UserRole) or item.text())
        raise ValueError("Choose an image context row first.")

    def _refresh_context_table(self, event=None):
        del event
        spectral_layers = self._spectral_layers()
        self.context_table.blockSignals(True)
        self.context_table.clearContents()
        self.context_table.setRowCount(len(spectral_layers))
        self._context_rows = []
        for row_index, layer in enumerate(spectral_layers):
            metadata = getattr(layer, "metadata", {})
            cube = np.asarray(metadata.get("spectral_cube"), dtype=np.float32)
            roi_layer = self._find_roi_shapes_layer(layer.name)
            labels_layer = self._find_bound_labels_layer(layer.name)
            mode_label, roi_count, status = self._context_summary(layer.name, cube)
            values = [
                layer.name,
                "" if roi_layer is None else roi_layer.name,
                "" if labels_layer is None else labels_layer.name,
                mode_label,
                str(roi_count),
                status,
            ]
            self._context_rows.append(layer.name)
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setData(Qt.UserRole, layer.name)
                self.context_table.setItem(row_index, column_index, item)
        self.context_table.resizeColumnsToContents()
        if spectral_layers:
            self.context_table.selectRow(0)
        self.context_table.blockSignals(False)

    def _on_layers_changed(self, event=None):
        del event
        self._refresh_blank_reference_combo()
        self._refresh_context_table()

    def _refresh_blank_reference_combo(self):
        current_name = self.blank_reference_combo.currentData()
        spectral_layers = self._spectral_layers()
        self.blank_reference_combo.blockSignals(True)
        self.blank_reference_combo.clear()
        self.blank_reference_combo.addItem("None", userData=None)
        selected_index = 0
        for index, layer in enumerate(spectral_layers, start=1):
            self.blank_reference_combo.addItem(layer.name, userData=layer.name)
            if layer.name == current_name:
                selected_index = index
        self.blank_reference_combo.setCurrentIndex(selected_index)
        self.blank_reference_combo.blockSignals(False)

    def _context_summary(self, spectral_layer_name: str, cube: np.ndarray) -> tuple[str, int, str]:
        roi_layer = self._find_roi_shapes_layer(spectral_layer_name)
        labels_layer = self._find_bound_labels_layer(spectral_layer_name)
        roi_items = self._collect_roi_items_for_layer(spectral_layer_name, cube)
        if roi_layer is not None and labels_layer is not None:
            return "Shapes + Labels", len(roi_items), "Kernel analysis uses labels inside shapes"
        if roi_layer is not None:
            return "Shapes only", len(roi_items), "Kernel analysis uses shape area"
        if labels_layer is not None:
            return "Labels only", len(roi_items), "Full image is implicit boundary"
        return "No ROI", 0, "Prepare shapes or bind labels"

    def _prepare_selected_roi_layer(self):
        try:
            spectral_layer = self._find_layer_by_name(self._selected_context_source_layer_name())
            if spectral_layer is None:
                raise ValueError("Selected image is no longer available.")
            roi_layer = self._find_roi_shapes_layer(spectral_layer.name)
            if roi_layer is None:
                roi_layer = self.viewer.add_shapes(
                    name=self._roi_layer_name(spectral_layer.name),
                    edge_color="yellow",
                    face_color="transparent",
                    edge_width=2,
                    metadata={"source_spectral_layer_name": spectral_layer.name},
                )
            self.viewer.layers.selection.active = roi_layer
            self._refresh_context_table()
            self.stats_report.setPlainText(
                f"Prepared ROI layer '{roi_layer.name}'. Draw shapes for spatial kernel analysis."
            )
        except Exception as exc:
            self.stats_report.setPlainText(str(exc))

    def _bind_active_labels_layer(self):
        try:
            spectral_layer_name = self._selected_context_source_layer_name()
            spectral_layer = self._find_layer_by_name(spectral_layer_name)
            active_layer = getattr(self.viewer.layers.selection, "active", None)
            if active_layer is None or active_layer.__class__.__name__ != "Labels":
                raise ValueError("Select a Labels layer in napari first.")
            metadata = getattr(spectral_layer, "metadata", {})
            cube = np.asarray(metadata.get("spectral_cube"), dtype=np.float32)
            labels = np.asarray(getattr(active_layer, "data", []))
            labels = np.squeeze(labels)
            if labels.ndim != 2 or labels.shape != cube.shape[1:]:
                raise ValueError("Labels layer shape must match the image YX shape.")
            active_layer.metadata = {
                **getattr(active_layer, "metadata", {}),
                "source_spectral_layer_name": spectral_layer_name,
            }
            self.viewer.layers.selection.active = spectral_layer
            self._refresh_context_table()
            self.stats_report.setPlainText(
                f"Bound labels layer '{active_layer.name}' to '{spectral_layer_name}' for spatial kernel analysis."
            )
        except Exception as exc:
            self.stats_report.setPlainText(str(exc))

    def _focus_selected_context_image(self):
        try:
            spectral_layer = self._find_layer_by_name(self._selected_context_source_layer_name())
            if spectral_layer is not None:
                self.viewer.layers.selection.active = spectral_layer
        except Exception:
            return

    def _labels_array_for_cube(self, labels_layer, cube: np.ndarray) -> np.ndarray:
        labels_data = np.asarray(getattr(labels_layer, "data", []))
        labels_data = np.squeeze(labels_data)
        expected_shape = tuple(int(value) for value in cube.shape[1:])
        if labels_data.ndim != 2 or tuple(int(value) for value in labels_data.shape) != expected_shape:
            raise ValueError(
                f"Labels layer '{labels_layer.name}' shape {tuple(labels_data.shape)} does not match image shape {expected_shape}."
            )
        return labels_data

    def _polygon_mask(self, vertices: np.ndarray, height: int, width: int) -> np.ndarray:
        if height <= 0 or width <= 0:
            return np.zeros((0, 0), dtype=bool)
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        points = np.column_stack(((grid_y + 0.5).ravel(), (grid_x + 0.5).ravel()))
        path = MplPath(vertices[:, :2], closed=True)
        return path.contains_points(points, radius=1e-9).reshape(height, width)

    def _build_shape_roi_item(self, cube: np.ndarray, roi_layer, shape_index: int, vertices: np.ndarray):
        if vertices.ndim != 2 or vertices.shape[1] < 2:
            return None
        y_min = max(0, int(np.floor(vertices[:, 0].min())))
        y_max = min(cube.shape[1], int(np.ceil(vertices[:, 0].max())))
        x_min = max(0, int(np.floor(vertices[:, 1].min())))
        x_max = min(cube.shape[2], int(np.ceil(vertices[:, 1].max())))
        if y_max <= y_min or x_max <= x_min:
            return None
        local_vertices = np.column_stack((vertices[:, 0] - y_min, vertices[:, 1] - x_min))
        mask = self._polygon_mask(local_vertices, y_max - y_min, x_max - x_min)
        if int(mask.sum()) <= 0:
            return None
        return {
            "roi_label": f"ROI {shape_index + 1}",
            "y_min": y_min,
            "y_max": y_max,
            "x_min": x_min,
            "x_max": x_max,
            "mask": mask,
        }

    def _collect_roi_items_for_layer(self, spectral_layer_name: str, cube: np.ndarray) -> list[dict]:
        roi_layer = self._find_roi_shapes_layer(spectral_layer_name)
        labels_layer = self._find_bound_labels_layer(spectral_layer_name)
        roi_items: list[dict] = []
        if roi_layer is not None and len(getattr(roi_layer, "data", [])) > 0:
            labels_data = None if labels_layer is None else self._labels_array_for_cube(labels_layer, cube)
            for shape_index, shape in enumerate(roi_layer.data):
                vertices = np.asarray(shape)
                item = self._build_shape_roi_item(cube, roi_layer, shape_index, vertices)
                if item is None:
                    continue
                if labels_data is not None:
                    local_labels = labels_data[item["y_min"]:item["y_max"], item["x_min"]:item["x_max"]]
                    item["mask"] = np.logical_and(item["mask"], local_labels > 0)
                    if int(item["mask"].sum()) <= 0:
                        continue
                roi_items.append(item)
            return roi_items
        if labels_layer is not None:
            labels_data = self._labels_array_for_cube(labels_layer, cube)
            for label_value in sorted(int(v) for v in np.unique(labels_data) if int(v) > 0):
                mask = labels_data == label_value
                coords = np.argwhere(mask)
                if coords.size == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0) + 1
                local_mask = mask[y_min:y_max, x_min:x_max]
                roi_items.append(
                    {
                        "roi_label": f"Label {label_value}",
                        "y_min": int(y_min),
                        "y_max": int(y_max),
                        "x_min": int(x_min),
                        "x_max": int(x_max),
                        "mask": local_mask,
                    }
                )
        return roi_items

    def _kernel_size_value(self) -> int:
        value = int(self.kernel_combo.currentText().strip())
        if value < 1 or value > 20:
            raise ValueError("Kernel size must be between 1 and 20.")
        return value

    def _min_coverage_value(self) -> float:
        value = float(self.min_coverage_edit.text().strip())
        if value <= 0.0 or value > 1.0:
            raise ValueError("Min valid coverage must be between 0 and 1.")
        return value

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

    def _blank_reference_layer(self, source_layer_name: str):
        if not self.blank_subtract_checkbox.isChecked():
            return None
        blank_layer_name = self.blank_reference_combo.currentData()
        if not blank_layer_name:
            return None
        if str(blank_layer_name) == source_layer_name:
            raise ValueError("Blank reference image must be different from the source image.")
        layer = self._find_layer_by_name(str(blank_layer_name))
        if layer is None:
            raise ValueError("Selected blank reference image is no longer available.")
        metadata = getattr(layer, "metadata", {})
        cube = metadata.get("spectral_cube")
        wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
        if cube is None or wavelengths.size == 0:
            raise ValueError("Selected blank reference does not contain spectral metadata.")
        return np.asarray(cube, dtype=np.float32), wavelengths

    def _apply_blank_reference_to_spectrum(
        self,
        spectrum: np.ndarray,
        *,
        source_layer_name: str,
        wavelengths: np.ndarray,
        source_cube_shape: tuple[int, ...],
        y_min: int | None = None,
        y_max: int | None = None,
        x_min: int | None = None,
        x_max: int | None = None,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        blank_reference = self._blank_reference_layer(source_layer_name)
        if blank_reference is None:
            return np.asarray(spectrum, dtype=np.float32)
        blank_cube, blank_wavelengths = blank_reference
        if blank_wavelengths.size != wavelengths.size:
            raise ValueError("Blank reference wavelength count does not match the source image.")
        if source_cube_shape[0] != blank_cube.shape[0]:
            raise ValueError("Blank reference spectral-bin count does not match the source image.")
        if blank_cube.shape[1:] == tuple(int(v) for v in source_cube_shape[1:]) and None not in {y_min, y_max, x_min, x_max}:
            local_blank = blank_cube[:, y_min:y_max, x_min:x_max]
            if mask is not None:
                blank_pixels = local_blank[:, mask]
            else:
                blank_pixels = local_blank.reshape(local_blank.shape[0], -1)
            blank_spectrum = blank_pixels.mean(axis=1)
        else:
            blank_spectrum = blank_cube.mean(axis=(1, 2))
        return np.asarray(spectrum, dtype=np.float32) - np.asarray(blank_spectrum, dtype=np.float32)

    def _compute_spatial_ratio(self):
        try:
            spectral_layer_name = self._selected_context_source_layer_name()
            spectral_layer = self._find_layer_by_name(spectral_layer_name)
            if spectral_layer is None:
                raise ValueError("Selected image is no longer available.")
            metadata = getattr(spectral_layer, "metadata", {})
            cube = np.asarray(metadata.get("spectral_cube"), dtype=np.float32)
            wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
            if cube.ndim != 3 or wavelengths.size == 0:
                raise ValueError("Selected image does not contain spectral cube metadata.")
            kernel_size = self._kernel_size_value()
            min_coverage = self._min_coverage_value()
            split_nm = self._split_value()
            roi_items = self._collect_roi_items_for_layer(spectral_layer_name, cube)
            if not roi_items:
                raise ValueError("No valid ROI or label regions found for spatial kernel analysis.")

            kernel_rows: list[dict] = []
            summary_rows: list[dict] = []
            kernel_counter = 0
            for roi_item in roi_items:
                roi_spectra = []
                roi_ratios = []
                roi_intensities = []
                roi_mask = roi_item["mask"]
                roi_cube = cube[:, roi_item["y_min"]:roi_item["y_max"], roi_item["x_min"]:roi_item["x_max"]]
                for y0 in range(0, roi_mask.shape[0], kernel_size):
                    for x0 in range(0, roi_mask.shape[1], kernel_size):
                        kernel_mask = roi_mask[y0:y0 + kernel_size, x0:x0 + kernel_size]
                        if kernel_mask.size == 0:
                            continue
                        valid_fraction = float(kernel_mask.mean())
                        if valid_fraction < min_coverage or int(kernel_mask.sum()) == 0:
                            continue
                        kernel_cube = roi_cube[:, y0:y0 + kernel_size, x0:x0 + kernel_size]
                        spectrum = np.asarray(kernel_cube[:, kernel_mask], dtype=np.float32).mean(axis=1)
                        corrected_spectrum = self._apply_blank_reference_to_spectrum(
                            spectrum,
                            source_layer_name=spectral_layer_name,
                            wavelengths=wavelengths,
                            source_cube_shape=tuple(int(v) for v in cube.shape),
                            y_min=int(roi_item["y_min"] + y0),
                            y_max=int(roi_item["y_min"] + y0 + kernel_mask.shape[0]),
                            x_min=int(roi_item["x_min"] + x0),
                            x_max=int(roi_item["x_min"] + x0 + kernel_mask.shape[1]),
                            mask=kernel_mask,
                        )
                        ratio = self._compute_ratio(wavelengths, corrected_spectrum, split_nm)
                        intensity = float(np.sum(corrected_spectrum) / len(corrected_spectrum))
                        kernel_counter += 1
                        row = {
                            "kernel_id": kernel_counter,
                            "image": spectral_layer_name,
                            "roi": roi_item["roi_label"],
                            "kernel_size": kernel_size,
                            "center_y": float(roi_item["y_min"] + y0 + (kernel_mask.shape[0] / 2.0)),
                            "center_x": float(roi_item["x_min"] + x0 + (kernel_mask.shape[1] / 2.0)),
                            "ratio": ratio,
                            "intensity": intensity,
                            "y_min": int(roi_item["y_min"] + y0),
                            "y_max": int(roi_item["y_min"] + y0 + kernel_mask.shape[0]),
                            "x_min": int(roi_item["x_min"] + x0),
                            "x_max": int(roi_item["x_min"] + x0 + kernel_mask.shape[1]),
                            "selected": False,
                        }
                        kernel_rows.append(row)
                        roi_spectra.append(corrected_spectrum)
                        roi_ratios.append(ratio)
                        roi_intensities.append(intensity)
                if roi_ratios:
                    summary_rows.append(
                        {
                            "level": "ROI",
                            "image": spectral_layer_name,
                            "roi": roi_item["roi_label"],
                            "kernel": f"{kernel_size}x{kernel_size}",
                            "n_kernels": len(roi_ratios),
                            "mean_ratio": float(np.mean(roi_ratios)),
                            "median_ratio": float(np.median(roi_ratios)),
                            "sd_ratio": float(np.std(roi_ratios, ddof=1)) if len(roi_ratios) > 1 else 0.0,
                            "mean_intensity": float(np.mean(roi_intensities)),
                        }
                    )
            if not kernel_rows:
                raise ValueError("No valid kernels were found with the current ROI and coverage settings.")

            image_ratios = [row["ratio"] for row in kernel_rows]
            image_intensities = [row["intensity"] for row in kernel_rows]
            summary_rows.append(
                {
                    "level": "Image",
                    "image": spectral_layer_name,
                    "roi": "All selected ROIs",
                    "kernel": f"{kernel_size}x{kernel_size}",
                    "n_kernels": len(kernel_rows),
                    "mean_ratio": float(np.mean(image_ratios)),
                    "median_ratio": float(np.median(image_ratios)),
                    "sd_ratio": float(np.std(image_ratios, ddof=1)) if len(image_ratios) > 1 else 0.0,
                    "mean_intensity": float(np.mean(image_intensities)),
                }
            )

            self._kernel_rows = kernel_rows
            self._summary_rows = summary_rows
            self._populate_summary_table()
            self._draw_scatter()
            self.stats_report.setPlainText(
                f"Computed {len(kernel_rows)} kernel point(s) across {len(roi_items)} ROI region(s) for '{spectral_layer_name}'. "
                f"Image mean ratio = {np.mean(image_ratios):.4f}."
            )
        except Exception as exc:
            self.stats_report.setPlainText(str(exc))

    def _draw_scatter(self):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        x = np.asarray([row["ratio"] for row in self._kernel_rows], dtype=np.float32)
        y = np.asarray([row["intensity"] for row in self._kernel_rows], dtype=np.float32)
        colors = ["#d62728" if row.get("selected") else "#1f77b4" for row in self._kernel_rows]
        axis.scatter(x, y, s=18, alpha=0.75, c=colors)
        axis.set_xlabel("Emission ratio")
        axis.set_ylabel("Mean spectral intensity")
        axis.set_title("Spatial ratio scatter")
        axis.grid(True, alpha=0.3)
        self.figure.subplots_adjust(left=0.1, right=0.96, top=0.9, bottom=0.16)
        self.canvas.draw()
        self._selector = RectangleSelector(axis, self._on_scatter_selected, useblit=False, button=[1], interactive=False)

    def _on_scatter_selected(self, eclick, erelease):
        if not self._kernel_rows:
            return
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        selected_count = 0
        for row in self._kernel_rows:
            selected = x0 <= row["ratio"] <= x1 and y0 <= row["intensity"] <= y1
            row["selected"] = bool(selected)
            if selected:
                selected_count += 1
        self._draw_scatter()
        self._update_selection_overlay()
        self.stats_report.setPlainText(f"Selected {selected_count} kernel point(s) from the scatter plot.")

    def _update_selection_overlay(self):
        if not self._kernel_rows:
            return
        image_name = self._kernel_rows[0]["image"]
        spectral_layer = self._find_layer_by_name(image_name)
        if spectral_layer is None:
            return
        metadata = getattr(spectral_layer, "metadata", {})
        cube = np.asarray(metadata.get("spectral_cube"), dtype=np.float32)
        overlay = np.zeros(cube.shape[1:], dtype=np.uint16)
        label_value = 1
        for row in self._kernel_rows:
            if not row.get("selected"):
                continue
            overlay[row["y_min"]:row["y_max"], row["x_min"]:row["x_max"]] = label_value
            label_value += 1
        overlay_layer = self._find_layer_by_name(self._selection_overlay_name)
        if overlay_layer is None:
            self.viewer.add_labels(
                overlay,
                name=self._selection_overlay_name,
                opacity=0.45,
                metadata={"source_spectral_layer_name": image_name},
            )
        else:
            overlay_layer.data = overlay
            overlay_layer.metadata = {
                **getattr(overlay_layer, "metadata", {}),
                "source_spectral_layer_name": image_name,
            }
            overlay_layer.refresh()

    def _clear_scatter_selection(self):
        for row in self._kernel_rows:
            row["selected"] = False
        overlay_layer = self._find_layer_by_name(self._selection_overlay_name)
        if overlay_layer is not None and self._kernel_rows:
            image_name = self._kernel_rows[0]["image"]
            spectral_layer = self._find_layer_by_name(image_name)
            if spectral_layer is not None:
                cube = np.asarray(getattr(spectral_layer, "metadata", {}).get("spectral_cube"), dtype=np.float32)
                overlay_layer.data = np.zeros(cube.shape[1:], dtype=np.uint16)
                overlay_layer.refresh()
        if self._kernel_rows:
            self._draw_scatter()

    def _populate_summary_table(self):
        self.summary_table.clearContents()
        self.summary_table.setRowCount(len(self._summary_rows))
        self.summary_table.setColumnCount(len(self.SUMMARY_COLUMNS))
        self.summary_table.setHorizontalHeaderLabels(self.SUMMARY_COLUMNS)
        for row_index, row in enumerate(self._summary_rows):
            for column_index, header in enumerate(self.SUMMARY_COLUMNS):
                value = row.get(header, "")
                if header == "n_kernels":
                    display_value = str(int(value))
                elif header in {"mean_ratio", "median_ratio", "sd_ratio"}:
                    display_value = f"{float(value):.4f}"
                elif header == "mean_intensity":
                    display_value = f"{float(value):.3f}"
                else:
                    display_value = str(value)
                item = QTableWidgetItem(display_value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.summary_table.setItem(row_index, column_index, item)
        self.summary_table.resizeColumnsToContents()

    def _export_raw_kernel_csv(self):
        if not self._kernel_rows:
            self.stats_report.setPlainText("No kernel data available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Raw Kernel CSV", "spatial_ratio_kernels.csv", "CSV files (*.csv)")
        if not path:
            return
        headers = ["kernel_id", "image", "roi", "kernel_size", "center_y", "center_x", "ratio", "intensity", "selected"]
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in self._kernel_rows:
                writer.writerow([row[header] for header in headers])
        self.stats_report.setPlainText(f"Exported {len(self._kernel_rows)} kernel row(s) to {Path(path).name}.")

    def _export_summary_csv(self):
        if not self._summary_rows:
            self.stats_report.setPlainText("No summary data available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Spatial Summary CSV", "spatial_ratio_summary.csv", "CSV files (*.csv)")
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.SUMMARY_COLUMNS)
            for row in self._summary_rows:
                writer.writerow([row.get(header, "") for header in self.SUMMARY_COLUMNS])
        self.stats_report.setPlainText(f"Exported {len(self._summary_rows)} summary row(s) to {Path(path).name}.")

    def _send_summary_to_analysis(self):
        if not self._summary_rows:
            self.stats_report.setPlainText("No spatial summary rows are available to send.")
            return
        pushed = 0
        for row in self._summary_rows:
            trace_label = str(row.get("roi", "")).strip()
            analysis_level = str(row.get("level", "Image")).strip().lower()
            source_image = str(row.get("image", "")).strip()
            dataset_name = f"{source_image} Spatial {trace_label or analysis_level}"
            ROI_SPECTRUM_STORE.add_or_replace_dataset(
                name=dataset_name,
                source_layer_name=source_image,
                mode=self.ratio_mode_combo.currentText(),
                wavelengths_nm=np.asarray([], dtype=np.float32),
                roi_labels=[trace_label] if trace_label else [],
                roi_areas_px=np.asarray([], dtype=np.float32),
                roi_spectra=np.asarray([], dtype=np.float32),
                pooled_spectrum=None,
                trace_kind="spatial_ratio_summary",
                trace_label=trace_label,
                measurement_kind="spatial_ratio",
                analysis_level=analysis_level,
                use_for_analysis=True,
                kernel_size=self._kernel_size_value(),
                split_nm=self._split_value(),
                n_kernels=int(row.get("n_kernels", 0)),
                n_excluded_kernels=0,
                mean_ratio=float(row.get("mean_ratio", 0.0)),
                median_ratio=float(row.get("median_ratio", 0.0)),
                sd_ratio=float(row.get("sd_ratio", 0.0)),
                mean_intensity=float(row.get("mean_intensity", 0.0)),
            )
            pushed += 1
        self.stats_report.setPlainText(f"Sent {pushed} spatial summary row(s) to Spectral Analysis.")
