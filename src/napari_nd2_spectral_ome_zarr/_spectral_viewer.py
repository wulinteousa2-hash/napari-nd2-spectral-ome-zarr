from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

from matplotlib.path import Path as MplPath
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QObject, Qt, QThread, QTimer, Signal
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
    QVBoxLayout,
    QWidget,
)

from PIL import Image

from ._qt_utils import float_parent_dock_later
from ._reader import _read_ome_zarr, build_layer_data
from ._roi_store import ROI_SPECTRUM_STORE
from ._spectral import (
    generate_pseudocolor_image,
    generate_pseudocolor_pair_image,
    load_pseudocolor_config,
    pseudocolor_pair_config,
    pseudocolor_config,
    render_visible_truecolor,
    save_pseudocolor_config,
)


class DerivedRenderWorker(QObject):
    finished = Signal(object, object, str)
    failed = Signal(str)

    def __init__(self, cube: np.ndarray, wavelengths: np.ndarray, use_gpu: bool, worker_count: int):
        super().__init__()
        self.cube = cube
        self.wavelengths = wavelengths
        self.use_gpu = use_gpu
        self.worker_count = worker_count

    def run(self):
        try:
            visible, truecolor = render_visible_truecolor(
                self.cube,
                self.wavelengths,
                use_gpu=self.use_gpu,
                max_workers=self.worker_count,
            )
            self.finished.emit(visible, truecolor, "Derived views ready.")
        except Exception as exc:
            self.failed.emit(str(exc))


class PseudocolorWorker(QObject):
    finished = Signal(object, str)
    failed = Signal(str)

    def __init__(self, cube: np.ndarray, wavelengths: np.ndarray, config: dict, use_gpu: bool):
        super().__init__()
        self.cube = cube
        self.wavelengths = wavelengths
        self.config = config
        self.use_gpu = use_gpu

    def run(self):
        try:
            if self.config.get("mode", "auto_shift") == "roi_pair":
                rgb = generate_pseudocolor_pair_image(
                    self.cube,
                    self.wavelengths,
                    left_reference=np.asarray(self.config["left_reference"], dtype=np.float32),
                    right_reference=np.asarray(self.config["right_reference"], dtype=np.float32),
                    gamma=float(self.config["gamma"]),
                    kernel_size=self.config["kernel_size"],
                    index_method=self.config.get("index_method", "correlation"),
                    use_bg_subtraction=bool(self.config.get("use_bg_subtraction", False)),
                    bg_spectrum=None if self.config.get("bg_spectrum") is None else np.asarray(self.config["bg_spectrum"], dtype=np.float32),
                    bg_similarity_threshold=float(self.config.get("bg_similarity_threshold", 0.9)),
                    auto_contrast=bool(self.config.get("auto_contrast", False)),
                    adaptive_eq=bool(self.config.get("adaptive_eq", False)),
                    adaptive_clip=float(self.config.get("adaptive_clip", 0.03)),
                    reference_wavelengths_nm=np.asarray(self.config.get("wavelengths_nm", self.wavelengths), dtype=np.float32),
                )
            else:
                rgb = generate_pseudocolor_image(
                    self.cube,
                    self.wavelengths,
                    reference_spectrum=np.asarray(self.config["reference_spectrum"], dtype=np.float32),
                    shift=float(self.config["shift"]),
                    gamma=float(self.config["gamma"]),
                    kernel_size=self.config["kernel_size"],
                    index_method=self.config.get("index_method", "correlation"),
                    use_bg_subtraction=bool(self.config.get("use_bg_subtraction", False)),
                    bg_spectrum=None if self.config.get("bg_spectrum") is None else np.asarray(self.config["bg_spectrum"], dtype=np.float32),
                    bg_similarity_threshold=float(self.config.get("bg_similarity_threshold", 0.9)),
                    auto_contrast=bool(self.config.get("auto_contrast", False)),
                    adaptive_eq=bool(self.config.get("adaptive_eq", False)),
                    adaptive_clip=float(self.config.get("adaptive_clip", 0.03)),
                    use_gpu=self.use_gpu,
                    reference_wavelengths_nm=np.asarray(self.config.get("wavelengths_nm", self.wavelengths), dtype=np.float32),
                )
            self.finished.emit(rgb, "Pseudocolor view added.")
        except Exception as exc:
            self.failed.emit(str(exc))


def _batch_pseudocolor_one(zarr_path: str, input_root: str, output_root: str, config: dict) -> str:
    data_tczyx, wavelengths_nm, _metadata, _preview = _read_ome_zarr(zarr_path)
    cube = np.asarray(data_tczyx[0, :, 0, :, :], dtype=np.float32)
    if config.get("mode", "auto_shift") == "roi_pair":
        rgb = generate_pseudocolor_pair_image(
            cube,
            wavelengths_nm,
            left_reference=np.asarray(config["left_reference"], dtype=np.float32),
            right_reference=np.asarray(config["right_reference"], dtype=np.float32),
            gamma=float(config["gamma"]),
            kernel_size=config["kernel_size"],
            index_method=config.get("index_method", "correlation"),
            use_bg_subtraction=bool(config.get("use_bg_subtraction", False)),
            bg_spectrum=None if config.get("bg_spectrum") is None else np.asarray(config["bg_spectrum"], dtype=np.float32),
            bg_similarity_threshold=float(config.get("bg_similarity_threshold", 0.9)),
            auto_contrast=bool(config.get("auto_contrast", False)),
            adaptive_eq=bool(config.get("adaptive_eq", False)),
            adaptive_clip=float(config.get("adaptive_clip", 0.03)),
            reference_wavelengths_nm=np.asarray(config.get("wavelengths_nm", wavelengths_nm), dtype=np.float32),
        )
    else:
        rgb = generate_pseudocolor_image(
            cube,
            wavelengths_nm,
            reference_spectrum=np.asarray(config["reference_spectrum"], dtype=np.float32),
            shift=float(config["shift"]),
            gamma=float(config["gamma"]),
            kernel_size=config["kernel_size"],
            index_method=config.get("index_method", "correlation"),
            use_bg_subtraction=bool(config.get("use_bg_subtraction", False)),
            bg_spectrum=None if config.get("bg_spectrum") is None else np.asarray(config["bg_spectrum"], dtype=np.float32),
            bg_similarity_threshold=float(config.get("bg_similarity_threshold", 0.9)),
            auto_contrast=bool(config.get("auto_contrast", False)),
            adaptive_eq=bool(config.get("adaptive_eq", False)),
            adaptive_clip=float(config.get("adaptive_clip", 0.03)),
            use_gpu=False,
            reference_wavelengths_nm=np.asarray(config.get("wavelengths_nm", wavelengths_nm), dtype=np.float32),
        )
    source = Path(zarr_path)
    relative_parent = source.relative_to(Path(input_root)).parent
    output_path = Path(output_root) / relative_parent / f"{source.stem}_pseudocolor.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path, format="TIFF", compression="tiff_lzw")
    return str(output_path)


class BatchPseudocolorWorker(QObject):
    progress = Signal(str)
    finished = Signal(int, str)
    failed = Signal(str)

    def __init__(self, input_dir: str, output_dir: str, config: dict, max_workers: int):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.max_workers = max_workers

    def run(self):
        try:
            input_root = Path(self.input_dir)
            output_root = Path(self.output_dir)
            zarr_paths = sorted(str(path) for path in input_root.rglob("*.zarr"))
            if not zarr_paths:
                raise ValueError("No .zarr files found in the selected folder.")
            done = 0
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {
                    executor.submit(_batch_pseudocolor_one, path, str(input_root), str(output_root), self.config): path
                    for path in zarr_paths
                }
                for future in as_completed(future_map):
                    done += 1
                    result_path = future.result()
                    self.progress.emit(f"Pseudocolor {done}/{len(zarr_paths)}: {Path(result_path).name}")
            self.finished.emit(done, str(output_root))
        except Exception as exc:
            self.failed.emit(str(exc))


class SpectralViewerWidget(QWidget):
    ROI_LAYER_SUFFIX = " ROI"
    ROI_LABEL_LAYER_SUFFIX = " ROI Text"
    COMPARISON_COLUMNS = ["plot", "image", "trace", "kind", "updated"]

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._render_thread = None
        self._render_worker = None
        self._batch_thread = None
        self._batch_worker = None
        self._pseudocolor_thread = None
        self._pseudocolor_worker = None
        self._loaded_pseudocolor_config = None
        self._pending_render_layer_name = None
        self._pending_render_metadata = None
        self._pending_pseudocolor_layer_name = None
        self._pending_pseudocolor_metadata = None
        self._last_plot_kind = "roi"
        self._bound_roi_layer_ids: set[int] = set()
        self._bound_labels_layer_ids: set[int] = set()
        self._roi_plot_timer = QTimer(self)
        self._roi_plot_timer.setSingleShot(True)
        self._roi_plot_timer.timeout.connect(self._auto_plot_roi_spectrum)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Normalized", "Absolute"])
        self.roi_source_combo = QComboBox()
        self.roi_source_combo.addItems(["Shapes", "Labels"])
        self.pool_checkbox = QCheckBox("Plot pooled ROI mean")
        self.pool_checkbox.setChecked(False)
        self.individual_checkbox = QCheckBox("Plot individual ROIs")
        self.individual_checkbox.setChecked(True)
        self.include_background_checkbox = QCheckBox("Include background label 0")
        self.include_background_checkbox.setChecked(False)
        self.show_legend_checkbox = QCheckBox("Show legend")
        self.show_legend_checkbox.setChecked(True)
        self.legend_outside_checkbox = QCheckBox("Legend outside")
        self.legend_outside_checkbox.setChecked(True)

        self.use_gpu_checkbox = QCheckBox("Use GPU for truecolor")
        self.use_gpu_checkbox.setChecked(True)
        self.worker_combo = QComboBox()
        self.worker_combo.addItems(["1", "2", "4", "8"])
        self.worker_combo.setCurrentText("2")

        self.plot_button = QPushButton("Refresh Active ROI Spectrum")
        self.plot_button.clicked.connect(self._plot_roi_spectrum)
        self.prepare_all_roi_button = QPushButton("Prepare ROI Layers")
        self.prepare_all_roi_button.clicked.connect(self._prepare_all_roi_layers)
        self.roi_image_combo = QComboBox()
        self.activate_roi_button = QPushButton("Activate ROI Layer")
        self.activate_roi_button.clicked.connect(self._activate_selected_roi_layer)
        self.labels_layer_combo = QComboBox()
        self.bind_labels_button = QPushButton("Bind Labels To Active Image")
        self.bind_labels_button.clicked.connect(self._bind_selected_labels_layer)
        self.clear_roi_button = QPushButton("Clear Active ROI")
        self.clear_roi_button.clicked.connect(self._clear_active_roi_layer)
        self.refresh_all_roi_button = QPushButton("Refresh All ROI Datasets")
        self.refresh_all_roi_button.clicked.connect(self._refresh_all_roi_datasets)

        self.show_split_button = QPushButton("Show Split")
        self.show_split_button.clicked.connect(self._show_split)

        self.show_truecolor_button = QPushButton("Show Truecolor")
        self.show_truecolor_button.clicked.connect(self._show_truecolor)
        self.show_pseudocolor_button = QPushButton("Show Pseudocolor")
        self.show_pseudocolor_button.clicked.connect(self._show_pseudocolor)
        self.save_config_button = QPushButton("Save Pseudocolor Config")
        self.save_config_button.clicked.connect(self._save_pseudocolor_config)
        self.load_config_button = QPushButton("Load Pseudocolor Config")
        self.load_config_button.clicked.connect(self._load_pseudocolor_config)
        self.batch_pseudocolor_button = QPushButton("Batch Pseudocolor")
        self.batch_pseudocolor_button.clicked.connect(self._batch_pseudocolor)
        self.export_dataset_button = QPushButton("Export Selected ROI CSV")
        self.export_dataset_button.clicked.connect(self._export_selected_roi_dataset)
        self.export_all_datasets_button = QPushButton("Export All ROI CSV")
        self.export_all_datasets_button.clicked.connect(self._export_all_roi_datasets)
        self.save_session_button = QPushButton("Save Session Package")
        self.save_session_button.clicked.connect(self._save_session_package)
        self.load_session_button = QPushButton("Load Session Package")
        self.load_session_button.clicked.connect(self._load_session_package)

        self.shift_edit = QLineEdit("2.0")
        self.gamma_edit = QLineEdit("1.2")
        self.pseudocolor_mode_combo = QComboBox()
        self.pseudocolor_mode_combo.addItems(["auto_shift", "roi_pair"])
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["1", "3", "4"])
        self.kernel_combo.setCurrentText("3")
        self.index_method_combo = QComboBox()
        self.index_method_combo.addItems(["correlation", "ratio", "distance"])
        self.bg_subtraction_checkbox = QCheckBox("Background subtraction")
        self.auto_contrast_checkbox = QCheckBox("Auto contrast")
        self.adaptive_eq_checkbox = QCheckBox("Adaptive EQ")

        self.status_label = QLabel("Select a spectral layer and use Shapes or Labels as the ROI source.")
        self.dataset_combo = QComboBox()
        self.dataset_combo.setEnabled(False)
        self.dataset_combo.addItem("No stored ROI datasets")
        self.comparison_table = QTableWidget()
        self.comparison_table.setAlternatingRowColors(False)
        self.comparison_table.setColumnCount(len(self.COMPARISON_COLUMNS))
        self.comparison_table.setHorizontalHeaderLabels(self.COMPARISON_COLUMNS)
        self.comparison_table.setMinimumHeight(180)
        self.refresh_comparison_button = QPushButton("Refresh Comparison Table")
        self.refresh_comparison_button.clicked.connect(self._refresh_comparison_table)
        self.plot_selected_comparison_button = QPushButton("Plot Selected Across Images")
        self.plot_selected_comparison_button.clicked.connect(self._plot_selected_comparison_rows)
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(320)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Plot mode"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(QLabel("ROI source"))
        controls.addWidget(self.roi_source_combo)
        controls.addWidget(self.individual_checkbox)
        controls.addWidget(self.pool_checkbox)
        controls.addWidget(self.include_background_checkbox)
        controls.addWidget(self.show_legend_checkbox)
        controls.addWidget(self.legend_outside_checkbox)
        controls.addWidget(self.use_gpu_checkbox)
        controls.addWidget(QLabel("Workers"))
        controls.addWidget(self.worker_combo)

        pseudocolor_controls = QHBoxLayout()
        pseudocolor_controls.addWidget(QLabel("Pseudo mode"))
        pseudocolor_controls.addWidget(self.pseudocolor_mode_combo)
        pseudocolor_controls.addWidget(QLabel("Shift"))
        pseudocolor_controls.addWidget(self.shift_edit)
        pseudocolor_controls.addWidget(QLabel("Gamma"))
        pseudocolor_controls.addWidget(self.gamma_edit)
        pseudocolor_controls.addWidget(QLabel("Kernel"))
        pseudocolor_controls.addWidget(self.kernel_combo)
        pseudocolor_controls.addWidget(QLabel("Method"))
        pseudocolor_controls.addWidget(self.index_method_combo)
        pseudocolor_controls.addWidget(self.bg_subtraction_checkbox)
        pseudocolor_controls.addWidget(self.auto_contrast_checkbox)
        pseudocolor_controls.addWidget(self.adaptive_eq_checkbox)

        roi_buttons = QHBoxLayout()
        roi_buttons.addWidget(self.prepare_all_roi_button)
        roi_buttons.addWidget(QLabel("ROI image"))
        roi_buttons.addWidget(self.roi_image_combo)
        roi_buttons.addWidget(self.activate_roi_button)
        roi_buttons.addWidget(QLabel("Labels layer"))
        roi_buttons.addWidget(self.labels_layer_combo)
        roi_buttons.addWidget(self.bind_labels_button)
        roi_buttons.addWidget(self.clear_roi_button)
        roi_buttons.addWidget(self.plot_button)
        roi_buttons.addWidget(self.refresh_all_roi_button)
        roi_buttons.addWidget(self.show_split_button)
        roi_buttons.addWidget(self.show_truecolor_button)

        config_buttons = QHBoxLayout()
        config_buttons.addWidget(self.show_pseudocolor_button)
        config_buttons.addWidget(self.save_config_button)
        config_buttons.addWidget(self.load_config_button)
        config_buttons.addWidget(self.batch_pseudocolor_button)

        dataset_buttons = QHBoxLayout()
        dataset_buttons.addWidget(QLabel("Stored ROI datasets"))
        dataset_buttons.addWidget(self.dataset_combo)
        dataset_buttons.addWidget(self.export_dataset_button)
        dataset_buttons.addWidget(self.export_all_datasets_button)
        dataset_buttons.addWidget(self.save_session_button)
        dataset_buttons.addWidget(self.load_session_button)

        comparison_buttons = QHBoxLayout()
        comparison_buttons.addWidget(QLabel("ROI comparison"))
        comparison_buttons.addWidget(self.refresh_comparison_button)
        comparison_buttons.addWidget(self.plot_selected_comparison_button)

        roi_group = QGroupBox("ROI Spectrum")
        roi_group_layout = QVBoxLayout()
        roi_group_layout.addLayout(controls)
        roi_group_layout.addLayout(roi_buttons)
        roi_group_layout.addWidget(self.canvas)
        roi_group.setLayout(roi_group_layout)

        comparison_group = QGroupBox("ROI Comparison")
        comparison_group_layout = QVBoxLayout()
        comparison_group_layout.addLayout(dataset_buttons)
        comparison_group_layout.addLayout(comparison_buttons)
        comparison_group_layout.addWidget(self.comparison_table)
        comparison_group.setLayout(comparison_group_layout)

        pseudocolor_group = QGroupBox("Pseudocolor")
        pseudocolor_group_layout = QVBoxLayout()
        pseudocolor_group_layout.addLayout(pseudocolor_controls)
        pseudocolor_group_layout.addLayout(config_buttons)
        pseudocolor_group.setLayout(pseudocolor_group_layout)

        layout = QVBoxLayout()
        layout.addWidget(roi_group, 1)
        layout.addWidget(comparison_group, 0)
        layout.addWidget(pseudocolor_group, 0)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        if hasattr(self.viewer, "layers") and hasattr(self.viewer.layers, "events"):
            self.viewer.layers.events.inserted.connect(self._on_layers_changed)
            self.viewer.layers.events.removed.connect(self._on_layers_changed)
        selection_events = getattr(getattr(self.viewer.layers, "selection", None), "events", None)
        if selection_events is not None and hasattr(selection_events, "active"):
            selection_events.active.connect(self._on_active_layer_changed)
        self._refresh_dataset_combo()
        self._refresh_labels_layer_combo()
        self._refresh_roi_image_combo()
        self._refresh_comparison_table()
        self._sync_roi_context_visibility()
        self.mode_combo.currentTextChanged.connect(self._refresh_active_plot_from_controls)
        self.roi_source_combo.currentTextChanged.connect(self._refresh_active_plot_from_controls)
        self.individual_checkbox.stateChanged.connect(self._refresh_active_plot_from_controls)
        self.pool_checkbox.stateChanged.connect(self._refresh_active_plot_from_controls)
        self.include_background_checkbox.stateChanged.connect(self._refresh_active_plot_from_controls)
        self.show_legend_checkbox.stateChanged.connect(self._refresh_active_plot_from_controls)
        self.legend_outside_checkbox.stateChanged.connect(self._refresh_active_plot_from_controls)
        float_parent_dock_later(self)

    def _active_spectral_layer(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            raise ValueError("No active layer selected.")
        metadata = getattr(layer, "metadata", {})
        cube = metadata.get("spectral_cube")
        wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
        if cube is None or wavelengths.size == 0:
            source_spectral_layer_name = metadata.get("source_spectral_layer_name")
            if source_spectral_layer_name:
                spectral_layer = self._find_layer_by_name(source_spectral_layer_name)
                if spectral_layer is not None:
                    metadata = getattr(spectral_layer, "metadata", {})
                    cube = metadata.get("spectral_cube")
                    wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
                    if cube is not None and wavelengths.size > 0:
                        return spectral_layer, cube, wavelengths, metadata
            if layer.__class__.__name__ == "Shapes" and layer.name.endswith(self.ROI_LAYER_SUFFIX):
                spectral_layer = self._find_layer_by_name(layer.name.removesuffix(self.ROI_LAYER_SUFFIX))
                if spectral_layer is not None:
                    metadata = getattr(spectral_layer, "metadata", {})
                    cube = metadata.get("spectral_cube")
                    wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
                    if cube is not None and wavelengths.size > 0:
                        return spectral_layer, cube, wavelengths, metadata
            raise ValueError("Active layer does not contain spectral metadata.")
        return layer, cube, wavelengths, metadata

    def _on_layers_changed(self, event=None):
        del event
        self._refresh_labels_layer_combo()
        self._refresh_roi_image_combo()
        self._reorder_roi_context_layers()
        self._sync_roi_context_visibility()

    def _on_active_layer_changed(self, event=None):
        del event
        self._sync_roi_context_visibility()

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

    def _refresh_labels_layer_combo(self):
        current_name = self.labels_layer_combo.currentText() if self.labels_layer_combo.count() else ""
        self.labels_layer_combo.blockSignals(True)
        self.labels_layer_combo.clear()
        labels_layers = [layer for layer in self.viewer.layers if layer.__class__.__name__ == "Labels"]
        if not labels_layers:
            self.labels_layer_combo.addItem("No labels layers", userData=None)
            self.labels_layer_combo.setEnabled(False)
            self.bind_labels_button.setEnabled(False)
            self.labels_layer_combo.blockSignals(False)
            return

        self.labels_layer_combo.setEnabled(True)
        self.bind_labels_button.setEnabled(True)
        selected_index = 0
        for index, layer in enumerate(labels_layers):
            self.labels_layer_combo.addItem(layer.name, userData=layer.name)
            if layer.name == current_name:
                selected_index = index
        self.labels_layer_combo.setCurrentIndex(selected_index)
        self.labels_layer_combo.blockSignals(False)

    def _refresh_roi_image_combo(self):
        current_name = self.roi_image_combo.currentData()
        spectral_layers = self._spectral_layers()
        self.roi_image_combo.blockSignals(True)
        self.roi_image_combo.clear()
        if not spectral_layers:
            self.roi_image_combo.addItem("No open spectral images", userData=None)
            self.roi_image_combo.setEnabled(False)
            self.activate_roi_button.setEnabled(False)
            self.roi_image_combo.blockSignals(False)
            return
        self.roi_image_combo.setEnabled(True)
        self.activate_roi_button.setEnabled(True)
        selected_index = 0
        for index, layer in enumerate(spectral_layers):
            self.roi_image_combo.addItem(layer.name, userData=layer.name)
            if layer.name == current_name:
                selected_index = index
        self.roi_image_combo.setCurrentIndex(selected_index)
        self.roi_image_combo.blockSignals(False)

    def _active_roi_shapes_layer(self, spectral_layer_name: str):
        active_layer = self.viewer.layers.selection.active
        if (
            active_layer is not None
            and active_layer.__class__.__name__ == "Shapes"
            and active_layer.name == self._roi_layer_name(spectral_layer_name)
        ):
            return active_layer
        return self._find_roi_shapes_layer(spectral_layer_name)

    def _roi_layer_name(self, spectral_layer_name: str) -> str:
        return f"{spectral_layer_name}{self.ROI_LAYER_SUFFIX}"

    def _roi_label_layer_name(self, spectral_layer_name: str) -> str:
        return f"{spectral_layer_name}{self.ROI_LABEL_LAYER_SUFFIX}"

    def _find_roi_shapes_layer(self, spectral_layer_name: str):
        target_name = self._roi_layer_name(spectral_layer_name)
        for layer in self.viewer.layers:
            if layer.__class__.__name__ == "Shapes" and layer.name == target_name:
                return layer
        return None

    def _find_roi_label_layer(self, spectral_layer_name: str):
        target_name = self._roi_label_layer_name(spectral_layer_name)
        for layer in self.viewer.layers:
            if layer.__class__.__name__ == "Points" and layer.name == target_name:
                return layer
        return None

    def _remove_legacy_roi_label_layer(self, spectral_layer_name: str):
        layer = self._find_roi_label_layer(spectral_layer_name)
        if layer is not None:
            self.viewer.layers.remove(layer)

    def _find_bound_labels_layer(self, spectral_layer_name: str):
        active_layer = self.viewer.layers.selection.active
        if (
            active_layer is not None
            and active_layer.__class__.__name__ == "Labels"
            and getattr(active_layer, "metadata", {}).get("source_spectral_layer_name") == spectral_layer_name
        ):
            return active_layer
        for layer in self.viewer.layers:
            if layer.__class__.__name__ != "Labels":
                continue
            if getattr(layer, "metadata", {}).get("source_spectral_layer_name") == spectral_layer_name:
                return layer
        return None

    def _move_layer_after(self, layer, anchor_layer):
        if layer is None or anchor_layer is None or layer is anchor_layer:
            return
        layers = self.viewer.layers
        try:
            current_index = list(layers).index(layer)
            anchor_index = list(layers).index(anchor_layer)
        except ValueError:
            return
        target_index = anchor_index + 1
        if current_index < target_index:
            target_index -= 1
        if current_index == target_index:
            return
        layers.move(current_index, target_index)

    def _reorder_roi_context_layers(self):
        previous_anchor = None
        for spectral_layer in self._spectral_layers():
            if previous_anchor is not None:
                self._move_layer_after(spectral_layer, previous_anchor)
            roi_layer = self._find_roi_shapes_layer(spectral_layer.name)
            if roi_layer is not None:
                self._move_layer_after(roi_layer, spectral_layer)
                previous_anchor = roi_layer
            else:
                previous_anchor = spectral_layer
            label_layer = self._find_roi_label_layer(spectral_layer.name)
            if label_layer is not None:
                self._move_layer_after(label_layer, previous_anchor)
                previous_anchor = label_layer

    def _source_layer_name_for_layer(self, layer) -> str | None:
        if layer is None:
            return None
        metadata = getattr(layer, "metadata", {})
        source_layer_name = metadata.get("source_spectral_layer_name")
        if source_layer_name:
            return str(source_layer_name)
        cube = metadata.get("spectral_cube")
        wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
        if cube is not None and wavelengths.size > 0:
            return str(layer.name)
        if layer.__class__.__name__ == "Shapes" and layer.name.endswith(self.ROI_LAYER_SUFFIX):
            return layer.name.removesuffix(self.ROI_LAYER_SUFFIX)
        if layer.__class__.__name__ == "Points" and layer.name.endswith(self.ROI_LABEL_LAYER_SUFFIX):
            return layer.name.removesuffix(self.ROI_LABEL_LAYER_SUFFIX)
        return None

    def _sync_roi_context_visibility(self):
        active_layer = getattr(self.viewer.layers.selection, "active", None)
        active_source_layer_name = self._source_layer_name_for_layer(active_layer)
        for layer in self.viewer.layers:
            if layer.__class__.__name__ not in {"Shapes", "Points", "Labels"}:
                continue
            source_layer_name = self._source_layer_name_for_layer(layer)
            if not source_layer_name:
                continue
            if layer.__class__.__name__ == "Points" and layer.name.endswith(self.ROI_LABEL_LAYER_SUFFIX):
                layer.visible = False
                continue
            layer.visible = bool(active_source_layer_name) and source_layer_name == active_source_layer_name
        if active_source_layer_name:
            combo_index = self.roi_image_combo.findData(active_source_layer_name)
            if combo_index >= 0:
                self.roi_image_combo.blockSignals(True)
                self.roi_image_combo.setCurrentIndex(combo_index)
                self.roi_image_combo.blockSignals(False)

    def _selected_labels_layer(self):
        layer_name = self.labels_layer_combo.currentData()
        if not layer_name:
            raise ValueError("Choose a labels layer to bind first.")
        layer = self._find_layer_by_name(layer_name)
        if layer is None or layer.__class__.__name__ != "Labels":
            raise ValueError("Selected labels layer is no longer available.")
        return layer

    def _bind_labels_layer_events(self, labels_layer):
        layer_id = id(labels_layer)
        if layer_id in self._bound_labels_layer_ids:
            return
        labels_layer.events.data.connect(self._schedule_auto_plot_roi_spectrum)
        self._bound_labels_layer_ids.add(layer_id)

    def _ensure_roi_shapes_layer(self, spectral_layer_name: str):
        self._remove_legacy_roi_label_layer(spectral_layer_name)
        layer = self._find_roi_shapes_layer(spectral_layer_name)
        if layer is not None:
            self._bind_roi_layer_events(layer)
            self._reorder_roi_context_layers()
            self.viewer.layers.selection.active = layer
            self._sync_roi_context_visibility()
            return layer
        layer = self.viewer.add_shapes(
            name=self._roi_layer_name(spectral_layer_name),
            edge_color="yellow",
            face_color="transparent",
            edge_width=2,
            metadata={"source_spectral_layer_name": spectral_layer_name},
        )
        self._bind_roi_layer_events(layer)
        self._reorder_roi_context_layers()
        self.viewer.layers.selection.active = layer
        self._sync_roi_context_visibility()
        return layer

    def _bind_roi_layer_events(self, roi_layer):
        layer_id = id(roi_layer)
        if layer_id in self._bound_roi_layer_ids:
            return
        roi_layer.events.data.connect(self._schedule_auto_plot_roi_spectrum)
        self._bound_roi_layer_ids.add(layer_id)

    def _schedule_auto_plot_roi_spectrum(self, event=None):
        del event
        self._roi_plot_timer.start(250)

    def _auto_plot_roi_spectrum(self):
        self._plot_roi_spectrum(require_rois=True)

    def _refresh_active_plot_from_controls(self, _value=None):
        if self._last_plot_kind == "comparison":
            self._plot_selected_comparison_rows()
            return
        self._plot_roi_spectrum(require_rois=True)

    def _ensure_roi_label_layer(self, spectral_layer_name: str):
        return self._find_roi_label_layer(spectral_layer_name)

    def _prepare_roi_layer(self):
        try:
            spectral_layer, _cube, _wavelengths, _metadata = self._active_spectral_layer()
            if self.roi_source_combo.currentText() == "Labels":
                labels_layer = self._find_bound_labels_layer(spectral_layer.name)
                if labels_layer is None:
                    self.status_label.setText("Bind a labels layer to the active image first.")
                    return
                self._bind_labels_layer_events(labels_layer)
                self.viewer.layers.selection.active = labels_layer
                self._sync_roi_context_visibility()
                self.status_label.setText(
                    f"Using labels layer '{labels_layer.name}' for '{spectral_layer.name}'. Nonzero label values become ROI populations."
                )
                return
            roi_layer = self._ensure_roi_shapes_layer(spectral_layer.name)
            self.status_label.setText(
                f"Using ROI layer '{roi_layer.name}'. Draw ROIs there for the active image only."
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _prepare_all_roi_layers(self):
        try:
            spectral_layers = self._spectral_layers()
            if not spectral_layers:
                self.status_label.setText("No open spectral images found.")
                return
            prepared_count = 0
            for spectral_layer in spectral_layers:
                if self.roi_source_combo.currentText() == "Labels":
                    labels_layer = self._find_bound_labels_layer(spectral_layer.name)
                    if labels_layer is None:
                        continue
                    self._bind_labels_layer_events(labels_layer)
                    prepared_count += 1
                    continue
                self._ensure_roi_shapes_layer(spectral_layer.name)
                prepared_count += 1
            if prepared_count == 0 and self.roi_source_combo.currentText() == "Labels":
                self.status_label.setText("No labels layers are bound to the open spectral images yet.")
                return
            mode_label = "labels bindings" if self.roi_source_combo.currentText() == "Labels" else "ROI layers"
            self.status_label.setText(f"Prepared {prepared_count} {mode_label} across the open spectral images.")
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _activate_selected_roi_layer(self):
        try:
            spectral_layer_name = self.roi_image_combo.currentData()
            if not spectral_layer_name:
                self.status_label.setText("Choose an open spectral image first.")
                return
            spectral_layer = self._find_layer_by_name(spectral_layer_name)
            if spectral_layer is None:
                self.status_label.setText("Selected spectral image is no longer open.")
                self._refresh_roi_image_combo()
                return
            if self.roi_source_combo.currentText() == "Labels":
                labels_layer = self._find_bound_labels_layer(spectral_layer_name)
                if labels_layer is None:
                    self.viewer.layers.selection.active = spectral_layer
                    self._sync_roi_context_visibility()
                    self.status_label.setText(
                        f"No bound labels layer exists for '{spectral_layer_name}'. Select the image and bind a labels layer first."
                    )
                    return
                self.viewer.layers.selection.active = labels_layer
                self._sync_roi_context_visibility()
                self.status_label.setText(f"Activated bound labels layer '{labels_layer.name}'.")
                return
            roi_layer = self._ensure_roi_shapes_layer(spectral_layer_name)
            self.status_label.setText(f"Activated ROI layer '{roi_layer.name}'.")
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _bind_selected_labels_layer(self):
        try:
            spectral_layer, cube, _wavelengths, _metadata = self._active_spectral_layer()
            labels_layer = self._selected_labels_layer()
            labels_data = self._labels_array_for_cube(labels_layer, cube)
            labels_layer.metadata = {
                **getattr(labels_layer, "metadata", {}),
                "source_spectral_layer_name": spectral_layer.name,
                "roi_source_kind": "labels",
                "labels_shape": tuple(int(value) for value in labels_data.shape),
            }
            self._bind_labels_layer_events(labels_layer)
            self.viewer.layers.selection.active = labels_layer
            self._sync_roi_context_visibility()
            self.status_label.setText(
                f"Bound labels layer '{labels_layer.name}' to '{spectral_layer.name}'. Label values define ROI populations."
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _clear_active_roi_layer(self):
        try:
            spectral_layer, _cube, _wavelengths, _metadata = self._active_spectral_layer()
            if self.roi_source_combo.currentText() == "Labels":
                labels_layer = self._find_bound_labels_layer(spectral_layer.name)
                if labels_layer is None:
                    self.status_label.setText("No bound labels layer exists for the active image.")
                    return
                labels_layer.metadata = {
                    key: value
                    for key, value in getattr(labels_layer, "metadata", {}).items()
                    if key != "source_spectral_layer_name"
                }
                label_layer = self._find_roi_label_layer(spectral_layer.name)
                if label_layer is not None:
                    label_layer.data = np.empty((0, 2), dtype=np.float32)
                    label_layer.properties = {"label": np.asarray([], dtype=object)}
                    label_layer.refresh()
                self.status_label.setText(
                    f"Removed labels binding for '{labels_layer.name}' from '{spectral_layer.name}'."
                )
                return
            roi_layer = self._find_roi_shapes_layer(spectral_layer.name)
            if roi_layer is None:
                self.status_label.setText("No ROI layer exists yet for the active image.")
                return
            roi_layer.data = []
            roi_layer.refresh()
            label_layer = self._find_roi_label_layer(spectral_layer.name)
            if label_layer is not None:
                label_layer.data = np.empty((0, 2), dtype=np.float32)
                label_layer.properties = {"label": np.asarray([], dtype=object)}
                label_layer.refresh()
            self.viewer.layers.selection.active = roi_layer
            self.status_label.setText(
                f"Cleared ROI shapes for '{spectral_layer.name}'. ROI numbering will restart from ROI 1."
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _collect_roi_items(self, cube: np.ndarray) -> list[dict]:
        spectral_layer, _cube, _wavelengths, _metadata = self._active_spectral_layer()
        if self.roi_source_combo.currentText() == "Labels":
            labels_layer = self._find_bound_labels_layer(spectral_layer.name)
            if labels_layer is None:
                return []
            roi_items = self._collect_label_roi_items(cube, labels_layer)
            self._update_roi_layer_metadata(labels_layer, roi_items)
            return roi_items
        roi_layer = self._active_roi_shapes_layer(spectral_layer.name)
        if roi_layer is None:
            return []
        rois = []
        for shape_index, shape in enumerate(roi_layer.data):
            vertices = np.asarray(shape)
            roi_item = self._build_roi_item(cube, roi_layer, shape_index, vertices)
            if roi_item is not None:
                rois.append(
                    roi_item
                )
        rois.sort(key=lambda item: item["x_center"])
        self._update_roi_layer_metadata(roi_layer, rois)
        return rois

    def _build_roi_item(self, cube: np.ndarray, roi_layer, shape_index: int, vertices: np.ndarray):
        if vertices.ndim != 2 or vertices.shape[1] < 2:
            return None
        y_min = max(0, int(np.floor(vertices[:, 0].min())))
        y_max = min(cube.shape[1], int(np.ceil(vertices[:, 0].max())))
        x_min = max(0, int(np.floor(vertices[:, 1].min())))
        x_max = min(cube.shape[2], int(np.ceil(vertices[:, 1].max())))
        if y_max <= y_min or x_max <= x_min:
            return None

        roi_cube = np.asarray(cube[:, y_min:y_max, x_min:x_max], dtype=np.float32)
        local_vertices = np.column_stack((vertices[:, 0] - y_min, vertices[:, 1] - x_min))
        mask = self._polygon_mask(local_vertices, roi_cube.shape[1], roi_cube.shape[2])
        area_px = int(mask.sum())
        if area_px <= 0:
            return None

        shape_type = self._shape_type_for_index(roi_layer, shape_index)
        centroid = vertices.mean(axis=0)
        return {
            "x_center": float(centroid[1]),
            "center": centroid.astype(np.float32),
            "roi_cube": roi_cube,
            "roi_mask": mask,
            "layer": roi_layer,
            "roi_label": "",
            "mask_value": None,
            "shape_index": shape_index,
            "shape_type": shape_type,
            "area_px": area_px,
        }

    def _collect_label_roi_items(self, cube: np.ndarray, labels_layer) -> list[dict]:
        labels_data = self._labels_array_for_cube(labels_layer, cube)
        label_values = np.unique(labels_data.astype(np.int64, copy=False))
        roi_items = []
        for label_value in sorted(int(value) for value in label_values):
            if label_value == 0 and not self.include_background_checkbox.isChecked():
                continue
            mask = labels_data == label_value
            area_px = int(mask.sum())
            if area_px <= 0:
                continue
            coords = np.argwhere(mask)
            center = coords.mean(axis=0).astype(np.float32)
            roi_items.append(
                {
                    "x_center": float(center[1]),
                    "center": center,
                    "roi_cube": np.asarray(cube, dtype=np.float32),
                    "roi_mask": mask,
                    "layer": labels_layer,
                    "roi_label": f"Label {label_value}",
                    "mask_value": int(label_value),
                    "shape_index": None,
                    "shape_type": "labels",
                    "area_px": area_px,
                }
            )
        return roi_items

    def _labels_array_for_cube(self, labels_layer, cube: np.ndarray) -> np.ndarray:
        labels_data = np.asarray(getattr(labels_layer, "data", []))
        labels_data = np.squeeze(labels_data)
        expected_shape = tuple(int(value) for value in cube.shape[1:])
        if labels_data.ndim != 2 or tuple(int(value) for value in labels_data.shape) != expected_shape:
            raise ValueError(
                f"Labels layer '{labels_layer.name}' shape {tuple(labels_data.shape)} does not match spectral image shape {expected_shape}."
            )
        return labels_data

    def _shape_type_for_index(self, roi_layer, shape_index: int) -> str:
        shape_types = getattr(roi_layer, "shape_type", None)
        if shape_types is None:
            return "polygon"
        try:
            return str(shape_types[shape_index])
        except Exception:
            return "polygon"

    def _polygon_mask(self, vertices: np.ndarray, height: int, width: int) -> np.ndarray:
        if height <= 0 or width <= 0:
            return np.zeros((0, 0), dtype=bool)
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        points = np.column_stack(((grid_y + 0.5).ravel(), (grid_x + 0.5).ravel()))
        path = MplPath(vertices[:, :2], closed=True)
        return path.contains_points(points, radius=1e-9).reshape(height, width)

    def _update_roi_layer_metadata(self, roi_layer, roi_items: list[dict]):
        area_map = {}
        for index, item in enumerate(roi_items, start=1):
            roi_label = item.get("roi_label") or f"ROI {index}"
            area_map[roi_label] = {
                "shape_index": item.get("shape_index"),
                "shape_type": item["shape_type"],
                "mask_value": item.get("mask_value"),
                "area_px": float(item["area_px"]),
            }
        roi_layer.metadata = {
            **getattr(roi_layer, "metadata", {}),
            "roi_metrics": area_map,
        }

    def _roi_mean_spectrum(self, roi_item: dict) -> np.ndarray:
        roi_pixels = roi_item["roi_cube"][:, roi_item["roi_mask"]]
        if roi_pixels.size == 0:
            raise ValueError("Selected ROI does not contain any pixels.")
        return roi_pixels.mean(axis=1)

    def _display_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        data = np.asarray(spectrum, dtype=np.float32)
        if self.mode_combo.currentText() != "Normalized":
            return data
        shifted = data - float(np.min(data))
        return shifted / (float(np.max(shifted)) + 1e-8)

    def _apply_roi_shape_labels(self, roi_items: list[dict]):
        labels_by_layer = {}
        for index, item in enumerate(roi_items, start=1):
            layer = item["layer"]
            if layer.__class__.__name__ != "Shapes":
                continue
            entries = labels_by_layer.setdefault(layer, [])
            roi_label = item.get("roi_label") or f"ROI {index}"
            entries.append(f"{roi_label}\n{item['area_px']} px")

        for layer, labels in labels_by_layer.items():
            source_spectral_layer_name = getattr(layer, "metadata", {}).get("source_spectral_layer_name")
            if source_spectral_layer_name:
                self._remove_legacy_roi_label_layer(str(source_spectral_layer_name))
            layer.properties = {
                **getattr(layer, "properties", {}),
                "roi_label_text": np.asarray(labels, dtype=object),
            }
            layer.text = {
                "string": "{roi_label_text}",
                "color": "yellow",
                "size": 12,
                "anchor": "upper_left",
            }
            layer.refresh()

    def _store_roi_dataset(
        self,
        *,
        layer_name: str,
        wavelengths: np.ndarray,
        roi_labels: list[str],
        roi_areas_px: list[float],
        roi_spectra: list[np.ndarray],
        pooled_spectrum: np.ndarray | None,
    ):
        dataset = ROI_SPECTRUM_STORE.add_or_replace_dataset(
            source_layer_name=layer_name,
            mode=self.mode_combo.currentText(),
            wavelengths_nm=wavelengths,
            roi_labels=roi_labels,
            roi_areas_px=np.asarray(roi_areas_px, dtype=np.float32),
            roi_spectra=np.stack(roi_spectra, axis=0),
            pooled_spectrum=pooled_spectrum,
        )
        self._refresh_dataset_combo(select_dataset_id=dataset.dataset_id)
        self._refresh_comparison_table()
        return dataset

    def _store_layer_roi_dataset(self, spectral_layer, cube: np.ndarray, wavelengths: np.ndarray):
        roi_items = self._collect_roi_items_for_layer(spectral_layer.name, cube)
        if not roi_items:
            return None
        roi_spectra = []
        roi_labels = []
        roi_areas_px = []
        for index, roi_item in enumerate(roi_items, start=1):
            spectrum = self._roi_mean_spectrum(roi_item)
            roi_spectra.append(spectrum)
            roi_labels.append(roi_item.get("roi_label") or f"ROI {index}")
            roi_areas_px.append(float(roi_item["area_px"]))
        self._apply_roi_shape_labels(roi_items)
        pooled = np.mean(np.stack(roi_spectra, axis=0), axis=0) if roi_spectra else None
        return self._store_roi_dataset(
            layer_name=spectral_layer.name,
            wavelengths=wavelengths,
            roi_labels=roi_labels,
            roi_areas_px=roi_areas_px,
            roi_spectra=roi_spectra,
            pooled_spectrum=pooled,
        )

    def _refresh_all_roi_datasets(self):
        try:
            stored_count = 0
            for spectral_layer in self._spectral_layers():
                metadata = getattr(spectral_layer, "metadata", {})
                cube = metadata.get("spectral_cube")
                wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
                if cube is None or wavelengths.size == 0:
                    continue
                dataset = self._store_layer_roi_dataset(
                    spectral_layer=spectral_layer,
                    cube=np.asarray(cube, dtype=np.float32),
                    wavelengths=wavelengths,
                )
                if dataset is not None:
                    stored_count += 1
            if stored_count == 0:
                self.status_label.setText("No ROI datasets found on the open spectral images.")
                return
            self.status_label.setText(
                f"Updated {stored_count} stored ROI dataset(s). Open Spectral Analysis to compare them across images."
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _refresh_dataset_combo(self, select_dataset_id: str | None = None):
        datasets = ROI_SPECTRUM_STORE.list_datasets()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        if not datasets:
            self.dataset_combo.addItem("No stored ROI datasets")
            self.dataset_combo.setEnabled(False)
            self.export_dataset_button.setEnabled(False)
            self.export_all_datasets_button.setEnabled(False)
            self.dataset_combo.blockSignals(False)
            self._refresh_comparison_table()
            return

        self.dataset_combo.setEnabled(True)
        self.export_dataset_button.setEnabled(True)
        self.export_all_datasets_button.setEnabled(True)
        selected_index = 0
        for index, dataset in enumerate(datasets):
            label = f"{dataset.name} [{dataset.created_at}]"
            self.dataset_combo.addItem(label, userData=index)
            if select_dataset_id is not None and dataset.dataset_id == select_dataset_id:
                selected_index = index
        self.dataset_combo.setCurrentIndex(selected_index)
        self.dataset_combo.blockSignals(False)
        self._refresh_comparison_table()

    def _refresh_comparison_table(self):
        datasets = ROI_SPECTRUM_STORE.list_datasets()
        rows: list[tuple[int, int | None, str, str, str]] = []
        for dataset_index, dataset in enumerate(datasets):
            for roi_index, roi_label in enumerate(dataset.roi_labels):
                rows.append((dataset_index, roi_index, dataset.source_layer_name, roi_label, "ROI"))
            if dataset.pooled_spectrum is not None:
                rows.append((dataset_index, None, dataset.source_layer_name, "Pooled", "Pooled"))

        self.comparison_table.clearContents()
        self.comparison_table.setRowCount(len(rows))
        self.comparison_table.setColumnCount(len(self.COMPARISON_COLUMNS))
        self.comparison_table.setHorizontalHeaderLabels(self.COMPARISON_COLUMNS)
        if not rows:
            self.plot_selected_comparison_button.setEnabled(False)
            self.comparison_table.resizeColumnsToContents()
            return

        self.plot_selected_comparison_button.setEnabled(True)
        for row_index, (dataset_index, trace_index, source_layer_name, trace_label, kind) in enumerate(rows):
            for column_index, column_name in enumerate(self.COMPARISON_COLUMNS):
                if column_name == "plot":
                    item = QTableWidgetItem()
                    item.setCheckState(Qt.Checked)
                    item.setFlags((item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled) & ~Qt.ItemIsEditable)
                elif column_name == "image":
                    item = QTableWidgetItem(source_layer_name)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                elif column_name == "trace":
                    item = QTableWidgetItem(trace_label)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                elif column_name == "kind":
                    item = QTableWidgetItem(kind)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                else:
                    item = QTableWidgetItem(datasets[dataset_index].created_at)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setData(Qt.UserRole, (dataset_index, trace_index))
                self.comparison_table.setItem(row_index, column_index, item)
        self.comparison_table.resizeColumnsToContents()

    def _plot_selected_comparison_rows(self):
        datasets = ROI_SPECTRUM_STORE.list_datasets()
        if not datasets:
            self.status_label.setText("No stored ROI datasets available for comparison.")
            return

        selected_rows: list[tuple[int, int | None]] = []
        for row_index in range(self.comparison_table.rowCount()):
            item = self.comparison_table.item(row_index, 0)
            if item is None or item.checkState() != Qt.Checked:
                continue
            dataset_index, trace_index = item.data(Qt.UserRole)
            selected_rows.append((dataset_index, trace_index))
        if not selected_rows:
            self.status_label.setText("Select at least one ROI or pooled trace in the comparison table.")
            return

        self.figure.clear()
        axis = self.figure.add_subplot(111)
        for dataset_index, trace_index in selected_rows:
            dataset = datasets[dataset_index]
            if trace_index is None:
                spectrum = dataset.pooled_spectrum
                if spectrum is None:
                    continue
                spectrum = self._display_spectrum(spectrum)
                label = f"{dataset.source_layer_name} | Pooled"
                axis.plot(dataset.wavelengths_nm, spectrum, linewidth=2.5, linestyle="--", label=label)
            else:
                spectrum = self._display_spectrum(dataset.roi_spectra[trace_index])
                roi_label = dataset.roi_labels[trace_index]
                label = f"{dataset.source_layer_name} | {roi_label}"
                axis.plot(dataset.wavelengths_nm, spectrum, linewidth=1.8, label=label)

        self._finalize_spectrum_axis(
            axis,
            title="ROI comparison across images",
            ylabel="Normalized intensity" if self.mode_combo.currentText() == "Normalized" else "Intensity",
        )
        self.canvas.draw()
        self._last_plot_kind = "comparison"
        self.status_label.setText(f"Plotted {len(selected_rows)} selected trace(s) across stored ROI datasets.")

    def _clear_plot(self):
        self.figure.clear()
        self.canvas.draw()

    def _finalize_spectrum_axis(self, axis, *, title: str, ylabel: str):
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        if self.show_legend_checkbox.isChecked():
            if self.legend_outside_checkbox.isChecked():
                legend = axis.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    fontsize=8,
                )
                if legend is not None:
                    legend.set_draggable(True)
                self.figure.subplots_adjust(left=0.08, right=0.74, top=0.9, bottom=0.18)
            else:
                legend = axis.legend(fontsize=8)
                if legend is not None:
                    legend.set_draggable(True)
                self.figure.subplots_adjust(left=0.08, right=0.96, top=0.9, bottom=0.18)
        else:
            self.figure.subplots_adjust(left=0.08, right=0.96, top=0.9, bottom=0.18)

    def _plot_roi_spectrum(self, require_rois: bool = False):
        try:
            layer, cube, wavelengths, _metadata = self._active_spectral_layer()
            roi_items = self._collect_roi_items(cube)
            if require_rois and not roi_items:
                self._clear_plot()
                if self.roi_source_combo.currentText() == "Labels":
                    self.status_label.setText("ROI spectrum ready. Bind a labels layer with nonzero label values to update spectra automatically.")
                else:
                    self.status_label.setText("ROI spectrum ready. Draw ROI 1, ROI 2, ROI 3... to update spectra automatically.")
                return

            roi_cubes = [item["roi_cube"] for item in roi_items] or [cube]
            self.figure.clear()
            axis = self.figure.add_subplot(111)
            pooled_spectra = []
            roi_labels = []
            roi_areas_px = []
            for index, roi_cube in enumerate(roi_cubes, start=1):
                roi_item = roi_items[index - 1] if roi_items else None
                raw_spectrum = self._roi_mean_spectrum(roi_item) if roi_item is not None else roi_cube.mean(axis=(1, 2))
                display_spectrum = self._display_spectrum(raw_spectrum)
                pooled_spectra.append(raw_spectrum)
                if roi_item is not None:
                    roi_label = roi_item.get("roi_label") or f"ROI {index}"
                    roi_labels.append(roi_label)
                    roi_areas_px.append(float(roi_item["area_px"]))
                else:
                    roi_labels.append(f"ROI {index}")
                    roi_areas_px.append(float(roi_cube.shape[1] * roi_cube.shape[2]))
                if self.individual_checkbox.isChecked():
                    label = f"{roi_labels[-1]} ({int(roi_areas_px[-1])} px)"
                    axis.plot(wavelengths, display_spectrum, linewidth=1.5, label=label)

            if roi_items:
                self._apply_roi_shape_labels(roi_items)

            pooled = None
            if self.pool_checkbox.isChecked() and pooled_spectra:
                pooled = np.mean(np.stack(pooled_spectra, axis=0), axis=0)
                axis.plot(
                    wavelengths,
                    self._display_spectrum(pooled),
                    linewidth=3,
                    linestyle="--",
                    color="black",
                    label="Pooled",
                )

            self._finalize_spectrum_axis(
                axis,
                title="ROI spectrum",
                ylabel="Normalized intensity" if self.mode_combo.currentText() == "Normalized" else "Intensity",
            )
            self.canvas.draw()
            self._last_plot_kind = "roi"
            stored_dataset = self._store_roi_dataset(
                layer_name=layer.name,
                wavelengths=wavelengths,
                roi_labels=roi_labels,
                roi_areas_px=roi_areas_px,
                roi_spectra=pooled_spectra,
                pooled_spectrum=pooled,
            )
            self.status_label.setText(
                f"Spectrum updated for {len(roi_cubes)} ROI(s). Stored {stored_dataset.name} in memory."
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _collect_roi_items_for_layer(self, spectral_layer_name: str, cube: np.ndarray) -> list[dict]:
        if self.roi_source_combo.currentText() == "Labels":
            labels_layer = self._find_bound_labels_layer(spectral_layer_name)
            if labels_layer is None:
                return []
            roi_items = self._collect_label_roi_items(cube, labels_layer)
            self._update_roi_layer_metadata(labels_layer, roi_items)
            return roi_items
        roi_layer = self._find_roi_shapes_layer(spectral_layer_name)
        if roi_layer is None:
            return []
        rois = []
        for shape_index, shape in enumerate(roi_layer.data):
            vertices = np.asarray(shape)
            roi_item = self._build_roi_item(cube, roi_layer, shape_index, vertices)
            if roi_item is not None:
                rois.append(roi_item)
        rois.sort(key=lambda item: item["x_center"])
        self._update_roi_layer_metadata(roi_layer, rois)
        return rois

    def _export_selected_roi_dataset(self):
        if ROI_SPECTRUM_STORE.count() == 0:
            self.status_label.setText("No stored ROI dataset to export.")
            return
        dataset_index = self.dataset_combo.currentData()
        if dataset_index is None:
            self.status_label.setText("Select a stored ROI dataset first.")
            return
        dataset = ROI_SPECTRUM_STORE.get_dataset(dataset_index)
        default_name = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in dataset.name) + ".csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export ROI Dataset CSV", default_name, "CSV files (*.csv)")
        if not path:
            return
        exported_path = ROI_SPECTRUM_STORE.export_dataset_csv(dataset_index, path)
        self.status_label.setText(f"Exported {dataset.name} to {exported_path.name}")

    def _export_all_roi_datasets(self):
        if ROI_SPECTRUM_STORE.count() == 0:
            self.status_label.setText("No stored ROI dataset to export.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select folder for ROI CSV export")
        if not output_dir:
            return
        exported_paths = ROI_SPECTRUM_STORE.export_all_csv(output_dir)
        self.status_label.setText(f"Exported {len(exported_paths)} ROI dataset CSV file(s).")

    @staticmethod
    def _safe_name(text: str) -> str:
        return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in text).strip("_") or "item"

    def _serialize_dataset(self, dataset) -> dict:
        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "source_layer_name": dataset.source_layer_name,
            "mode": dataset.mode,
            "wavelengths_nm": np.asarray(dataset.wavelengths_nm, dtype=np.float32).tolist(),
            "roi_labels": list(dataset.roi_labels),
            "roi_areas_px": np.asarray(dataset.roi_areas_px, dtype=np.float32).tolist(),
            "roi_spectra": np.asarray(dataset.roi_spectra, dtype=np.float32).tolist(),
            "pooled_spectrum": None if dataset.pooled_spectrum is None else np.asarray(dataset.pooled_spectrum, dtype=np.float32).tolist(),
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

    def _deserialize_dataset(self, payload: dict):
        dataset = ROI_SPECTRUM_STORE.add_or_replace_dataset(
            source_layer_name=str(payload["source_layer_name"]),
            mode=str(payload.get("mode", "Absolute")),
            wavelengths_nm=np.asarray(payload["wavelengths_nm"], dtype=np.float32),
            roi_labels=list(payload["roi_labels"]),
            roi_areas_px=np.asarray(payload["roi_areas_px"], dtype=np.float32),
            roi_spectra=np.asarray(payload["roi_spectra"], dtype=np.float32),
            pooled_spectrum=(
                None
                if payload.get("pooled_spectrum") is None
                else np.asarray(payload["pooled_spectrum"], dtype=np.float32)
            ),
        )
        datasets = ROI_SPECTRUM_STORE.list_datasets()
        dataset_index = next(
            (index for index, entry in enumerate(datasets) if entry.dataset_id == dataset.dataset_id),
            None,
        )
        if dataset_index is not None:
            ROI_SPECTRUM_STORE.update_metadata(
                dataset_index,
                animal_id=str(payload.get("animal_id", "")),
                group_label=str(payload.get("group_label", "")),
                genotype=str(payload.get("genotype", "")),
                sex=str(payload.get("sex", "")),
                age=str(payload.get("age", "")),
                region=str(payload.get("region", "")),
                batch=str(payload.get("batch", "")),
                blind_id=str(payload.get("blind_id", "")),
                name=str(payload.get("name", dataset.name)),
                created_at=str(payload.get("created_at", dataset.created_at)),
            )

    def _save_truecolor_tiff(self, output_path: Path, cube: np.ndarray, wavelengths: np.ndarray):
        _visible, rgb = render_visible_truecolor(
            np.asarray(cube, dtype=np.float32),
            np.asarray(wavelengths, dtype=np.float32),
            use_gpu=False,
            max_workers=int(self.worker_combo.currentText()),
        )
        Image.fromarray(rgb, mode="RGB").save(output_path, format="TIFF", compression="tiff_lzw")

    def _add_layer_payloads(self, layer_payloads: list):
        for data, kwargs, layer_type in layer_payloads:
            layer_name = kwargs.get("name")
            if layer_name and self._find_layer_by_name(layer_name) is not None:
                continue
            if layer_type == "image":
                self.viewer.add_image(data, **kwargs)

    def _load_source_layer_from_path(self, source_path: str):
        if not source_path:
            return
        layer_payloads = build_layer_data(
            source_path,
            use_gpu=False,
            include_visible_layer=True,
            include_truecolor_layer=True,
            include_raw_layer=False,
            zarr_use_preview=True,
        )
        self._add_layer_payloads(layer_payloads)

    def _save_session_package(self):
        try:
            package_dir = QFileDialog.getExistingDirectory(self, "Select folder for session package")
            if not package_dir:
                return
            package_root = Path(package_dir)
            package_root.mkdir(parents=True, exist_ok=True)
            roi_shapes_dir = package_root / "roi_shapes"
            roi_datasets_dir = package_root / "roi_datasets"
            truecolor_dir = package_root / "truecolor"
            for folder in (roi_shapes_dir, roi_datasets_dir, truecolor_dir):
                folder.mkdir(parents=True, exist_ok=True)

            spectral_entries = []
            for spectral_layer in self._spectral_layers():
                metadata = getattr(spectral_layer, "metadata", {})
                cube = metadata.get("spectral_cube")
                wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
                source_path = str(metadata.get("source_path", ""))
                safe_name = self._safe_name(spectral_layer.name)

                roi_layer = self._find_roi_shapes_layer(spectral_layer.name)
                roi_shapes_file = None
                if roi_layer is not None and len(getattr(roi_layer, "data", [])) > 0:
                    roi_shapes_payload = {
                        "source_layer_name": spectral_layer.name,
                        "source_path": source_path,
                        "shapes": [np.asarray(shape, dtype=np.float32).tolist() for shape in roi_layer.data],
                        "shape_types": [self._shape_type_for_index(roi_layer, index) for index, _shape in enumerate(roi_layer.data)],
                    }
                    roi_shapes_file = f"{safe_name}_shapes.json"
                    (roi_shapes_dir / roi_shapes_file).write_text(json.dumps(roi_shapes_payload, indent=2), encoding="utf-8")

                truecolor_file = None
                if cube is not None and wavelengths.size > 0:
                    truecolor_file = f"{safe_name}_truecolor.tif"
                    self._save_truecolor_tiff(truecolor_dir / truecolor_file, np.asarray(cube, dtype=np.float32), wavelengths)

                spectral_entries.append(
                    {
                        "source_layer_name": spectral_layer.name,
                        "source_path": source_path,
                        "roi_shapes_file": roi_shapes_file,
                        "truecolor_file": truecolor_file,
                    }
                )

            dataset_files = []
            for dataset in ROI_SPECTRUM_STORE.list_datasets():
                dataset_file = f"{self._safe_name(dataset.dataset_id)}.json"
                (roi_datasets_dir / dataset_file).write_text(
                    json.dumps(self._serialize_dataset(dataset), indent=2),
                    encoding="utf-8",
                )
                dataset_files.append(dataset_file)

            manifest = {
                "version": 1,
                "spectral_entries": spectral_entries,
                "dataset_files": dataset_files,
            }
            (package_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            self.status_label.setText(
                f"Saved session package with {len(spectral_entries)} image entry(s) and {len(dataset_files)} ROI dataset(s)."
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _load_session_package(self):
        try:
            package_dir = QFileDialog.getExistingDirectory(self, "Select session package folder")
            if not package_dir:
                return
            package_root = Path(package_dir)
            manifest_path = package_root / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("Selected folder does not contain manifest.json.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            for entry in manifest.get("spectral_entries", []):
                source_layer_name = str(entry.get("source_layer_name", ""))
                source_path = str(entry.get("source_path", ""))
                if source_layer_name and self._find_layer_by_name(source_layer_name) is None and source_path:
                    self._load_source_layer_from_path(source_path)

                roi_shapes_file = entry.get("roi_shapes_file")
                if roi_shapes_file:
                    payload = json.loads((package_root / "roi_shapes" / roi_shapes_file).read_text(encoding="utf-8"))
                    roi_layer = self._ensure_roi_shapes_layer(str(payload["source_layer_name"]))
                    roi_layer.data = []
                    shapes = [np.asarray(shape, dtype=np.float32) for shape in payload.get("shapes", [])]
                    shape_types = list(payload.get("shape_types", []))
                    for shape, shape_type in zip(shapes, shape_types, strict=False):
                        roi_layer.add(shape, shape_type=shape_type)
                    spectral_layer = self._find_layer_by_name(str(payload["source_layer_name"]))
                    if spectral_layer is not None:
                        metadata = getattr(spectral_layer, "metadata", {})
                        cube = metadata.get("spectral_cube")
                        if cube is not None:
                            roi_items = self._collect_roi_items_for_layer(str(payload["source_layer_name"]), np.asarray(cube, dtype=np.float32))
                            if roi_items:
                                self._apply_roi_shape_labels(roi_items)

            for dataset_file in manifest.get("dataset_files", []):
                payload = json.loads((package_root / "roi_datasets" / dataset_file).read_text(encoding="utf-8"))
                self._deserialize_dataset(payload)

            self._refresh_dataset_combo()
            self._refresh_comparison_table()
            self.status_label.setText("Loaded session package.")
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _show_split(self):
        try:
            layer, cube, _wavelengths, metadata = self._active_spectral_layer()
            self.viewer.add_image(cube, name=f"{layer.name} split", channel_axis=0, metadata=metadata)
            self.status_label.setText("Split spectral view added.")
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _show_truecolor(self):
        try:
            layer, cube, wavelengths, metadata = self._active_spectral_layer()
            if self._render_thread is not None and self._render_thread.isRunning():
                self.status_label.setText("Derived render is already running.")
                return

            self.status_label.setText("Rendering visible and truecolor views in background...")
            self._render_thread = QThread(self)
            self._pending_render_layer_name = layer.name
            self._pending_render_metadata = metadata
            self._render_worker = DerivedRenderWorker(
                cube=np.asarray(cube, dtype=np.float32),
                wavelengths=np.asarray(wavelengths, dtype=np.float32),
                use_gpu=self.use_gpu_checkbox.isChecked(),
                worker_count=int(self.worker_combo.currentText()),
            )
            self._render_worker.moveToThread(self._render_thread)
            self._render_thread.started.connect(self._render_worker.run)
            self._render_worker.finished.connect(self._on_render_finished)
            self._render_worker.failed.connect(self._on_render_failed)
            self._render_worker.finished.connect(self._render_thread.quit)
            self._render_worker.failed.connect(self._render_thread.quit)
            self._render_thread.finished.connect(self._cleanup_render_thread)
            self._render_thread.start()
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _current_pseudocolor_config(self) -> dict:
        if self._loaded_pseudocolor_config is not None:
            config = dict(self._loaded_pseudocolor_config)
            config["mode"] = self.pseudocolor_mode_combo.currentText()
            config["kernel_size"] = int(self.kernel_combo.currentText())
            config["shift"] = float(self.shift_edit.text().strip())
            config["gamma"] = float(self.gamma_edit.text().strip())
            config["index_method"] = self.index_method_combo.currentText()
            config["use_bg_subtraction"] = self.bg_subtraction_checkbox.isChecked()
            config["auto_contrast"] = self.auto_contrast_checkbox.isChecked()
            config["adaptive_eq"] = self.adaptive_eq_checkbox.isChecked()
            return config
        _layer, cube, wavelengths, _metadata = self._active_spectral_layer()
        roi_items = self._collect_roi_items(cube)
        mode = self.pseudocolor_mode_combo.currentText()
        if not roi_items:
            raise ValueError("Define at least one ROI for pseudocolor.")
        bg_spectrum = None
        if self.bg_subtraction_checkbox.isChecked() and len(roi_items) >= 2:
            bg_spectrum = self._roi_mean_spectrum(roi_items[0])
        if mode == "roi_pair":
            if len(roi_items) < 2:
                raise ValueError("ROI pair mode requires at least two ROIs. Leftmost ROI is blue/left, rightmost ROI is red/right.")
            return pseudocolor_pair_config(
                left_reference=self._roi_mean_spectrum(roi_items[0]),
                right_reference=self._roi_mean_spectrum(roi_items[-1]),
                wavelengths_nm=np.asarray(wavelengths, dtype=np.float32),
                gamma=float(self.gamma_edit.text().strip()),
                kernel_size=int(self.kernel_combo.currentText()),
                index_method=self.index_method_combo.currentText(),
                use_bg_subtraction=self.bg_subtraction_checkbox.isChecked(),
                bg_spectrum=bg_spectrum,
                auto_contrast=self.auto_contrast_checkbox.isChecked(),
                adaptive_eq=self.adaptive_eq_checkbox.isChecked(),
            )
        if self.pool_checkbox.isChecked():
            reference_spectrum = np.mean(
                np.stack([self._roi_mean_spectrum(roi_item) for roi_item in roi_items], axis=0),
                axis=0,
            )
        else:
            reference_spectrum = self._roi_mean_spectrum(roi_items[-1])
        return pseudocolor_config(
            reference_spectrum=reference_spectrum,
            wavelengths_nm=np.asarray(wavelengths, dtype=np.float32),
            shift=float(self.shift_edit.text().strip()),
            gamma=float(self.gamma_edit.text().strip()),
            kernel_size=int(self.kernel_combo.currentText()),
            index_method=self.index_method_combo.currentText(),
            use_bg_subtraction=self.bg_subtraction_checkbox.isChecked(),
            bg_spectrum=bg_spectrum,
            auto_contrast=self.auto_contrast_checkbox.isChecked(),
            adaptive_eq=self.adaptive_eq_checkbox.isChecked(),
        )

    def _show_pseudocolor(self):
        try:
            layer, cube, wavelengths, metadata = self._active_spectral_layer()
            config = self._current_pseudocolor_config()
            if self._pseudocolor_thread is not None and self._pseudocolor_thread.isRunning():
                self.status_label.setText("Pseudocolor render is already running.")
                return
            self.status_label.setText("Rendering pseudocolor in background...")
            self._pseudocolor_thread = QThread(self)
            self._pending_pseudocolor_layer_name = layer.name
            self._pending_pseudocolor_metadata = metadata
            self._pseudocolor_worker = PseudocolorWorker(
                cube=np.asarray(cube, dtype=np.float32),
                wavelengths=np.asarray(wavelengths, dtype=np.float32),
                config=config,
                use_gpu=self.use_gpu_checkbox.isChecked(),
            )
            self._pseudocolor_worker.moveToThread(self._pseudocolor_thread)
            self._pseudocolor_thread.started.connect(self._pseudocolor_worker.run)
            self._pseudocolor_worker.finished.connect(self._on_pseudocolor_finished)
            self._pseudocolor_worker.failed.connect(self._on_pseudocolor_failed)
            self._pseudocolor_worker.finished.connect(self._pseudocolor_thread.quit)
            self._pseudocolor_worker.failed.connect(self._pseudocolor_thread.quit)
            self._pseudocolor_thread.finished.connect(self._cleanup_pseudocolor_thread)
            self._pseudocolor_thread.start()
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _save_pseudocolor_config(self):
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Save Pseudocolor Config", "", "JSON files (*.json)")
            if not path:
                return
            save_pseudocolor_config(path, self._current_pseudocolor_config())
            self.status_label.setText(f"Saved pseudocolor config: {Path(path).name}")
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _load_pseudocolor_config(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Load Pseudocolor Config", "", "JSON files (*.json)")
            if not path:
                return
            config = load_pseudocolor_config(path)
            self.pseudocolor_mode_combo.setCurrentText(config.get("mode", "auto_shift"))
            self.shift_edit.setText(str(config.get("shift", 2.0)))
            self.gamma_edit.setText(str(config.get("gamma", 1.2)))
            self.kernel_combo.setCurrentText(str(config["kernel_size"]))
            self.index_method_combo.setCurrentText(config.get("index_method", "correlation"))
            self.bg_subtraction_checkbox.setChecked(bool(config.get("use_bg_subtraction", False)))
            self.auto_contrast_checkbox.setChecked(bool(config.get("auto_contrast", False)))
            self.adaptive_eq_checkbox.setChecked(bool(config.get("adaptive_eq", False)))
            self.status_label.setText(f"Loaded pseudocolor config: {Path(path).name}")
            self._loaded_pseudocolor_config = config
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _batch_pseudocolor(self):
        if self._batch_thread is not None and self._batch_thread.isRunning():
            self.status_label.setText("Batch pseudocolor is already running.")
            return
        try:
            config = self._current_pseudocolor_config()
            input_dir = QFileDialog.getExistingDirectory(self, "Select input folder with OME-Zarr")
            if not input_dir:
                return
            output_dir = QFileDialog.getExistingDirectory(self, "Select output folder for pseudocolor TIFFs")
            if not output_dir:
                return
            self._batch_thread = QThread(self)
            self._batch_worker = BatchPseudocolorWorker(
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
                max_workers=int(self.worker_combo.currentText()),
            )
            self._batch_worker.moveToThread(self._batch_thread)
            self._batch_thread.started.connect(self._batch_worker.run)
            self._batch_worker.progress.connect(self.status_label.setText)
            self._batch_worker.finished.connect(self._on_batch_pseudocolor_finished)
            self._batch_worker.failed.connect(self._on_batch_pseudocolor_failed)
            self._batch_worker.finished.connect(self._batch_thread.quit)
            self._batch_worker.failed.connect(self._batch_thread.quit)
            self._batch_thread.finished.connect(self._cleanup_batch_thread)
            self._batch_thread.start()
            self.status_label.setText("Batch pseudocolor started.")
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _on_render_finished(self, merged: np.ndarray, rgb: np.ndarray, message: str):
        layer_name = self._pending_render_layer_name or "spectral"
        metadata = self._pending_render_metadata or {}
        self.viewer.add_image(merged, name=f"{layer_name} visible sum", metadata=metadata, colormap="gray")
        self.viewer.add_image(rgb, name=f"{layer_name} truecolor", metadata=metadata, rgb=True)
        self.status_label.setText(message)

    def _on_render_failed(self, error_text: str):
        self.status_label.setText(f"Derived render failed: {error_text}")

    def _cleanup_render_thread(self):
        if self._render_worker is not None:
            self._render_worker.deleteLater()
        if self._render_thread is not None:
            self._render_thread.deleteLater()
        self._render_worker = None
        self._render_thread = None
        self._pending_render_layer_name = None
        self._pending_render_metadata = None

    def _on_batch_pseudocolor_finished(self, count: int, output_dir: str):
        self.status_label.setText(f"Batch pseudocolor saved {count} file(s) to {output_dir}")

    def _on_batch_pseudocolor_failed(self, error_text: str):
        self.status_label.setText(f"Batch pseudocolor failed: {error_text}")

    def _cleanup_batch_thread(self):
        if self._batch_worker is not None:
            self._batch_worker.deleteLater()
        if self._batch_thread is not None:
            self._batch_thread.deleteLater()
        self._batch_worker = None
        self._batch_thread = None

    def _on_pseudocolor_finished(self, rgb: np.ndarray, message: str):
        layer_name = self._pending_pseudocolor_layer_name or "spectral"
        metadata = self._pending_pseudocolor_metadata or {}
        self.viewer.add_image(rgb, name=f"{layer_name} pseudocolor", metadata=metadata, rgb=True)
        self.status_label.setText(message)

    def _on_pseudocolor_failed(self, error_text: str):
        self.status_label.setText(f"Pseudocolor failed: {error_text}")

    def _cleanup_pseudocolor_thread(self):
        if self._pseudocolor_worker is not None:
            self._pseudocolor_worker.deleteLater()
        if self._pseudocolor_thread is not None:
            self._pseudocolor_thread.deleteLater()
        self._pseudocolor_worker = None
        self._pseudocolor_thread = None
        self._pending_pseudocolor_layer_name = None
        self._pending_pseudocolor_metadata = None
