from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PIL import Image

from ._qt_utils import float_parent_dock_later
from ._reader import _read_ome_zarr
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
    ROI_LABEL_LAYER_SUFFIX = " ROI Labels"

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

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Normalized", "Absolute"])
        self.pool_checkbox = QCheckBox("Plot pooled ROI mean")
        self.pool_checkbox.setChecked(True)
        self.individual_checkbox = QCheckBox("Plot individual ROIs")
        self.individual_checkbox.setChecked(True)

        self.use_gpu_checkbox = QCheckBox("Use GPU for truecolor")
        self.use_gpu_checkbox.setChecked(True)
        self.worker_combo = QComboBox()
        self.worker_combo.addItems(["1", "2", "4", "8"])
        self.worker_combo.setCurrentText("2")

        self.plot_button = QPushButton("Plot ROI Spectrum")
        self.plot_button.clicked.connect(self._plot_roi_spectrum)
        self.prepare_roi_button = QPushButton("Prepare ROI Layer")
        self.prepare_roi_button.clicked.connect(self._prepare_roi_layer)
        self.clear_roi_button = QPushButton("Clear Active ROI")
        self.clear_roi_button.clicked.connect(self._clear_active_roi_layer)

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

        self.status_label = QLabel("Select a spectral layer and optional rectangle in a Shapes layer.")
        self.dataset_combo = QComboBox()
        self.dataset_combo.setEnabled(False)
        self.dataset_combo.addItem("No stored ROI datasets")
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Plot mode"))
        controls.addWidget(self.mode_combo)
        controls.addWidget(self.individual_checkbox)
        controls.addWidget(self.pool_checkbox)
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

        buttons = QHBoxLayout()
        buttons.addWidget(self.prepare_roi_button)
        buttons.addWidget(self.clear_roi_button)
        buttons.addWidget(self.plot_button)
        buttons.addWidget(self.show_split_button)
        buttons.addWidget(self.show_truecolor_button)
        buttons.addWidget(self.show_pseudocolor_button)

        config_buttons = QHBoxLayout()
        config_buttons.addWidget(self.save_config_button)
        config_buttons.addWidget(self.load_config_button)
        config_buttons.addWidget(self.batch_pseudocolor_button)

        dataset_buttons = QHBoxLayout()
        dataset_buttons.addWidget(QLabel("Stored ROI datasets"))
        dataset_buttons.addWidget(self.dataset_combo)
        dataset_buttons.addWidget(self.export_dataset_button)
        dataset_buttons.addWidget(self.export_all_datasets_button)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(pseudocolor_controls)
        layout.addLayout(buttons)
        layout.addLayout(config_buttons)
        layout.addLayout(dataset_buttons)
        layout.addWidget(self.canvas)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self._refresh_dataset_combo()
        float_parent_dock_later(self)

    def _active_spectral_layer(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            raise ValueError("No active layer selected.")
        metadata = getattr(layer, "metadata", {})
        cube = metadata.get("spectral_cube")
        wavelengths = np.asarray(metadata.get("wavelengths_nm", []), dtype=np.float32)
        if cube is None or wavelengths.size == 0:
            raise ValueError("Active layer does not contain spectral metadata.")
        return layer, cube, wavelengths, metadata

    def _extract_roi(self, cube: np.ndarray) -> np.ndarray:
        rois = self._extract_rois(cube)
        return rois[-1] if rois else cube

    def _extract_rois(self, cube: np.ndarray) -> list[np.ndarray]:
        roi_items = self._collect_roi_items(cube)
        return [item["roi_cube"] for item in roi_items]

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

    def _ensure_roi_shapes_layer(self, spectral_layer_name: str):
        layer = self._find_roi_shapes_layer(spectral_layer_name)
        if layer is not None:
            self.viewer.layers.selection.active = layer
            return layer
        layer = self.viewer.add_shapes(
            name=self._roi_layer_name(spectral_layer_name),
            edge_color="yellow",
            face_color="transparent",
            edge_width=2,
        )
        self.viewer.layers.selection.active = layer
        return layer

    def _ensure_roi_label_layer(self, spectral_layer_name: str):
        layer = self._find_roi_label_layer(spectral_layer_name)
        if layer is not None:
            return layer
        return self.viewer.add_points(
            np.empty((0, 2), dtype=np.float32),
            name=self._roi_label_layer_name(spectral_layer_name),
            size=1,
            opacity=0.0,
            properties={"label": np.asarray([], dtype=object)},
            text={"string": "{label}", "color": "yellow", "size": 12, "anchor": "center"},
        )

    def _prepare_roi_layer(self):
        try:
            spectral_layer, _cube, _wavelengths, _metadata = self._active_spectral_layer()
            roi_layer = self._ensure_roi_shapes_layer(spectral_layer.name)
            self._ensure_roi_label_layer(spectral_layer.name)
            self.status_label.setText(
                f"Using ROI layer '{roi_layer.name}'. Draw ROIs there for the active image only."
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

    def _clear_active_roi_layer(self):
        try:
            spectral_layer, _cube, _wavelengths, _metadata = self._active_spectral_layer()
            roi_layer = self._find_roi_shapes_layer(spectral_layer.name)
            if roi_layer is None:
                self.status_label.setText("No ROI layer exists yet for the active image.")
                return
            roi_layer.data = []
            roi_layer.refresh()
            label_layer = self._ensure_roi_label_layer(spectral_layer.name)
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
        roi_layer = self._find_roi_shapes_layer(spectral_layer.name)
        if roi_layer is None:
            return []
        rois = []
        for shape_index, shape in enumerate(roi_layer.data):
            vertices = np.asarray(shape)
            y_min = int(np.floor(vertices[:, 0].min()))
            y_max = int(np.ceil(vertices[:, 0].max()))
            x_min = int(np.floor(vertices[:, 1].min()))
            x_max = int(np.ceil(vertices[:, 1].max()))
            roi = cube[:, max(0, y_min):y_max, max(0, x_min):x_max]
            if roi.size > 0 and roi.shape[1] > 0 and roi.shape[2] > 0:
                x_center = float(vertices[:, 1].mean())
                rois.append(
                    {
                        "x_center": x_center,
                        "roi_cube": roi,
                        "layer": roi_layer,
                        "shape_index": shape_index,
                    }
                )
        rois.sort(key=lambda item: item["x_center"])
        return rois

    def _apply_roi_shape_labels(self, roi_items: list[dict]):
        labels_by_layer = {}
        for index, item in enumerate(roi_items, start=1):
            layer = item["layer"]
            entries = labels_by_layer.setdefault(layer, [])
            vertices = np.asarray(layer.data[item["shape_index"]], dtype=np.float32)
            center = vertices.mean(axis=0)
            entries.append((center, f"ROI {index}"))

        for layer, entries in labels_by_layer.items():
            label_layer = self._ensure_roi_label_layer(layer.name.removesuffix(self.ROI_LAYER_SUFFIX))
            points = np.asarray([center for center, _label in entries], dtype=np.float32)
            labels = np.asarray([label for _center, label in entries], dtype=object)
            label_layer.data = points
            label_layer.properties = {"label": labels}
            label_layer.text = {"string": "{label}", "color": "yellow", "size": 12, "anchor": "center"}
            label_layer.refresh()

    def _store_roi_dataset(
        self,
        *,
        layer_name: str,
        wavelengths: np.ndarray,
        roi_labels: list[str],
        roi_spectra: list[np.ndarray],
        pooled_spectrum: np.ndarray | None,
    ):
        dataset = ROI_SPECTRUM_STORE.add_dataset(
            source_layer_name=layer_name,
            mode=self.mode_combo.currentText(),
            wavelengths_nm=wavelengths,
            roi_labels=roi_labels,
            roi_spectra=np.stack(roi_spectra, axis=0),
            pooled_spectrum=pooled_spectrum,
        )
        self._refresh_dataset_combo(select_dataset_id=dataset.dataset_id)
        return dataset

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

    def _plot_roi_spectrum(self):
        try:
            layer, cube, wavelengths, _metadata = self._active_spectral_layer()
            roi_items = self._collect_roi_items(cube)
            roi_cubes = [item["roi_cube"] for item in roi_items] or [cube]
            self.figure.clear()
            axis = self.figure.add_subplot(111)
            pooled_spectra = []
            roi_labels = []
            for index, roi_cube in enumerate(roi_cubes, start=1):
                spectrum = roi_cube.mean(axis=(1, 2))
                if self.mode_combo.currentText() == "Normalized":
                    spectrum = spectrum - spectrum.min()
                    spectrum = spectrum / (spectrum.max() + 1e-8)
                pooled_spectra.append(spectrum)
                roi_labels.append(f"ROI {index}")
                if self.individual_checkbox.isChecked():
                    axis.plot(wavelengths, spectrum, linewidth=1.5, label=f"ROI {index}")

            if roi_items:
                self._apply_roi_shape_labels(roi_items)

            pooled = None
            if self.pool_checkbox.isChecked() and pooled_spectra:
                pooled = np.mean(np.stack(pooled_spectra, axis=0), axis=0)
                axis.plot(wavelengths, pooled, linewidth=3, linestyle="--", color="black", label="Pooled")

            axis.set_xlabel("Wavelength (nm)")
            axis.set_ylabel("Normalized intensity" if self.mode_combo.currentText() == "Normalized" else "Intensity")
            axis.set_title("ROI spectrum")
            axis.grid(True, alpha=0.3)
            axis.legend()
            self.canvas.draw()
            stored_dataset = self._store_roi_dataset(
                layer_name=layer.name,
                wavelengths=wavelengths,
                roi_labels=roi_labels,
                roi_spectra=pooled_spectra,
                pooled_spectrum=pooled,
            )
            self.status_label.setText(
                f"Spectrum updated for {len(roi_cubes)} ROI(s). Stored {stored_dataset.name} in memory."
            )
        except Exception as exc:
            self.status_label.setText(str(exc))

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
        rois = self._extract_rois(cube)
        mode = self.pseudocolor_mode_combo.currentText()
        if not rois:
            raise ValueError("Define at least one ROI for pseudocolor.")
        bg_spectrum = None
        if self.bg_subtraction_checkbox.isChecked() and len(rois) >= 2:
            bg_spectrum = rois[0].mean(axis=(1, 2))
        if mode == "roi_pair":
            if len(rois) < 2:
                raise ValueError("ROI pair mode requires at least two ROIs. Leftmost ROI is blue/left, rightmost ROI is red/right.")
            return pseudocolor_pair_config(
                left_reference=rois[0].mean(axis=(1, 2)),
                right_reference=rois[-1].mean(axis=(1, 2)),
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
            reference_spectrum = np.mean(np.stack([roi.mean(axis=(1, 2)) for roi in rois], axis=0), axis=0)
        else:
            reference_spectrum = rois[-1].mean(axis=(1, 2))
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
