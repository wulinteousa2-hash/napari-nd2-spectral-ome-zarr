from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ._nd2 import load_nd2_dataset
from ._ome_zarr import export_dataset_to_ome_zarr
from ._qt_utils import float_parent_dock_later
from ._reader import build_layer_data, inspect_ome_zarr
from ._spectral import get_gpu_status_text, gpu_available


def _convert_one_nd2_to_ome_zarr(nd2_path: str, input_root: str, output_root: str) -> str:
    dataset = load_nd2_dataset(nd2_path)
    source = Path(nd2_path)
    relative_parent = source.relative_to(Path(input_root)).parent
    output_path = Path(output_root) / relative_parent / f"{source.stem}.ome.zarr"
    export_dataset_to_ome_zarr(
        data_tczyx=dataset.data_tczyx,
        output_path=str(output_path),
        metadata=dataset.metadata,
    )
    return str(output_path)


class BatchExportWorker(QObject):
    progress = Signal(str)
    finished = Signal(int, str)
    failed = Signal(str)

    def __init__(self, input_dir: str, output_dir: str, max_workers: int):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_workers = max_workers

    def run(self):
        try:
            input_root = Path(self.input_dir)
            output_root = Path(self.output_dir)
            nd2_paths = sorted(input_root.rglob("*.nd2"))
            if not nd2_paths:
                raise ValueError("No ND2 files found in the selected folder.")

            exported_count = 0
            self.progress.emit(f"Starting batch export with {self.max_workers} worker(s) for {len(nd2_paths)} file(s)...")
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {
                    executor.submit(_convert_one_nd2_to_ome_zarr, str(path), str(input_root), str(output_root)): str(path)
                    for path in nd2_paths
                }
                for future in as_completed(future_map):
                    source_path = future_map[future]
                    exported_path = future.result()
                    exported_count += 1
                    self.progress.emit(f"Exported {exported_count}/{len(nd2_paths)}: {Path(source_path).name} -> {Path(exported_path).name}")
            self.finished.emit(exported_count, str(output_root))
        except Exception as exc:
            self.failed.emit(str(exc))


class DropPathLineEdit(QLineEdit):
    pathDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return
        local_path = urls[0].toLocalFile()
        if local_path:
            self.setText(local_path)
            self.pathDropped.emit(local_path)
            event.acceptProposedAction()
        else:
            event.ignore()


class Nd2SpectralWidget(QWidget):
    ZARR_BATCH_COLUMNS = [
        "open",
        "name",
        "relative_path",
        "axes",
        "shape",
        "preview_shape",
        "wavelengths",
        "spectral",
    ]

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._dataset = None
        self._batch_thread = None
        self._batch_worker = None
        self._zarr_batch_entries: list[dict] = []
        self._updating_zarr_table = False

        self.open_input_edit = QLineEdit()
        self.zarr_input_edit = DropPathLineEdit()
        self.zarr_batch_root_edit = QLineEdit()
        self.batch_input_edit = QLineEdit()
        self.output_dir_edit = QLineEdit()
        self.status_label = QLabel("Select one ND2 file to preview, drag-drop a Zarr folder, or batch export ND2 files.")
        self.gpu_checkbox = QCheckBox("Use GPU for truecolor render")
        self.gpu_checkbox.setChecked(gpu_available())
        self.gpu_checkbox.setEnabled(gpu_available())
        self.gpu_checkbox.stateChanged.connect(self._update_gpu_indicator)
        self.gpu_indicator = QLabel()
        self.gpu_indicator.setAlignment(Qt.AlignCenter)
        self.worker_count_edit = QLineEdit()
        self.worker_count_edit.setText(str(self._default_worker_count()))
        self.zarr_gray_checkbox = QCheckBox("Visible sum")
        self.zarr_truecolor_checkbox = QCheckBox("Truecolor")
        self.zarr_truecolor_checkbox.setChecked(True)
        self.zarr_raw_checkbox = QCheckBox("Raw spectral")
        self.zarr_preview_checkbox = QCheckBox("Use preview pyramid level")
        self.zarr_preview_checkbox.setChecked(True)
        self.zarr_info_label = QLabel("No Zarr selected.")
        self.zarr_info_label.setWordWrap(True)
        self.zarr_batch_table = QTableWidget()
        self.zarr_batch_table.setAlternatingRowColors(False)
        self._configure_zarr_batch_table_palette()
        self._update_gpu_indicator()
        self.zarr_input_edit.pathDropped.connect(self._update_zarr_info)
        self.zarr_input_edit.textChanged.connect(self._update_zarr_info)

        browse_open_input = QPushButton("Browse ND2")
        browse_open_input.clicked.connect(self._pick_open_input_path)

        open_button = QPushButton("Open ND2")
        open_button.clicked.connect(self._open_nd2)

        browse_zarr_input = QPushButton("Browse Zarr")
        browse_zarr_input.clicked.connect(self._pick_zarr_input_path)

        open_zarr_button = QPushButton("Open Zarr")
        open_zarr_button.clicked.connect(self._open_zarr)

        browse_zarr_batch_root = QPushButton("Browse Zarr Folder")
        browse_zarr_batch_root.clicked.connect(self._pick_zarr_batch_root)

        scan_zarr_button = QPushButton("Scan Zarr Folder")
        scan_zarr_button.clicked.connect(self._scan_zarr_batch_root)

        select_all_zarr_button = QPushButton("Select All")
        select_all_zarr_button.clicked.connect(lambda: self._set_all_zarr_rows_checked(True))

        clear_all_zarr_button = QPushButton("Clear All")
        clear_all_zarr_button.clicked.connect(lambda: self._set_all_zarr_rows_checked(False))

        open_selected_zarr_button = QPushButton("Open Selected Zarr")
        open_selected_zarr_button.clicked.connect(self._open_selected_zarr_batch)

        browse_batch_input = QPushButton("Browse Folder")
        browse_batch_input.clicked.connect(self._pick_batch_input_path)

        browse_output = QPushButton("Browse Output")
        browse_output.clicked.connect(self._pick_output_dir)

        export_button = QPushButton("Batch Export OME-Zarr")
        export_button.clicked.connect(self._export_ome_zarr_batch)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Preview ND2 input"))
        open_input_row = QHBoxLayout()
        open_input_row.addWidget(self.open_input_edit)
        open_input_row.addWidget(browse_open_input)
        layout.addLayout(open_input_row)
        layout.addWidget(open_button)

        layout.addWidget(QLabel("Open OME-Zarr input"))
        zarr_input_row = QHBoxLayout()
        zarr_input_row.addWidget(self.zarr_input_edit)
        zarr_input_row.addWidget(browse_zarr_input)
        layout.addLayout(zarr_input_row)

        zarr_options_row = QHBoxLayout()
        zarr_options_row.addWidget(self.zarr_gray_checkbox)
        zarr_options_row.addWidget(self.zarr_truecolor_checkbox)
        zarr_options_row.addWidget(self.zarr_raw_checkbox)
        zarr_options_row.addWidget(self.zarr_preview_checkbox)
        layout.addLayout(zarr_options_row)
        layout.addWidget(open_zarr_button)
        layout.addWidget(self.zarr_info_label)

        layout.addWidget(QLabel("Batch OME-Zarr folder"))
        zarr_batch_row = QHBoxLayout()
        zarr_batch_row.addWidget(self.zarr_batch_root_edit)
        zarr_batch_row.addWidget(browse_zarr_batch_root)
        zarr_batch_row.addWidget(scan_zarr_button)
        layout.addLayout(zarr_batch_row)

        zarr_batch_actions = QHBoxLayout()
        zarr_batch_actions.addWidget(select_all_zarr_button)
        zarr_batch_actions.addWidget(clear_all_zarr_button)
        zarr_batch_actions.addWidget(open_selected_zarr_button)
        layout.addLayout(zarr_batch_actions)
        layout.addWidget(self.zarr_batch_table)

        layout.addWidget(QLabel("Batch ND2 input folder"))
        batch_input_row = QHBoxLayout()
        batch_input_row.addWidget(self.batch_input_edit)
        batch_input_row.addWidget(browse_batch_input)
        layout.addLayout(batch_input_row)

        layout.addWidget(QLabel("Batch OME-Zarr output folder"))
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(browse_output)
        layout.addLayout(output_row)

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.gpu_checkbox)
        gpu_row.addWidget(self.gpu_indicator)
        layout.addLayout(gpu_row)

        layout.addWidget(QLabel("Batch worker count"))
        layout.addWidget(self.worker_count_edit)
        layout.addWidget(export_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        float_parent_dock_later(self)

    def _default_worker_count(self) -> int:
        cpu = os.cpu_count() or 4
        return max(2, min(8, cpu // 2))

    def _update_gpu_indicator(self):
        enabled = self.gpu_checkbox.isChecked() and self.gpu_checkbox.isEnabled()
        self.gpu_indicator.setText(get_gpu_status_text(enabled))
        if enabled:
            self.gpu_indicator.setStyleSheet("background-color: #1f7a1f; color: white; padding: 4px;")
        else:
            self.gpu_indicator.setStyleSheet("background-color: #6f1d1b; color: white; padding: 4px;")

    def _pick_open_input_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select ND2 file", "", "ND2 files (*.nd2)")
        if path:
            self.open_input_edit.setText(path)
            if not self.batch_input_edit.text().strip():
                self.batch_input_edit.setText(str(Path(path).parent))
            if not self.output_dir_edit.text().strip():
                self.output_dir_edit.setText(str(Path(path).parent / "ome_zarr_output"))

    def _pick_batch_input_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select ND2 input folder")
        if path:
            self.batch_input_edit.setText(path)
            if not self.output_dir_edit.text().strip():
                self.output_dir_edit.setText(str(Path(path) / "ome_zarr_output"))

    def _pick_zarr_input_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select OME-Zarr folder")
        if path:
            self.zarr_input_edit.setText(path)

    def _pick_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select OME-Zarr output folder")
        if path:
            self.output_dir_edit.setText(path)

    def _pick_zarr_batch_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder containing OME-Zarr datasets")
        if path:
            self.zarr_batch_root_edit.setText(path)

    def _update_zarr_info(self, path: str | None = None):
        selected_path = (path or self.zarr_input_edit.text()).strip()
        if not selected_path:
            self.zarr_info_label.setText("No Zarr selected.")
            return
        if not selected_path.lower().endswith(".zarr"):
            self.zarr_info_label.setText("Selected path is not a .zarr folder.")
            return
        try:
            info = inspect_ome_zarr(selected_path)
            wavelengths_text = (
                f"{info['wavelength_count']} channels, {info['wavelength_min_nm']:.1f}-{info['wavelength_max_nm']:.1f} nm"
                if info["wavelength_count"] > 0 and info["wavelength_min_nm"] is not None and info["wavelength_max_nm"] is not None
                else "no wavelength metadata"
            )
            self.zarr_info_label.setText(
                f"Name: {info['name']}\n"
                f"Axes: {''.join(info['axes'])}\n"
                f"Shape: {info['shape']}\n"
                f"Preview shape: {info['preview_shape']}\n"
                f"Spectral: {info['is_spectral']}\n"
                f"Wavelengths: {wavelengths_text}"
            )
        except Exception as exc:
            self.zarr_info_label.setText(f"Could not inspect Zarr: {exc}")

    def _open_nd2(self):
        path = self.open_input_edit.text().strip()
        if not path:
            self.status_label.setText("Choose an ND2 file first.")
            return
        for data, kwargs, layer_type in build_layer_data(path, use_gpu=self._use_gpu()):
            self.viewer.add_image(data, **kwargs) if layer_type == "image" else None
        self._dataset = load_nd2_dataset(path)
        self.status_label.setText(f"Opened {Path(path).name}")

    def _open_zarr(self):
        path = self.zarr_input_edit.text().strip()
        if not path:
            self.status_label.setText("Choose an OME-Zarr folder first.")
            return
        if not path.lower().endswith(".zarr"):
            self.status_label.setText("Selected folder must end with .zarr.")
            return
        if not (self.zarr_gray_checkbox.isChecked() or self.zarr_truecolor_checkbox.isChecked() or self.zarr_raw_checkbox.isChecked()):
            self.status_label.setText("Select at least one Zarr view to open.")
            return
        try:
            for data, kwargs, layer_type in build_layer_data(
                path,
                use_gpu=self._use_gpu(),
                include_visible_layer=self.zarr_gray_checkbox.isChecked(),
                include_truecolor_layer=self.zarr_truecolor_checkbox.isChecked(),
                include_raw_layer=self.zarr_raw_checkbox.isChecked(),
                zarr_use_preview=self.zarr_preview_checkbox.isChecked(),
            ):
                self.viewer.add_image(data, **kwargs) if layer_type == "image" else None
            self.status_label.setText(f"Opened {Path(path).name} with selected Zarr views.")
        except Exception as exc:
            self.status_label.setText(f"Failed to open Zarr: {exc}")

    def _scan_zarr_batch_root(self):
        root_text = self.zarr_batch_root_edit.text().strip()
        if not root_text:
            self.status_label.setText("Choose a folder containing .zarr datasets first.")
            return
        root = Path(root_text)
        if not root.exists():
            self.status_label.setText("Selected batch Zarr folder does not exist.")
            return
        zarr_paths = sorted({path for path in root.rglob("*.zarr") if path.is_dir()})
        if not zarr_paths:
            self._zarr_batch_entries = []
            self._populate_zarr_batch_table()
            self.status_label.setText("No .zarr folders found in the selected directory.")
            return

        entries = []
        for path in zarr_paths:
            try:
                info = inspect_ome_zarr(str(path))
                wavelength_text = (
                    f"{info['wavelength_count']} ({info['wavelength_min_nm']:.1f}-{info['wavelength_max_nm']:.1f} nm)"
                    if info["wavelength_count"] > 0 and info["wavelength_min_nm"] is not None and info["wavelength_max_nm"] is not None
                    else "0"
                )
                entries.append(
                    {
                        "open": True,
                        "path": str(path),
                        "name": info["name"],
                        "relative_path": f"./{path.parent.relative_to(root)}/" if path.parent != root else "./",
                        "axes": "".join(info["axes"]),
                        "shape": str(info["shape"]),
                        "preview_shape": str(info["preview_shape"]),
                        "wavelengths": wavelength_text,
                        "spectral": str(info["is_spectral"]),
                    }
                )
            except Exception as exc:
                entries.append(
                    {
                        "open": False,
                        "path": str(path),
                        "name": path.name,
                        "relative_path": f"./{path.parent.relative_to(root)}/" if path.parent != root else "./",
                        "axes": "error",
                        "shape": "error",
                        "preview_shape": "error",
                        "wavelengths": str(exc),
                        "spectral": "error",
                    }
                )
        self._zarr_batch_entries = entries
        self._populate_zarr_batch_table()
        self.status_label.setText(f"Found {len(entries)} Zarr dataset(s).")

    def _populate_zarr_batch_table(self):
        self._updating_zarr_table = True
        self.zarr_batch_table.clear()
        self.zarr_batch_table.setColumnCount(len(self.ZARR_BATCH_COLUMNS))
        self.zarr_batch_table.setHorizontalHeaderLabels(self.ZARR_BATCH_COLUMNS)
        self.zarr_batch_table.setRowCount(len(self._zarr_batch_entries))
        for row_index, entry in enumerate(self._zarr_batch_entries):
            for column_index, column_name in enumerate(self.ZARR_BATCH_COLUMNS):
                if column_name == "open":
                    item = QTableWidgetItem()
                    item.setCheckState(Qt.Checked if entry["open"] else Qt.Unchecked)
                    item.setFlags((item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled) & ~Qt.ItemIsEditable)
                else:
                    item = QTableWidgetItem(str(entry[column_name]))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._style_zarr_batch_item(item, row_index)
                item.setData(Qt.UserRole, row_index)
                self.zarr_batch_table.setItem(row_index, column_index, item)
        self.zarr_batch_table.resizeColumnsToContents()
        self._updating_zarr_table = False

    def _configure_zarr_batch_table_palette(self):
        palette = self.zarr_batch_table.palette()
        text_color = palette.color(QPalette.Text)
        base_color = palette.color(QPalette.Base)

        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            palette.setColor(group, QPalette.Text, text_color)
            palette.setColor(group, QPalette.WindowText, text_color)
            palette.setColor(group, QPalette.HighlightedText, text_color)
            palette.setColor(group, QPalette.Base, base_color)
            palette.setColor(group, QPalette.AlternateBase, base_color)

        self.zarr_batch_table.setPalette(palette)

    def _style_zarr_batch_item(self, item: QTableWidgetItem, row_index: int):
        palette = self.zarr_batch_table.palette()
        item.setForeground(palette.brush(QPalette.Text))
        item.setBackground(palette.brush(QPalette.Base))

    def _set_all_zarr_rows_checked(self, checked: bool):
        if not self._zarr_batch_entries:
            self.status_label.setText("No Zarr datasets loaded into the batch table.")
            return
        for entry in self._zarr_batch_entries:
            entry["open"] = checked
        self._populate_zarr_batch_table()
        self.status_label.setText(
            f"{'Selected' if checked else 'Cleared'} {len(self._zarr_batch_entries)} Zarr dataset(s) in the batch table."
        )

    def _open_selected_zarr_batch(self):
        selected_entries = []
        for row_index, entry in enumerate(self._zarr_batch_entries):
            item = self.zarr_batch_table.item(row_index, 0)
            is_checked = item is not None and item.checkState() == Qt.Checked
            entry["open"] = is_checked
            if is_checked:
                selected_entries.append(entry)
        if not selected_entries:
            self.status_label.setText("Select at least one Zarr dataset in the batch table.")
            return
        if not (self.zarr_gray_checkbox.isChecked() or self.zarr_truecolor_checkbox.isChecked() or self.zarr_raw_checkbox.isChecked()):
            self.status_label.setText("Select at least one Zarr view to open.")
            return
        opened_count = 0
        for entry in selected_entries:
            try:
                for data, kwargs, layer_type in build_layer_data(
                    entry["path"],
                    use_gpu=self._use_gpu(),
                    include_visible_layer=self.zarr_gray_checkbox.isChecked(),
                    include_truecolor_layer=self.zarr_truecolor_checkbox.isChecked(),
                    include_raw_layer=self.zarr_raw_checkbox.isChecked(),
                    zarr_use_preview=self.zarr_preview_checkbox.isChecked(),
                ):
                    self.viewer.add_image(data, **kwargs) if layer_type == "image" else None
                opened_count += 1
            except Exception as exc:
                self.status_label.setText(f"Failed to open {Path(entry['path']).name}: {exc}")
                return
        self.status_label.setText(f"Opened {opened_count} selected Zarr dataset(s).")

    def _use_gpu(self) -> bool:
        return self.gpu_checkbox.isEnabled() and self.gpu_checkbox.isChecked()

    def _export_ome_zarr_batch(self):
        if self._batch_thread is not None and self._batch_thread.isRunning():
            self.status_label.setText("Batch export is already running.")
            return

        input_dir = self.batch_input_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not input_dir:
            self.status_label.setText("Choose an ND2 input folder.")
            return
        if not output_dir:
            self.status_label.setText("Choose an OME-Zarr output folder.")
            return

        try:
            max_workers = max(1, int(self.worker_count_edit.text().strip()))
        except ValueError:
            self.status_label.setText("Worker count must be an integer.")
            return

        self._set_batch_controls_enabled(False)
        self._batch_thread = QThread(self)
        self._batch_worker = BatchExportWorker(input_dir=input_dir, output_dir=output_dir, max_workers=max_workers)
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self.status_label.setText)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._cleanup_batch_thread)
        self._batch_thread.start()

    def _set_batch_controls_enabled(self, enabled: bool):
        self.batch_input_edit.setEnabled(enabled)
        self.output_dir_edit.setEnabled(enabled)
        self.worker_count_edit.setEnabled(enabled)

    def _on_batch_finished(self, exported_count: int, output_root: str):
        self.status_label.setText(f"Batch exported {exported_count} file(s) to {output_root}")
        self._set_batch_controls_enabled(True)

    def _on_batch_failed(self, error_text: str):
        self.status_label.setText(f"Batch export failed: {error_text}")
        self._set_batch_controls_enabled(True)

    def _cleanup_batch_thread(self):
        if self._batch_worker is not None:
            self._batch_worker.deleteLater()
        if self._batch_thread is not None:
            self._batch_thread.deleteLater()
        self._batch_worker = None
        self._batch_thread = None
