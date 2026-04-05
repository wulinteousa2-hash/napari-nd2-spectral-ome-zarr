from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from qtpy.QtCore import QObject, Qt, QThread, QTimer, Signal
from qtpy.QtGui import QColor, QFontMetrics, QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QPlainTextEdit,
    QProgressBar,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from ._nd2 import load_nd2_dataset
from ._ome_zarr import export_dataset_to_ome_zarr
from ._qt_utils import float_parent_dock_later
from ._reader import build_layer_data, inspect_ome_zarr
from ._spectral import get_gpu_status_text, gpu_available


def _convert_one_nd2_to_ome_zarr(nd2_path: str, input_root: str, output_root: str) -> dict:
    dataset = load_nd2_dataset(nd2_path)
    source = Path(nd2_path)
    relative_parent = source.relative_to(Path(input_root)).parent
    output_path = Path(output_root) / relative_parent / f"{source.stem}.ome.zarr"
    export_dataset_to_ome_zarr(
        data_tczyx=dataset.data_tczyx,
        output_path=str(output_path),
        metadata=dataset.metadata,
    )
    return {
        "source_path": str(source),
        "relative_source_path": str(source.relative_to(Path(input_root))),
        "output_path": str(output_path),
        "relative_output_path": str(output_path.relative_to(Path(output_root))),
        "metadata": dataset.metadata,
    }


class BatchExportWorker(QObject):
    progress = Signal(str)
    progress_counts = Signal(int, int)
    file_result = Signal(str, str, str, str)
    finished = Signal(int, str, str, object)
    failed = Signal(str)

    def __init__(self, input_path: str, output_dir: str, max_workers: int):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.max_workers = max_workers

    def run(self):
        try:
            input_root, nd2_paths = _resolve_nd2_batch_source(self.input_path)
            output_root = Path(self.output_dir)
            if not nd2_paths:
                raise ValueError("No ND2 files found in the selected input.")

            exported_count = 0
            manifest_entries: list[dict] = []
            self.progress.emit(f"Starting batch export with {self.max_workers} worker(s) for {len(nd2_paths)} file(s)...")
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {
                    executor.submit(_convert_one_nd2_to_ome_zarr, str(path), str(input_root), str(output_root)): str(path)
                    for path in nd2_paths
                }
                failures: list[dict] = []
                for future in as_completed(future_map):
                    source_path = future_map[future]
                    try:
                        export_record = future.result()
                        exported_count += 1
                        manifest_entries.append(export_record)
                        self.file_result.emit(
                            str(export_record["source_path"]),
                            "Converted",
                            str(export_record["relative_output_path"]),
                            "",
                        )
                        self.progress.emit(
                            f"Converted {Path(source_path).name} -> {Path(export_record['output_path']).name}"
                        )
                    except Exception as exc:
                        failures.append({"source_path": source_path, "error": str(exc)})
                        self.file_result.emit(str(source_path), "Failed", "", str(exc))
                        self.progress.emit(f"Failed: {Path(source_path).name} -> {exc}")
                    self.progress_counts.emit(exported_count + len(failures), len(nd2_paths))
            manifest_path = _write_export_manifest(
                output_root=output_root,
                input_root=input_root,
                input_path=self.input_path,
                entries=manifest_entries,
            )
            self.finished.emit(exported_count, str(output_root), str(manifest_path), failures)
        except Exception as exc:
            self.failed.emit(str(exc))


def _resolve_nd2_batch_source(input_path: str) -> tuple[Path, list[Path]]:
    source = Path(input_path).expanduser()
    if not source.exists():
        raise ValueError("Selected ND2 input path does not exist.")
    if source.is_file():
        if source.suffix.lower() != ".nd2":
            raise ValueError("Selected input file must be an .nd2 file.")
        return source.parent, [source]
    nd2_paths = sorted(path for path in source.rglob("*.nd2") if path.is_file())
    return source, nd2_paths


def _write_export_manifest(*, output_root: Path, input_root: Path, input_path: str, entries: list[dict]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    temp_manifest_path = output_root / "manifest.json.__tmp__"
    payload = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(Path(input_path).expanduser()),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "export_count": len(entries),
        "entries": sorted(entries, key=lambda entry: str(entry.get("relative_source_path", ""))),
    }
    temp_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_manifest_path.replace(manifest_path)
    return manifest_path


class DropPathLineEdit(QLineEdit):
    pathDropped = Signal(str)

    def __init__(self, placeholder: str = ""):
        super().__init__()
        self.setAcceptDrops(True)
        self.setReadOnly(True)
        self.setMinimumHeight(44)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setPlaceholderText(placeholder)
        self.setStyleSheet(
            "QLineEdit {"
            "border: 2px dashed #60a5fa;"
            "border-radius: 10px;"
            "padding: 8px 10px;"
            "background-color: #1f2937;"
            "color: #f9fafb;"
            "selection-background-color: #2563eb;"
            "font-weight: 600;"
            "}"
            "QLineEdit:focus {"
            "border: 2px dashed #93c5fd;"
            "background-color: #243244;"
            "}"
            "QLineEdit[readOnly=\"true\"] {"
            "background-color: #1f2937;"
            "color: #f9fafb;"
            "}"
        )

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
            self.clearFocus()
            self.update()
            window = self.window()
            if window is not None:
                QTimer.singleShot(0, window.update)
                QTimer.singleShot(0, window.repaint)
        else:
            event.ignore()


class DropPathBox(QFrame):
    pathDropped = Signal(str)

    def __init__(self, title: str, hint: str, placeholder: str):
        super().__init__()
        self.path_edit = DropPathLineEdit(placeholder)
        self.path_edit.pathDropped.connect(self.pathDropped.emit)
        self.title_label = QLabel(title)
        self.hint_label = QLabel(hint)
        self.hint_label.setWordWrap(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame {"
            "border: 1px solid #4b5563;"
            "border-radius: 10px;"
            "background-color: #2a2f3a;"
            "}"
            "QLabel { color: #d1d5db; }"
        )
        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.path_edit)
        layout.addWidget(self.hint_label)
        self.setLayout(layout)

    def text(self) -> str:
        return self.path_edit.text()

    def setText(self, text: str):
        self.path_edit.setText(text)

    def setEnabled(self, enabled: bool):
        super().setEnabled(enabled)
        self.path_edit.setEnabled(enabled)


class SelectableTableWidget(QTableWidget):
    toggle_rows_requested = Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_rows_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class Nd2SpectralWidget(QWidget):
    ND2_BATCH_COLUMNS = [
        "name",
        "relative_path",
        "status",
        "output",
        "error",
    ]
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
        self._batch_table_mode = "none"
        self._nd2_batch_entries: list[dict] = []
        self._status_message = "Ready."
        self._zarr_batch_entries: list[dict] = []
        self._updating_zarr_table = False

        self.batch_input_box = DropPathBox(
            "ND2 source",
            "Drag a folder, subfolder, or single .nd2 file here. Folder inputs are scanned recursively.",
            "Drop ND2 file or folder here",
        )
        self.zarr_scan_box = DropPathBox(
            "OME-Zarr source",
            "Drag a folder to scan all nested .zarr datasets, or drag one .zarr folder to open just that dataset.",
            "Drop .zarr folder or parent folder here",
        )
        self.output_dir_box = DropPathBox(
            "OME-Zarr output",
            "Drag an output folder here, or browse to choose where converted datasets will be written.",
            "Drop output folder here",
        )
        self.status_label = QLabel()
        self.status_label.setWordWrap(False)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.status_label.setFixedHeight(34)
        self.status_label.setStyleSheet(
            "QLabel {"
            "background-color: #2b313c;"
            "color: #e5e7eb;"
            "padding: 8px;"
            "border: 1px solid #4b5563;"
            "border-radius: 6px;"
            "font-weight: 600;"
            "}"
        )
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
        self.zarr_batch_table = SelectableTableWidget()
        self.zarr_batch_table.setAlternatingRowColors(False)
        self.zarr_batch_table.setMinimumHeight(320)
        self.zarr_batch_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.zarr_batch_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.zarr_batch_table.setStyleSheet(
            "QTableWidget::indicator {"
            "width: 16px;"
            "height: 16px;"
            "border: 1px solid #374151;"
            "border-radius: 3px;"
            "background-color: #f8fafc;"
            "}"
            "QTableWidget::indicator:checked {"
            "background-color: #2563eb;"
            "border: 1px solid #1d4ed8;"
            "image: none;"
            "}"
            "QTableWidget::indicator:unchecked {"
            "background-color: #f8fafc;"
            "border: 1px solid #6b7280;"
            "}"
        )
        self.zarr_batch_table.toggle_rows_requested.connect(self._toggle_selected_zarr_rows)
        self.zarr_batch_table.itemSelectionChanged.connect(self._sync_selected_rows_to_zarr_checks)
        self.export_progress_bar = QProgressBar()
        self.export_progress_bar.setRange(0, 1)
        self.export_progress_bar.setValue(0)
        self.export_progress_bar.setFormat("Idle")
        self.export_error_log = QPlainTextEdit()
        self.export_error_log.setReadOnly(True)
        self.export_error_log.setPlaceholderText("Conversion failures will appear here for troubleshooting.")
        self._configure_zarr_batch_table_palette()
        self._update_gpu_indicator()
        self.batch_input_box.pathDropped.connect(self._handle_batch_input_dropped)
        self.zarr_scan_box.pathDropped.connect(self._handle_zarr_scan_dropped)
        self.output_dir_box.pathDropped.connect(self._handle_output_dir_dropped)

        browse_zarr_batch_root = QPushButton("Browse Zarr Folder")
        browse_zarr_batch_root.clicked.connect(self._pick_zarr_batch_root)

        scan_zarr_button = QPushButton("Scan Zarr Folder")
        scan_zarr_button.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; font-weight: 700; padding: 8px 12px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
        )
        scan_zarr_button.clicked.connect(self._scan_zarr_batch_root)

        self.select_all_zarr_button = QPushButton("Select All")
        self.select_all_zarr_button.clicked.connect(lambda: self._set_all_zarr_rows_checked(True))

        self.clear_all_zarr_button = QPushButton("Clear All")
        self.clear_all_zarr_button.clicked.connect(lambda: self._set_all_zarr_rows_checked(False))

        self.open_selected_zarr_button = QPushButton("Open Selected Zarr")
        self.open_selected_zarr_button.clicked.connect(self._open_selected_zarr_batch)

        browse_batch_input = QPushButton("Browse Folder")
        browse_batch_input.clicked.connect(self._pick_batch_input_path)

        browse_output = QPushButton("Browse Output")
        browse_output.clicked.connect(self._pick_output_dir)

        export_button = QPushButton("Convert To OME-Zarr")
        export_button.setStyleSheet(
            "QPushButton { background-color: #0f766e; color: white; font-weight: 700; padding: 8px 12px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #0d9488; }"
        )
        export_button.clicked.connect(self._export_ome_zarr_batch)

        layout = QVBoxLayout()
        export_group = QGroupBox("ND2 Spectral Export")
        export_layout = QVBoxLayout()
        export_layout.addWidget(self.status_label)
        export_layout.addWidget(self.export_progress_bar)
        export_layout.addWidget(export_button)

        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch_input_box, 1)
        batch_row.addWidget(QLabel("or"))
        batch_row.addWidget(browse_batch_input, 0)
        export_layout.addLayout(batch_row)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_box, 1)
        output_row.addWidget(QLabel("or"))
        output_row.addWidget(browse_output, 0)
        export_layout.addLayout(output_row)

        zarr_source_row = QHBoxLayout()
        zarr_source_row.addWidget(self.zarr_scan_box, 1)
        zarr_source_row.addWidget(QLabel("or"))
        zarr_source_row.addWidget(browse_zarr_batch_root, 0)
        export_layout.addLayout(zarr_source_row)

        export_layout.addWidget(scan_zarr_button)

        zarr_options_row = QHBoxLayout()
        zarr_options_row.addWidget(self.zarr_gray_checkbox)
        zarr_options_row.addWidget(self.zarr_truecolor_checkbox)
        zarr_options_row.addWidget(self.zarr_raw_checkbox)
        zarr_options_row.addWidget(self.zarr_preview_checkbox)
        export_layout.addLayout(zarr_options_row)

        zarr_batch_actions = QHBoxLayout()
        zarr_batch_actions.addWidget(self.select_all_zarr_button)
        zarr_batch_actions.addWidget(self.clear_all_zarr_button)
        zarr_batch_actions.addWidget(self.open_selected_zarr_button)
        export_layout.addLayout(zarr_batch_actions)
        export_layout.addWidget(self.zarr_batch_table)

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.gpu_checkbox)
        gpu_row.addWidget(self.gpu_indicator)
        export_layout.addLayout(gpu_row)

        export_layout.addWidget(QLabel("Batch worker count"))
        export_layout.addWidget(self.worker_count_edit)
        error_group = QGroupBox("Conversion Errors")
        error_layout = QVBoxLayout()
        error_layout.addWidget(self.export_error_log)
        error_group.setLayout(error_layout)
        export_layout.addWidget(error_group)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        self.setLayout(layout)
        self._update_status_label()
        float_parent_dock_later(self)

    def _default_worker_count(self) -> int:
        cpu = os.cpu_count() or 4
        return max(2, min(8, cpu // 2))

    def _update_status_label(self):
        prefix = "Status: "
        metrics = QFontMetrics(self.status_label.font())
        available_width = max(120, self.status_label.width() - 16)
        elided = metrics.elidedText(prefix + self._status_message, Qt.ElideRight, available_width)
        self.status_label.setText(elided)
        self.status_label.setToolTip(prefix + self._status_message)

    def _set_status(self, message: str):
        self._status_message = str(message).replace("\n", " ").strip() or "Ready."
        self._update_status_label()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_status_label()

    def _update_gpu_indicator(self):
        enabled = self.gpu_checkbox.isChecked() and self.gpu_checkbox.isEnabled()
        self.gpu_indicator.setText(get_gpu_status_text(enabled))
        if enabled:
            self.gpu_indicator.setStyleSheet("background-color: #1f7a1f; color: white; padding: 4px;")
        else:
            self.gpu_indicator.setStyleSheet("background-color: #6f1d1b; color: white; padding: 4px;")

    def _pick_batch_input_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select ND2 input folder")
        if path:
            self.batch_input_box.setText(path)
            self._update_default_output_dir(path)
            self._scan_nd2_batch_source(path)

    def _pick_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select OME-Zarr output folder")
        if path:
            self.output_dir_box.setText(path)

    def _update_default_output_dir(self, input_path: str):
        if self.output_dir_box.text().strip():
            return
        source = Path(input_path)
        default_root = source.parent if source.is_file() else source
        self.output_dir_box.setText(str(default_root / "ome_zarr_output"))

    def _handle_batch_input_dropped(self, path: str):
        self._update_default_output_dir(path)
        try:
            self._scan_nd2_batch_source(path)
        except Exception as exc:
            self._set_status(str(exc))

    def _handle_output_dir_dropped(self, path: str):
        output_path = Path(path)
        self.output_dir_box.setText(str(output_path if output_path.is_dir() else output_path.parent))

    def _pick_zarr_batch_root(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder containing OME-Zarr datasets")
        if path:
            self.zarr_scan_box.setText(path)

    def _handle_zarr_scan_dropped(self, path: str):
        self.zarr_scan_box.setText(path)
        selected_path = Path(path)
        if selected_path.is_dir() and selected_path.suffix.lower() == ".zarr":
            self._scan_zarr_batch_root()
            return
        if selected_path.is_dir():
            self._scan_zarr_batch_root()
            return
        self._set_status("Drop a folder or .zarr directory into the OME-Zarr source box.")

    def _scan_nd2_batch_source(self, source_text: str | None = None):
        path_text = (source_text or self.batch_input_box.text()).strip()
        if not path_text:
            self._set_status("Choose or drop an ND2 source first.")
            return
        input_root, nd2_paths = _resolve_nd2_batch_source(path_text)
        if not nd2_paths:
            self._nd2_batch_entries = []
            self._populate_nd2_batch_table()
            self._set_status("No ND2 files found in the selected source.")
            return

        output_root = Path(self.output_dir_box.text().strip()) if self.output_dir_box.text().strip() else None
        entries = []
        for path in nd2_paths:
            relative_source = path.relative_to(input_root)
            planned_output = ""
            if output_root is not None:
                planned_output = str((output_root / relative_source.parent / f"{path.stem}.ome.zarr").relative_to(output_root))
            entries.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "relative_path": str(relative_source),
                    "status": "Ready",
                    "output": planned_output,
                    "error": "",
                }
            )
        self._nd2_batch_entries = entries
        self._populate_nd2_batch_table()
        if Path(path_text).is_file():
            self._set_status("Selected 1 ND2 file. Conversion status will be tracked in the table below.")
        else:
            self._set_status(
                f"Loaded {len(entries)} ND2 file(s) for conversion. Original folder structure will be preserved."
            )

    def _scan_zarr_batch_root(self):
        root_text = self.zarr_scan_box.text().strip()
        if not root_text:
            self._set_status("Choose or drop a folder containing .zarr datasets first.")
            return
        root = Path(root_text)
        if not root.exists():
            self._set_status("Selected OME-Zarr source does not exist.")
            return
        if root.is_dir() and root.suffix.lower() == ".zarr":
            zarr_paths = [root]
            scan_root = root.parent
        else:
            zarr_paths = sorted({path for path in root.rglob("*.zarr") if path.is_dir()})
            scan_root = root
        if not zarr_paths:
            self._zarr_batch_entries = []
            self._populate_zarr_batch_table()
            self._set_status("No .zarr folders found in the selected directory.")
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
                        "relative_path": f"./{path.parent.relative_to(scan_root)}/" if path.parent != scan_root else "./",
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
                        "relative_path": f"./{path.parent.relative_to(scan_root)}/" if path.parent != scan_root else "./",
                        "axes": "error",
                        "shape": "error",
                        "preview_shape": "error",
                        "wavelengths": str(exc),
                        "spectral": "error",
                    }
                )
        self._zarr_batch_entries = entries
        self._populate_zarr_batch_table()
        self._set_status(f"Found {len(entries)} Zarr dataset(s).")

    def _populate_zarr_batch_table(self):
        self._batch_table_mode = "zarr"
        self.select_all_zarr_button.setEnabled(True)
        self.clear_all_zarr_button.setEnabled(True)
        self.open_selected_zarr_button.setEnabled(True)
        path_group_colors: dict[str, tuple[QColor, QColor]] = {}
        self._updating_zarr_table = True
        self.zarr_batch_table.clear()
        self.zarr_batch_table.setColumnCount(len(self.ZARR_BATCH_COLUMNS))
        self.zarr_batch_table.setHorizontalHeaderLabels(self.ZARR_BATCH_COLUMNS)
        self.zarr_batch_table.setRowCount(len(self._zarr_batch_entries))
        for row_index, entry in enumerate(self._zarr_batch_entries):
            relative_path = str(entry.get("relative_path", ""))
            if relative_path not in path_group_colors:
                if len(path_group_colors) % 2 == 0:
                    path_group_colors[relative_path] = (QColor("#dbeafe"), QColor("#111827"))
                else:
                    path_group_colors[relative_path] = (QColor("#dcfce7"), QColor("#111827"))
            background_color, foreground_color = path_group_colors[relative_path]
            for column_index, column_name in enumerate(self.ZARR_BATCH_COLUMNS):
                if column_name == "open":
                    item = QTableWidgetItem()
                    item.setCheckState(Qt.Checked if entry["open"] else Qt.Unchecked)
                    item.setFlags((item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled) & ~Qt.ItemIsEditable)
                else:
                    item = QTableWidgetItem(str(entry[column_name]))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._style_zarr_batch_item(item, row_index, background_color=background_color, foreground_color=foreground_color)
                item.setData(Qt.UserRole, row_index)
                self.zarr_batch_table.setItem(row_index, column_index, item)
        self.zarr_batch_table.resizeColumnsToContents()
        self._updating_zarr_table = False

    def _populate_nd2_batch_table(self):
        self._batch_table_mode = "nd2"
        self.select_all_zarr_button.setEnabled(False)
        self.clear_all_zarr_button.setEnabled(False)
        self.open_selected_zarr_button.setEnabled(False)
        self._updating_zarr_table = True
        self.zarr_batch_table.clear()
        self.zarr_batch_table.setColumnCount(len(self.ND2_BATCH_COLUMNS))
        self.zarr_batch_table.setHorizontalHeaderLabels(self.ND2_BATCH_COLUMNS)
        self.zarr_batch_table.setRowCount(len(self._nd2_batch_entries))
        for row_index, entry in enumerate(self._nd2_batch_entries):
            for column_index, column_name in enumerate(self.ND2_BATCH_COLUMNS):
                item = QTableWidgetItem(str(entry.get(column_name, "")))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._style_zarr_batch_item(item, row_index)
                item.setData(Qt.UserRole, entry["path"])
                self.zarr_batch_table.setItem(row_index, column_index, item)
        self.zarr_batch_table.resizeColumnsToContents()
        self._updating_zarr_table = False

    def _update_nd2_batch_row(self, source_path: str, status: str, output_rel: str, error_text: str):
        for row_index, entry in enumerate(self._nd2_batch_entries):
            if entry["path"] != source_path:
                continue
            entry["status"] = status
            if output_rel:
                entry["output"] = output_rel
            entry["error"] = error_text
            for column_name, value in (
                ("status", entry["status"]),
                ("output", entry["output"]),
                ("error", entry["error"]),
            ):
                column_index = self.ND2_BATCH_COLUMNS.index(column_name)
                item = self.zarr_batch_table.item(row_index, column_index)
                if item is None:
                    item = QTableWidgetItem("")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.zarr_batch_table.setItem(row_index, column_index, item)
                item.setText(str(value))
            break

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

    def _style_zarr_batch_item(
        self,
        item: QTableWidgetItem,
        row_index: int,
        *,
        background_color: QColor | None = None,
        foreground_color: QColor | None = None,
    ):
        del row_index
        palette = self.zarr_batch_table.palette()
        item.setForeground(foreground_color or palette.color(QPalette.Text))
        item.setBackground(background_color or palette.color(QPalette.Base))

    def _selected_table_rows(self) -> list[int]:
        selection_model = self.zarr_batch_table.selectionModel()
        if selection_model is None:
            return []
        return sorted({index.row() for index in selection_model.selectedRows()})

    def _sync_selected_rows_to_zarr_checks(self):
        if self._batch_table_mode != "zarr" or self._updating_zarr_table:
            return
        selected_rows = set(self._selected_table_rows())
        for row_index, entry in enumerate(self._zarr_batch_entries):
            should_check = row_index in selected_rows
            entry["open"] = should_check
            item = self.zarr_batch_table.item(row_index, 0)
            if item is not None:
                item.setCheckState(Qt.Checked if should_check else Qt.Unchecked)

    def _toggle_selected_zarr_rows(self):
        if self._batch_table_mode != "zarr" or self._updating_zarr_table:
            return
        selected_rows = self._selected_table_rows()
        if not selected_rows:
            return
        any_unchecked = any(not self._zarr_batch_entries[row_index]["open"] for row_index in selected_rows)
        for row_index in selected_rows:
            self._zarr_batch_entries[row_index]["open"] = any_unchecked
            item = self.zarr_batch_table.item(row_index, 0)
            if item is not None:
                item.setCheckState(Qt.Checked if any_unchecked else Qt.Unchecked)

    def _set_all_zarr_rows_checked(self, checked: bool):
        if self._batch_table_mode != "zarr":
            return
        if not self._zarr_batch_entries:
            self._set_status("No Zarr datasets loaded into the batch table.")
            return
        for entry in self._zarr_batch_entries:
            entry["open"] = checked
        self._populate_zarr_batch_table()
        self._set_status(
            f"{'Selected' if checked else 'Cleared'} {len(self._zarr_batch_entries)} Zarr dataset(s) in the batch table."
        )

    def _open_selected_zarr_batch(self):
        if self._batch_table_mode != "zarr":
            self._set_status("Scan OME-Zarr sources first to open selected datasets.")
            return
        selected_entries = []
        for row_index, entry in enumerate(self._zarr_batch_entries):
            item = self.zarr_batch_table.item(row_index, 0)
            is_checked = item is not None and item.checkState() == Qt.Checked
            entry["open"] = is_checked
            if is_checked:
                selected_entries.append(entry)
        if not selected_entries:
            self._set_status("Select at least one Zarr dataset in the batch table.")
            return
        if not (self.zarr_gray_checkbox.isChecked() or self.zarr_truecolor_checkbox.isChecked() or self.zarr_raw_checkbox.isChecked()):
            self._set_status("Select at least one Zarr view to open.")
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
                self._set_status(f"Failed to open {Path(entry['path']).name}: {exc}")
                return
        self._set_status(f"Opened {opened_count} selected Zarr dataset(s).")

    def _use_gpu(self) -> bool:
        return self.gpu_checkbox.isEnabled() and self.gpu_checkbox.isChecked()

    def _reset_export_feedback(self, *, total: int):
        self.export_progress_bar.setRange(0, max(1, total))
        self.export_progress_bar.setValue(0)
        self.export_progress_bar.setFormat(f"0/{total}")
        self.export_error_log.clear()

    def _update_export_progress(self, completed: int, total: int):
        self.export_progress_bar.setRange(0, max(1, total))
        self.export_progress_bar.setValue(completed)
        self.export_progress_bar.setFormat(f"{completed}/{total}")

    def _append_export_failure_lines(self, failures: list[dict]):
        if not failures:
            return
        lines = [f"{Path(str(item.get('source_path', 'unknown'))).name}: {item.get('error', 'unknown error')}" for item in failures]
        self.export_error_log.setPlainText("\n".join(lines))

    def _export_ome_zarr_batch(self):
        if self._batch_thread is not None and self._batch_thread.isRunning():
            self._set_status("Batch export is already running.")
            return

        input_path = self.batch_input_box.text().strip()
        output_dir = self.output_dir_box.text().strip()
        if not input_path:
            self._set_status("Choose or drop an ND2 source first.")
            return
        if not output_dir:
            self._set_status("Choose or drop an OME-Zarr output folder.")
            return

        try:
            _input_root, nd2_paths = _resolve_nd2_batch_source(input_path)
        except Exception as exc:
            self._set_status(str(exc))
            return
        try:
            max_workers = max(1, int(self.worker_count_edit.text().strip()))
        except ValueError:
            self._set_status("Worker count must be an integer.")
            return

        self._reset_export_feedback(total=len(nd2_paths))
        if self._batch_table_mode != "nd2" or len(self._nd2_batch_entries) != len(nd2_paths):
            self._scan_nd2_batch_source(input_path)
        for entry in self._nd2_batch_entries:
            entry["status"] = "Queued"
            entry["error"] = ""
        self._populate_nd2_batch_table()
        self._set_batch_controls_enabled(False)
        self._batch_thread = QThread(self)
        self._batch_worker = BatchExportWorker(input_path=input_path, output_dir=output_dir, max_workers=max_workers)
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._set_status)
        self._batch_worker.progress_counts.connect(self._update_export_progress)
        self._batch_worker.file_result.connect(self._update_nd2_batch_row)
        self._batch_worker.finished.connect(self._on_batch_finished)
        self._batch_worker.failed.connect(self._on_batch_failed)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.failed.connect(self._batch_thread.quit)
        self._batch_thread.finished.connect(self._cleanup_batch_thread)
        self._batch_thread.start()
        self._set_status(
            f"Queued conversion for {len(nd2_paths)} ND2 file(s). Original folder structure will be preserved."
        )

    def _set_batch_controls_enabled(self, enabled: bool):
        self.batch_input_box.setEnabled(enabled)
        self.output_dir_box.setEnabled(enabled)
        self.worker_count_edit.setEnabled(enabled)

    def _on_batch_finished(self, exported_count: int, output_root: str, manifest_path: str, failures: object):
        failure_list = list(failures) if isinstance(failures, list) else []
        self._append_export_failure_lines(failure_list)
        self.export_progress_bar.setValue(self.export_progress_bar.maximum())
        self.export_progress_bar.setFormat(f"{self.export_progress_bar.maximum()}/{self.export_progress_bar.maximum()}")
        if failure_list:
            self._set_status(
                f"Batch exported {exported_count} file(s) to {output_root}. {len(failure_list)} file(s) failed. Saved manifest: {Path(manifest_path).name}"
            )
        else:
            self._set_status(
                f"Batch exported {exported_count} file(s) to {output_root}. Saved manifest: {Path(manifest_path).name}"
            )
        self._set_batch_controls_enabled(True)

    def _on_batch_failed(self, error_text: str):
        self.export_progress_bar.setFormat("Failed")
        self._set_status(f"Batch export failed: {error_text}")
        if not self.export_error_log.toPlainText().strip():
            self.export_error_log.setPlainText(error_text)
        self._set_batch_controls_enabled(True)

    def _cleanup_batch_thread(self):
        if self._batch_worker is not None:
            self._batch_worker.deleteLater()
        if self._batch_thread is not None:
            self._batch_thread.deleteLater()
        self._batch_worker = None
        self._batch_thread = None
