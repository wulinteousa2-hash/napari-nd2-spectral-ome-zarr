from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFontMetrics, QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
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

from ._qt_utils import float_parent_dock_later
from ._reader import build_layer_data, inspect_ome_zarr
from ._spectral import get_gpu_status_text, gpu_available


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


class SelectableTableWidget(QTableWidget):
    toggle_rows_requested = Signal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_rows_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class OmeZarrBrowserWidget(QWidget):
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
        self._zarr_batch_entries: list[dict] = []
        self._updating_zarr_table = False
        self._status_message = "Ready."

        self.zarr_scan_box = DropPathBox(
            "OME-Zarr source",
            "Drag a folder to scan all nested .zarr datasets, or drag one .zarr folder to open just that dataset.",
            "Drop .zarr folder or parent folder here",
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
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Find name, path, date")
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.gpu_checkbox = QCheckBox("GPU")
        self.gpu_checkbox.setChecked(gpu_available())
        self.gpu_checkbox.setEnabled(gpu_available())
        self.gpu_checkbox.stateChanged.connect(self._update_gpu_indicator)
        self.gpu_indicator = QLabel()
        self.gpu_indicator.setAlignment(Qt.AlignCenter)
        self.zarr_gray_checkbox = QCheckBox("Visible sum")
        self.zarr_truecolor_checkbox = QCheckBox("Truecolor")
        self.zarr_truecolor_checkbox.setChecked(True)
        self.truecolor_auto_clean_checkbox = QCheckBox("Clean bg")
        self.truecolor_auto_clean_checkbox.setChecked(False)
        self.truecolor_clean_combo = QComboBox()
        self.truecolor_clean_combo.addItems(["Low", "Med", "High"])
        self.truecolor_clean_combo.setCurrentText("Med")
        self.zarr_raw_checkbox = QCheckBox("Raw spectral")
        self.zarr_preview_checkbox = QCheckBox("Preview")
        self.zarr_preview_checkbox.setChecked(True)
        self.zarr_batch_table = SelectableTableWidget()
        self.zarr_batch_table.setAlternatingRowColors(False)
        self.zarr_batch_table.setMinimumHeight(360)
        self.zarr_batch_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
        self._configure_zarr_batch_table_palette()
        self._update_gpu_indicator()
        self.zarr_scan_box.pathDropped.connect(self._handle_zarr_scan_dropped)

        browse_zarr_batch_root = QPushButton("Browse")
        browse_zarr_batch_root.clicked.connect(self._pick_zarr_batch_root)

        scan_zarr_button = QPushButton("Scan")
        scan_zarr_button.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; font-weight: 700; padding: 8px 12px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
        )
        scan_zarr_button.clicked.connect(self._scan_zarr_batch_root)

        self.select_all_zarr_button = QPushButton("Select Visible")
        self.select_all_zarr_button.clicked.connect(lambda: self._set_visible_zarr_rows_checked(True))

        self.clear_all_zarr_button = QPushButton("Clear Visible")
        self.clear_all_zarr_button.clicked.connect(lambda: self._set_visible_zarr_rows_checked(False))

        self.open_selected_zarr_button = QPushButton("Open Checked")
        self.open_selected_zarr_button.clicked.connect(self._open_selected_zarr_batch)

        layout = QVBoxLayout()
        browser_group = QGroupBox("OME-Zarr Browser")
        browser_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        browser_layout = QVBoxLayout()
        browser_layout.addWidget(self.status_label)

        zarr_source_row = QHBoxLayout()
        zarr_source_row.addWidget(self.zarr_scan_box, 1)
        zarr_source_row.addWidget(QLabel("or"))
        zarr_source_row.addWidget(browse_zarr_batch_root, 0)
        browser_layout.addLayout(zarr_source_row)

        browser_layout.addWidget(scan_zarr_button)
        browser_layout.addWidget(QLabel("Find"))
        browser_layout.addWidget(self.filter_edit)

        zarr_options_row = QHBoxLayout()
        zarr_options_row.addWidget(self.zarr_gray_checkbox)
        zarr_options_row.addWidget(self.zarr_truecolor_checkbox)
        zarr_options_row.addWidget(self.truecolor_auto_clean_checkbox)
        zarr_options_row.addWidget(QLabel("Clean"))
        zarr_options_row.addWidget(self.truecolor_clean_combo)
        zarr_options_row.addWidget(self.zarr_raw_checkbox)
        zarr_options_row.addWidget(self.zarr_preview_checkbox)
        browser_layout.addLayout(zarr_options_row)

        zarr_batch_actions = QHBoxLayout()
        zarr_batch_actions.addWidget(self.select_all_zarr_button)
        zarr_batch_actions.addWidget(self.clear_all_zarr_button)
        zarr_batch_actions.addWidget(self.open_selected_zarr_button)
        browser_layout.addLayout(zarr_batch_actions)
        browser_layout.addWidget(self.zarr_batch_table, 1)

        gpu_row = QHBoxLayout()
        gpu_row.addWidget(self.gpu_checkbox)
        gpu_row.addWidget(self.gpu_indicator)
        browser_layout.addLayout(gpu_row)

        browser_group.setLayout(browser_layout)
        layout.addWidget(browser_group, 1)
        self.setLayout(layout)
        self._set_status("Drop or browse an OME-Zarr source, then scan.")
        float_parent_dock_later(self, minimum_size=QSize(720, 560))

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
        self._apply_filter()

    def _populate_zarr_batch_table(self):
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
                self._style_zarr_batch_item(item, background_color=background_color, foreground_color=foreground_color)
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

    def _style_zarr_batch_item(
        self,
        item: QTableWidgetItem,
        *,
        background_color: QColor | None = None,
        foreground_color: QColor | None = None,
    ):
        palette = self.zarr_batch_table.palette()
        item.setForeground(foreground_color or palette.color(QPalette.Text))
        item.setBackground(background_color or palette.color(QPalette.Base))

    def _entry_matches_filter(self, entry: dict, query: str) -> bool:
        if not query:
            return True
        searchable_text = " ".join(str(entry.get(key, "")) for key in ("name", "relative_path", "path", "wavelengths", "shape", "preview_shape"))
        return query in searchable_text.lower()

    def _visible_zarr_row_indices(self) -> list[int]:
        return [row for row in range(len(self._zarr_batch_entries)) if not self.zarr_batch_table.isRowHidden(row)]

    def _apply_filter(self):
        query = self.filter_edit.text().strip().lower()
        visible_count = 0
        for row_index, entry in enumerate(self._zarr_batch_entries):
            is_match = self._entry_matches_filter(entry, query)
            self.zarr_batch_table.setRowHidden(row_index, not is_match)
            if is_match:
                visible_count += 1
        if not self._zarr_batch_entries:
            self._set_status("No Zarr datasets loaded into the browser table.")
        elif query:
            self._set_status(f"Matched {visible_count}/{len(self._zarr_batch_entries)} Zarr dataset(s) for '{self.filter_edit.text().strip()}'.")
        else:
            self._set_status(f"Found {len(self._zarr_batch_entries)} Zarr dataset(s).")

    def _selected_table_rows(self) -> list[int]:
        selection_model = self.zarr_batch_table.selectionModel()
        if selection_model is None:
            return []
        return sorted({index.row() for index in selection_model.selectedRows() if not self.zarr_batch_table.isRowHidden(index.row())})

    def _sync_selected_rows_to_zarr_checks(self):
        if self._updating_zarr_table:
            return
        selected_rows = set(self._selected_table_rows())
        if not selected_rows:
            return
        for row_index in self._visible_zarr_row_indices():
            should_check = row_index in selected_rows
            self._zarr_batch_entries[row_index]["open"] = should_check
            item = self.zarr_batch_table.item(row_index, 0)
            if item is not None:
                item.setCheckState(Qt.Checked if should_check else Qt.Unchecked)

    def _toggle_selected_zarr_rows(self):
        if self._updating_zarr_table:
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

    def _set_visible_zarr_rows_checked(self, checked: bool):
        visible_rows = self._visible_zarr_row_indices()
        if not visible_rows:
            self._set_status("No visible Zarr datasets to update.")
            return
        for row_index in visible_rows:
            self._zarr_batch_entries[row_index]["open"] = checked
            item = self.zarr_batch_table.item(row_index, 0)
            if item is not None:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self._set_status(
            f"{'Selected' if checked else 'Cleared'} {len(visible_rows)} visible Zarr dataset(s) in the browser table."
        )

    def _open_selected_zarr_batch(self):
        selected_entries = []
        for row_index in self._visible_zarr_row_indices():
            entry = self._zarr_batch_entries[row_index]
            item = self.zarr_batch_table.item(row_index, 0)
            is_checked = item is not None and item.checkState() == Qt.Checked
            entry["open"] = is_checked
            if is_checked:
                selected_entries.append(entry)
        if not selected_entries:
            self._set_status("Select at least one visible Zarr dataset in the browser table.")
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
                    truecolor_auto_clean_background=self.truecolor_auto_clean_checkbox.isChecked(),
                    truecolor_clean_strength=self.truecolor_clean_combo.currentText().lower(),
                ):
                    self.viewer.add_image(data, **kwargs) if layer_type == "image" else None
                opened_count += 1
            except Exception as exc:
                self._set_status(f"Failed to open {Path(entry['path']).name}: {exc}")
                return
        self._set_status(f"Opened {opened_count} visible Zarr dataset(s).")

    def _use_gpu(self) -> bool:
        return self.gpu_checkbox.isEnabled() and self.gpu_checkbox.isChecked()
