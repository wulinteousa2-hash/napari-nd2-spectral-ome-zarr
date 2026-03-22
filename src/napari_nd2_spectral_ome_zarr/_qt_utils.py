from __future__ import annotations

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QDockWidget, QWidget


def float_parent_dock_later(widget: QWidget, attempts: int = 10, delay_ms: int = 50) -> None:
    def _try_float(remaining: int):
        dock_widget = _find_parent_dock(widget)
        if dock_widget is not None:
            dock_widget.setFloating(True)
            dock_widget.raise_()
            return
        if remaining > 0:
            QTimer.singleShot(delay_ms, lambda: _try_float(remaining - 1))

    QTimer.singleShot(0, lambda: _try_float(attempts))


def _find_parent_dock(widget: QWidget) -> QDockWidget | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QDockWidget):
            return parent
        parent = parent.parentWidget()
    return None
