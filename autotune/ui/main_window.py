from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QTabWidget, QTextEdit,
)
from PySide6.QtGui import QFont

from autotune.msp.transport import MSPTransport
from autotune.fc.controller import FCController
from autotune.ui.connection_panel import ConnectionPanel
from autotune.ui.pid_panel import PIDPanel
from autotune.ui.rate_panel import RatePanel
from autotune.ui.chart_panel import ChartPanel
from autotune.ui.tuning_wizard import TuningWizard
from autotune.ui.profile_panel import ProfilePanel
from autotune.utils.logger import setup_logger


class MainWindow(QWidget):
    status_message = Signal(str)

    def __init__(self):
        super().__init__()
        self.logger = setup_logger("autotune.ui")
        self.transport = MSPTransport()
        self.controller = FCController(self.transport)
        self._telemetry_timer: QTimer | None = None
        self._telemetry_buffer: list[dict] = []

        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tab_widget = QTabWidget()

        self.connection_panel = ConnectionPanel(self.transport, self.controller)
        self.connection_panel.connection_changed.connect(self._on_connection_changed)
        self.connection_panel.status_message.connect(self.status_message.emit)
        self.tab_widget.addTab(self.connection_panel, "连接")

        self.pid_panel = PIDPanel(self.controller)
        self.tab_widget.addTab(self.pid_panel, "PID 配置")

        self.rate_panel = RatePanel(self.controller)
        self.tab_widget.addTab(self.rate_panel, "Rate 配置")

        self.chart_panel = ChartPanel()
        self.tab_widget.addTab(self.chart_panel, "实时数据")

        self.tuning_wizard = TuningWizard(self.controller)
        self.tuning_wizard.status_message.connect(self.status_message.emit)
        self.tuning_wizard.pid_updated.connect(self._on_pid_updated)
        self.tuning_wizard.rate_updated.connect(self._on_rate_updated)
        self.tab_widget.addTab(self.tuning_wizard, "自动调参")

        self.profile_panel = ProfilePanel(self.controller)
        self.profile_panel.pid_load_request.connect(self._on_pid_updated)
        self.profile_panel.rate_load_request.connect(self._on_rate_updated)
        self.tab_widget.addTab(self.profile_panel, "方案管理")

        layout.addWidget(self.tab_widget)

    def _apply_style(self):
        from autotune.ui.styles import load_stylesheet
        stylesheet = load_stylesheet()
        if stylesheet:
            self.setStyleSheet(stylesheet)

    def _on_connection_changed(self, connected: bool):
        if connected:
            self.controller.identify()
            self.controller.read_all()
            self.pid_panel.load_from_controller()
            self.rate_panel.load_from_controller()
            self._start_telemetry()
        else:
            self._stop_telemetry()

    def _start_telemetry(self):
        if self._telemetry_timer is None:
            self._telemetry_timer = QTimer()
            self._telemetry_timer.timeout.connect(self._poll_telemetry)
            self._telemetry_timer.start(50)

    def _stop_telemetry(self):
        if self._telemetry_timer is not None:
            self._telemetry_timer.stop()
            self._telemetry_timer = None

    def _poll_telemetry(self):
        if not self.transport.is_connected:
            return
        try:
            data = self.controller.read_telemetry_snapshot()
            self._telemetry_buffer.append(data)
            if len(self._telemetry_buffer) > 10000:
                self._telemetry_buffer = self._telemetry_buffer[-5000:]

            self.chart_panel.update_data(data)
        except Exception:
            pass

    def _on_pid_updated(self, profile_dict: dict):
        self.pid_panel.load_from_dict(profile_dict)

    def _on_rate_updated(self, profile_dict: dict):
        self.rate_panel.load_from_dict(profile_dict)

    def cleanup(self):
        self._stop_telemetry()
        if self.transport.is_connected:
            self.transport.disconnect()

    def get_telemetry_buffer(self) -> list[dict]:
        return list(self._telemetry_buffer)

    def clear_telemetry_buffer(self):
        self._telemetry_buffer.clear()