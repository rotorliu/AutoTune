from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
)
from autotune.ui.main_window import MainWindow
from autotune.utils.logger import setup_logger


class AutoTuneApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = setup_logger()
        self.logger.info("AutoTune application starting")

        self.setWindowTitle("AutoTune - Betaflight PID/Rate Auto Tuning Tool")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.main_window = MainWindow()
        layout.addWidget(self.main_window)

        self.main_window.status_message.connect(self._on_status_message)

    def _on_status_message(self, message):
        self.statusBar().showMessage(message)

    def closeEvent(self, event):
        self.main_window.cleanup()
        self.logger.info("AutoTune application closing")
        event.accept()