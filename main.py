import sys
from PySide6.QtWidgets import QApplication
from autotune.app import AutoTuneApp


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AutoTune")
    app.setOrganizationName("AutoTune")

    window = AutoTuneApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()