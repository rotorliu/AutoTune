import os

STYLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources")

DEFAULT_STYLE = os.path.join(STYLES_DIR, "styles.qss")


def load_stylesheet(filepath: str = DEFAULT_STYLE) -> str:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def apply_theme(app, filepath: str = DEFAULT_STYLE):
    stylesheet = load_stylesheet(filepath)
    if stylesheet:
        app.setStyleSheet(stylesheet)