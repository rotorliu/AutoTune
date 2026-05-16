from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QMessageBox, QFormLayout,
    QDoubleSpinBox, QSpinBox, QComboBox,
)


class FilterPanel(QWidget):

    LOWPASS_TYPES = ["PT1", "BIQUAD", "PT2", "PT3"]

    FILTER_FIELDS = {
        "gyro": [
            ("gyro_lowpass_hz", "低通 1 截止频率 (Hz)", 30.0, 1000.0, 1.0, 0, "dyn_min_hz"),
            ("gyro_lowpass2_hz", "低通 2 截止频率 (Hz)", 50.0, 1000.0, 1.0, 0, "dyn_max_hz"),
            ("gyro_lowpass_type", "低通 1 类型", 0, 0, 0, 0, None),
            ("gyro_lowpass2_type", "低通 2 类型", 0, 0, 0, 0, None),
        ],
        "dterm": [
            ("dterm_lowpass_hz", "低通 1 截止频率 (Hz)", 30.0, 500.0, 1.0, 0, "dyn_min_hz"),
            ("dterm_lowpass2_hz", "低通 2 截止频率 (Hz)", 50.0, 500.0, 1.0, 0, "dyn_max_hz"),
            ("dterm_lowpass_type", "低通 1 类型", 0, 0, 0, 0, None),
            ("dterm_lowpass2_type", "低通 2 类型", 0, 0, 0, 0, None),
        ],
        "notch": [
            ("gyro_notch_hz", "陀螺仪陷波中心 (Hz) [0=关闭]", 0.0, 1000.0, 1.0, 1, None),
            ("gyro_notch_cutoff", "陀螺仪陷波截止 (Hz) [0=关闭]", 0.0, 1000.0, 1.0, 1, None),
            ("dterm_notch_hz", "D-Term陷波中心 (Hz) [0=关闭]", 0.0, 1000.0, 1.0, 1, None),
            ("dterm_notch_cutoff", "D-Term陷波截止 (Hz) [0=关闭]", 0.0, 1000.0, 1.0, 1, None),
        ],
        "yaw": [
            ("yaw_lowpass_hz", "Yaw 低通截止频率 (Hz)", 0.0, 500.0, 1.0, 0, None),
        ],
    }

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._widgets: dict[str, QWidget] = {}
        self._type_combos: dict[str, QComboBox] = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("滤波器配置")
        title.setProperty("class", "header")
        layout.addWidget(title)

        btn_layout = QHBoxLayout()
        self.read_btn = QPushButton("从飞控读取")
        self.read_btn.clicked.connect(self.load_from_controller)
        btn_layout.addWidget(self.read_btn)

        self.write_btn = QPushButton("写入飞控")
        self.write_btn.setProperty("class", "danger")
        self.write_btn.clicked.connect(self._write_to_fc)
        btn_layout.addWidget(self.write_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        cols_layout = QHBoxLayout()

        gyro_group = QGroupBox("陀螺仪低通滤波")
        gyro_layout = QFormLayout(gyro_group)
        self._build_fields(gyro_layout, self.FILTER_FIELDS["gyro"])
        cols_layout.addWidget(gyro_group)

        dterm_group = QGroupBox("D-Term 低通滤波")
        dterm_layout = QFormLayout(dterm_group)
        self._build_fields(dterm_layout, self.FILTER_FIELDS["dterm"])
        cols_layout.addWidget(dterm_group)

        layout.addLayout(cols_layout)

        notch_group = QGroupBox("陷波滤波器")
        notch_layout = QFormLayout(notch_group)
        self._build_fields(notch_layout, self.FILTER_FIELDS["notch"])
        layout.addWidget(notch_group)

        yaw_group = QGroupBox("Yaw 滤波")
        yaw_layout = QFormLayout(yaw_group)
        self._build_fields(yaw_layout, self.FILTER_FIELDS["yaw"])
        layout.addWidget(yaw_group)

        layout.addStretch()

    def _build_fields(self, form_layout: QFormLayout, fields: list):
        for key, label, min_v, max_v, step, decimals, _ in fields:
            if "type" in key:
                combo = QComboBox()
                combo.addItems(self.LOWPASS_TYPES)
                combo.setMinimumWidth(120)
                form_layout.addRow(QLabel(label), combo)
                self._type_combos[key] = combo
            else:
                if decimals == 0:
                    widget = QSpinBox()
                    widget.setRange(int(min_v), int(max_v))
                    widget.setSingleStep(int(step))
                else:
                    widget = QDoubleSpinBox()
                    widget.setRange(min_v, max_v)
                    widget.setSingleStep(step)
                    widget.setDecimals(decimals)
                widget.setMinimumWidth(120)
                form_layout.addRow(QLabel(label), widget)
                self._widgets[key] = widget

    def load_from_controller(self):
        config = self.controller.config.filter_config
        self._populate(config)

    def load_from_dict(self, config_dict: dict):
        from autotune.fc.config import FilterConfig
        config = FilterConfig.from_dict(config_dict)
        self._populate(config)

    def _populate(self, config):
        for key, widget in self._widgets.items():
            val = getattr(config, key, 0)
            if isinstance(widget, QSpinBox):
                widget.setValue(int(val))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(val))

        for key, combo in self._type_combos.items():
            type_val = int(getattr(config, key, 0))
            if 0 <= type_val < len(self.LOWPASS_TYPES):
                combo.setCurrentIndex(type_val)

    def _write_to_fc(self):
        reply = QMessageBox.question(
            self, "确认写入",
            "确定要将当前显示的滤波器配置写入飞控吗？",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            from autotune.fc.config import FilterConfig
            config = FilterConfig()

            for key, widget in self._widgets.items():
                val = widget.value()
                setattr(config, key, val)

            for key, combo in self._type_combos.items():
                setattr(config, key, combo.currentIndex())

            self.controller.write_filter_config(config)
            QMessageBox.information(self, "成功", "滤波器配置已写入飞控！")
        except Exception as e:
            QMessageBox.critical(self, "写入失败", str(e))