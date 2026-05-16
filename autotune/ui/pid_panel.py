from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QMessageBox, QAbstractItemView,
)


class PIDPanel(QWidget):
    COLUMNS = ["轴", "P", "I", "D", "FF", "D_Min", "D_Min_Gain", "D_Min_Advance", "D_Gain_Boost"]
    PARAM_KEYS = ["P", "I", "D", "FF", "D_Min", "D_Min_Gain", "D_Min_Advance", "D_Gain_Boost"]

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("PID 参数配置")
        title.setProperty("class", "header")
        layout.addWidget(title)

        btn_layout = QHBoxLayout()
        self.read_btn = QPushButton("从飞控读取")
        self.read_btn.clicked.connect(self.load_from_controller)
        btn_layout.addWidget(self.read_btn)

        self.read_adv_btn = QPushButton("读取高级PID")
        self.read_adv_btn.clicked.connect(self._read_advanced)
        btn_layout.addWidget(self.read_adv_btn)

        self.write_btn = QPushButton("写入飞控")
        self.write_btn.setProperty("class", "danger")
        self.write_btn.clicked.connect(self._write_to_fc)
        btn_layout.addWidget(self.write_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        table_group = QGroupBox("PID 参数")
        table_layout = QVBoxLayout(table_group)

        num_cols = len(self.COLUMNS)
        self.table = QTableWidget()
        self.table.setColumnCount(num_cols)
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setRowCount(3)

        axis_names = ["Roll", "Pitch", "Yaw"]
        for i, name in enumerate(axis_names):
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, item)

            for col in range(1, num_cols):
                spin_item = QTableWidgetItem("0.0")
                spin_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, spin_item)

        table_layout.addWidget(self.table)
        layout.addWidget(table_group)

        layout.addStretch()

    def load_from_controller(self):
        profile = self.controller.pid_profile
        self._populate_table(profile)

    def _read_advanced(self):
        try:
            profile = self.controller.read_pid_profile_advanced()
            self._populate_table(profile)
        except Exception as e:
            QMessageBox.warning(self, "读取失败", f"高级 PID 读取失败: {e}")

    def load_from_dict(self, profile_dict: dict):
        from autotune.fc.pid import PIDProfile
        profile = PIDProfile.from_dict(profile_dict)
        self._populate_table(profile)

    def _populate_table(self, profile):
        for i in range(3):
            axis = profile.get_axis(i)
            axis_adv = profile.get_axis_advanced(i)

            values = [
                axis.p, axis.i, axis.d,
                axis_adv.ff_gain, axis_adv.d_min,
                axis_adv.d_min_gain, axis_adv.d_min_advance, axis_adv.d_gain_boost,
            ]

            for col, val in enumerate(values):
                self.table.item(i, col + 1).setText(f"{val:.0f}" if col < 3 else f"{val:.0f}")

    def _collect_from_table(self) -> dict:
        data = {}
        axes = ["Roll", "Pitch", "Yaw"]
        for i, axis in enumerate(axes):
            axis_data = {}
            for col, key in enumerate(self.PARAM_KEYS):
                text = self.table.item(i, col + 1).text() or "0"
                axis_data[key] = float(text)
            data[axis] = axis_data
            data[f"{axis}_Advanced"] = {
                "FF": axis_data["FF"],
                "D_Min": axis_data["D_Min"],
                "D_Min_Gain": axis_data["D_Min_Gain"],
                "D_Min_Advance": axis_data["D_Min_Advance"],
                "D_Gain_Boost": axis_data["D_Gain_Boost"],
            }
        return data

    def _write_to_fc(self):
        reply = QMessageBox.question(
            self, "确认写入",
            "确定要将当前显示的 PID 参数写入飞控吗？\n"
            "建议先备份当前配置！",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            table_data = self._collect_from_table()

            from autotune.fc.pid import PIDProfile, PIDAxis, PIDAdvancedAxis
            profile = PIDProfile()
            profile.use_advanced = True

            for i, axis_name in enumerate(["Roll", "Pitch", "Yaw"]):
                axis_data = table_data[axis_name]
                axis = profile.get_axis(i)
                axis.p = axis_data["P"]
                axis.i = axis_data["I"]
                axis.d = axis_data["D"]

                axis_adv = profile.get_axis_advanced(i)
                axis_adv.ff_gain = axis_data["FF"]
                axis_adv.d_min = axis_data["D_Min"]
                axis_adv.d_min_gain = axis_data["D_Min_Gain"]
                axis_adv.d_min_advance = axis_data["D_Min_Advance"]
                axis_adv.d_gain_boost = axis_data["D_Gain_Boost"]

            self.controller.write_pid_profile(profile)
            QMessageBox.information(self, "成功", "PID 参数已写入飞控！")
        except Exception as e:
            QMessageBox.critical(self, "写入失败", str(e))

    def get_pid_values(self) -> dict:
        return self._collect_from_table()