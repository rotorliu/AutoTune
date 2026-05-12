from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QMessageBox, QAbstractItemView,
)


class PIDPanel(QWidget):
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

        self.write_btn = QPushButton("写入飞控")
        self.write_btn.setProperty("class", "danger")
        self.write_btn.clicked.connect(self._write_to_fc)
        btn_layout.addWidget(self.write_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        table_group = QGroupBox("PID 参数")
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["轴", "P", "I", "D", "D_Min", "FF"])
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

            for col in range(1, 6):
                spin_item = QTableWidgetItem("0.0")
                spin_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, spin_item)

        table_layout.addWidget(self.table)
        layout.addWidget(table_group)

        layout.addStretch()

    def load_from_controller(self):
        profile = self.controller.pid_profile
        self._populate_table(profile)

    def load_from_dict(self, profile_dict: dict):
        from autotune.fc.pid import PIDProfile
        profile = PIDProfile.from_dict(profile_dict)
        self._populate_table(profile)

    def _populate_table(self, profile):
        for i, axis_name in enumerate(["Roll", "Pitch", "Yaw"]):
            axis = profile.get_axis(i)
            self.table.item(i, 1).setText(f"{axis.p:.0f}")
            self.table.item(i, 2).setText(f"{axis.i:.0f}")
            self.table.item(i, 3).setText(f"{axis.d:.0f}")

            axis_adv = profile.get_axis_advanced(i)
            self.table.item(i, 4).setText(f"{axis_adv.d_min:.0f}")
            self.table.item(i, 5).setText(f"{axis_adv.ff_gain:.0f}")

    def _collect_from_table(self) -> dict:
        data = {"Roll": {}, "Pitch": {}, "Yaw": {}, "Yaw_Advanced": {}}
        axes = ["Roll", "Pitch", "Yaw"]
        for i, axis in enumerate(axes):
            data[axis] = {
                "P": float(self.table.item(i, 1).text() or "0"),
                "I": float(self.table.item(i, 2).text() or "0"),
                "D": float(self.table.item(i, 3).text() or "0"),
            }
            data[f"{axis}_Advanced"] = {
                "D_Min": float(self.table.item(i, 4).text() or "0"),
                "FF": float(self.table.item(i, 5).text() or "0"),
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

            for i, axis_name in enumerate(["Roll", "Pitch", "Yaw"]):
                axis_data = table_data[axis_name]
                axis = profile.get_axis(i)
                axis.p = axis_data["P"]
                axis.i = axis_data["I"]
                axis.d = axis_data["D"]

            self.controller.write_pid_profile(profile)
            QMessageBox.information(self, "成功", "PID 参数已写入飞控！")
        except Exception as e:
            QMessageBox.critical(self, "写入失败", str(e))

    def get_pid_values(self) -> dict:
        return self._collect_from_table()