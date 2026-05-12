from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QMessageBox, QAbstractItemView,
)
import numpy as np
import pyqtgraph as pg


class RatePanel(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Rate 参数配置")
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

        table_group = QGroupBox("Rate 参数")
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["轴", "RC Rate", "Super Rate", "RC Expo"])
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

            for col in range(1, 4):
                spin_item = QTableWidgetItem("0.0")
                spin_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, col, spin_item)

        table_layout.addWidget(self.table)
        layout.addWidget(table_group)

        chart_group = QGroupBox("Rate 曲线预览")
        chart_layout = QVBoxLayout(chart_group)

        self.curve_widget = pg.GraphicsLayoutWidget()
        self.curve_plot = self.curve_widget.addPlot()
        self.curve_plot.setLabel("left", "角速率 (°/s)")
        self.curve_plot.setLabel("bottom", "摇杆输入 (%)")
        self.curve_plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve_plot.setMinimumHeight(250)

        self.curve_lines = {}
        colors = {"Roll": (255, 100, 100), "Pitch": (100, 255, 100), "Yaw": (100, 100, 255)}
        for axis_name, color in colors.items():
            pen = pg.mkPen(color=color, width=2)
            self.curve_lines[axis_name] = self.curve_plot.plot(
                [], [], pen=pen, name=axis_name
            )

        legend = self.curve_plot.addLegend()
        self.curve_plot.setLimits(xMin=-105, xMax=105)

        chart_layout.addWidget(self.curve_widget)
        layout.addWidget(chart_group)

    def load_from_controller(self):
        profile = self.controller.rate_profile
        self._populate_table(profile)
        self._update_curves(profile)

    def load_from_dict(self, profile_dict: dict):
        from autotune.fc.rate import RateProfile
        profile = RateProfile.from_dict(profile_dict)
        self._populate_table(profile)
        self._update_curves(profile)

    def _populate_table(self, profile):
        for i in range(3):
            axis = profile.get_axis(i)
            self.table.item(i, 1).setText(f"{axis.rc_rate:.2f}")
            self.table.item(i, 2).setText(f"{axis.super_rate:.3f}")
            self.table.item(i, 3).setText(f"{axis.rc_expo:.2f}")

    def _update_curves(self, profile):
        rc_inputs = np.linspace(-100, 100, 200) / 100.0
        for i, axis_name in enumerate(["Roll", "Pitch", "Yaw"]):
            axis = profile.get_axis(i)
            rates = np.array([axis.compute_angular_rate(ri) for ri in rc_inputs])
            self.curve_lines[axis_name].setData(rc_inputs * 100, rates)

    def _write_to_fc(self):
        reply = QMessageBox.question(
            self, "确认写入",
            "确定要将当前显示的 Rate 参数写入飞控吗？",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            from autotune.fc.rate import RateProfile, RateAxis

            profile = RateProfile()
            for i in range(3):
                axis = profile.get_axis(i)
                axis.rc_rate = float(self.table.item(i, 1).text() or "1.0")
                axis.super_rate = float(self.table.item(i, 2).text() or "0.7")
                axis.rc_expo = float(self.table.item(i, 3).text() or "0.0")

            self.controller.write_rate_profile(profile)
            QMessageBox.information(self, "成功", "Rate 参数已写入飞控！")
        except Exception as e:
            QMessageBox.critical(self, "写入失败", str(e))