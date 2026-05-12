import numpy as np
from collections import deque
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox,
)


class ChartPanel(QWidget):
    def __init__(self, buffer_size: int = 500):
        super().__init__()
        self.buffer_size = buffer_size

        self._gyro_x = deque(maxlen=buffer_size)
        self._gyro_y = deque(maxlen=buffer_size)
        self._gyro_z = deque(maxlen=buffer_size)
        self._motor_0 = deque(maxlen=buffer_size)
        self._motor_1 = deque(maxlen=buffer_size)
        self._motor_2 = deque(maxlen=buffer_size)
        self._motor_3 = deque(maxlen=buffer_size)
        self._time = deque(maxlen=buffer_size)
        self._t = 0

        self._fft_buffers = {
            "gyro_x": deque(maxlen=buffer_size),
            "gyro_y": deque(maxlen=buffer_size),
        }

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title_bar = QHBoxLayout()
        title = QLabel("实时数据监测")
        title.setProperty("class", "header")
        title_bar.addWidget(title)
        title_bar.addStretch()

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setCheckable(True)
        title_bar.addWidget(self.pause_btn)
        layout.addLayout(title_bar)

        self.main_widget = pg.GraphicsLayoutWidget()

        self.gyro_plot = self.main_widget.addPlot(row=0, col=0, rowspan=1, colspan=1)
        self.gyro_plot.setLabel("left", "角速率 (°/s)")
        self.gyro_plot.setLabel("bottom", "时间 (s)")
        self.gyro_plot.showGrid(x=True, y=True, alpha=0.3)
        self.gyro_plot.addLegend()

        colors = {"Roll": (255, 80, 80), "Pitch": (80, 255, 80), "Yaw": (80, 80, 255)}
        self._gyro_curves = {}
        for axis_name, color in colors.items():
            pen = pg.mkPen(color=color, width=1.5)
            self._gyro_curves[axis_name] = self.gyro_plot.plot(
                [], [], pen=pen, name=axis_name
            )

        self.motor_plot = self.main_widget.addPlot(row=0, col=1, rowspan=1, colspan=1)
        self.motor_plot.setLabel("left", "电机输出")
        self.motor_plot.setLabel("bottom", "时间 (s)")
        self.motor_plot.showGrid(x=True, y=True, alpha=0.3)
        self.motor_plot.addLegend()

        motor_colors = {0: (255, 140, 0), 1: (0, 200, 200), 2: (255, 0, 255), 3: (255, 255, 0)}
        self._motor_curves = {}
        for motor_idx, color in motor_colors.items():
            pen = pg.mkPen(color=color, width=1.5)
            self._motor_curves[f"M{motor_idx + 1}"] = self.motor_plot.plot(
                [], [], pen=pen, name=f"M{motor_idx + 1}"
            )

        self.fft_plot = self.main_widget.addPlot(row=1, col=0, rowspan=1, colspan=2)
        self.fft_plot.setLabel("left", "幅度")
        self.fft_plot.setLabel("bottom", "频率 (Hz)")
        self.fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_plot.setLogMode(x=False, y=False)
        self.fft_plot.addLegend()

        self._fft_curves = {}
        for axis_name in ("Roll", "Pitch"):
            color = (255, 80, 80) if axis_name == "Roll" else (80, 255, 80)
            pen = pg.mkPen(color=color, width=2)
            self._fft_curves[axis_name] = self.fft_plot.plot(
                [], [], pen=pen, name=f"{axis_name} FFT"
            )

        layout.addWidget(self.main_widget, stretch=1)

    def update_data(self, data: dict):
        if self.pause_btn.isChecked():
            return

        self._t += 1

        self._gyro_x.append(data.get("gyro_x", 0))
        self._gyro_y.append(data.get("gyro_y", 0))
        self._gyro_z.append(data.get("gyro_z", 0))

        self._motor_0.append(data.get("motor_0", 0))
        self._motor_1.append(data.get("motor_1", 0))
        self._motor_2.append(data.get("motor_2", 0))
        self._motor_3.append(data.get("motor_3", 0))

        t_val = self._t / 1000.0
        self._time.append(t_val)

        if self._t % 5 == 0:
            t_arr = list(self._time)
            if len(t_arr) > 1:
                self._gyro_curves["Roll"].setData(t_arr, list(self._gyro_x))
                self._gyro_curves["Pitch"].setData(t_arr, list(self._gyro_y))
                self._gyro_curves["Yaw"].setData(t_arr, list(self._gyro_z))

                self._motor_curves["M1"].setData(t_arr, list(self._motor_0))
                self._motor_curves["M2"].setData(t_arr, list(self._motor_1))
                self._motor_curves["M3"].setData(t_arr, list(self._motor_2))
                self._motor_curves["M4"].setData(t_arr, list(self._motor_3))

        if self._t % 20 == 0:
            self._update_fft()

    def _update_fft(self):
        gyro_data = {
            "Roll": list(self._gyro_x),
            "Pitch": list(self._gyro_y),
        }

        sample_rate = 1000.0

        for axis_name, data in gyro_data.items():
            if len(data) < 50:
                continue

            arr = np.array(data)
            arr = arr - np.mean(arr)

            n = len(arr)
            freq = np.fft.rfftfreq(n, d=1.0 / sample_rate)
            spectrum = np.abs(np.fft.rfft(arr))

            mask = freq <= 500
            self._fft_curves[axis_name].setData(freq[mask], spectrum[mask])

    def clear(self):
        self._gyro_x.clear()
        self._gyro_y.clear()
        self._gyro_z.clear()
        self._motor_0.clear()
        self._motor_1.clear()
        self._motor_2.clear()
        self._motor_3.clear()
        self._time.clear()
        self._t = 0