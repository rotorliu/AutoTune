from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QTextEdit, QGridLayout,
    QMessageBox,
)
from PySide6.QtGui import QFont

from autotune.msp.transport import MSPTransport


class ConnectionPanel(QWidget):
    connection_changed = Signal(bool)
    status_message = Signal(str)

    def __init__(self, transport: MSPTransport, controller):
        super().__init__()
        self.transport = transport
        self.controller = controller

        self._init_ui()
        self._refresh_ports()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("飞控连接")
        title.setProperty("class", "header")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)

        conn_group = QGroupBox("串口连接")
        conn_layout = QGridLayout(conn_group)
        conn_layout.setSpacing(8)

        conn_layout.addWidget(QLabel("串口:"), 0, 0)
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(200)
        conn_layout.addWidget(self.port_combo, 0, 1)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._refresh_ports)
        conn_layout.addWidget(self.refresh_btn, 0, 2)

        conn_layout.addWidget(QLabel("波特率:"), 1, 0)
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(["115200", "500000", "921600", "1000000", "1500000"])
        self.baud_combo.setCurrentText("115200")
        conn_layout.addWidget(self.baud_combo, 1, 1)

        self.connect_btn = QPushButton("连接")
        self.connect_btn.clicked.connect(self._toggle_connection)
        conn_layout.addWidget(self.connect_btn, 1, 2)

        layout.addWidget(conn_group)

        info_group = QGroupBox("飞控信息")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(6)

        info_layout.addWidget(QLabel("型号:"), 0, 0)
        self.fc_variant_label = QLabel("--")
        info_layout.addWidget(self.fc_variant_label, 0, 1)

        info_layout.addWidget(QLabel("固件版本:"), 1, 0)
        self.fc_version_label = QLabel("--")
        info_layout.addWidget(self.fc_version_label, 1, 1)

        info_layout.addWidget(QLabel("目标:"), 2, 0)
        self.fc_target_label = QLabel("--")
        info_layout.addWidget(self.fc_target_label, 2, 1)

        info_layout.addWidget(QLabel("构建日期:"), 3, 0)
        self.fc_build_label = QLabel("--")
        info_layout.addWidget(self.fc_build_label, 3, 1)

        layout.addWidget(info_group)

        status_group = QGroupBox("连接日志")
        status_layout = QVBoxLayout(status_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        status_layout.addWidget(self.log_text)
        layout.addWidget(status_group)

        layout.addStretch()

    def _refresh_ports(self):
        self.port_combo.clear()
        ports = self.transport.list_ports()
        bf_ports = self.transport.filter_betaflight_ports(ports)

        for port in bf_ports:
            desc = port.get("description", "")
            device = port.get("device", "")
            text = f"{device} - {desc}" if desc else device
            self.port_combo.addItem(text, port)

        if self.port_combo.count() == 0:
            self._log("未检测到串口设备")

    def _toggle_connection(self):
        if self.transport.is_connected:
            self.transport.disconnect()
            self.connect_btn.setText("连接")
            self.connect_btn.setProperty("class", "")
            self.connect_btn.style().unpolish(self.connect_btn)
            self.connect_btn.style().polish(self.connect_btn)
            self._log("已断开连接")
            self.connection_changed.emit(False)
        else:
            idx = self.port_combo.currentIndex()
            if idx < 0:
                QMessageBox.warning(self, "错误", "请选择串口")
                return

            port_data = self.port_combo.itemData(idx)
            port_name = port_data.get("device", "")

            try:
                baudrate = int(self.baud_combo.currentText())
            except ValueError:
                baudrate = 115200

            success = self.transport.connect(port_name, baudrate)
            if success:
                self.connect_btn.setText("断开")
                self._log(f"已连接: {port_name} @ {baudrate} baud")
                self.connection_changed.emit(True)

                try:
                    info = self.controller.identify()
                    self.fc_variant_label.setText(info.identifier)
                    self.fc_version_label.setText(info.version)
                    self.fc_target_label.setText(info.target)
                    self.fc_build_label.setText(f"{info.build_date} {info.build_time}")
                    self._log(f"飞控: {info.identifier} v{info.version}")
                except Exception as e:
                    self._log(f"读取飞控信息失败: {e}")

                self.status_message.emit(f"已连接: {info.identifier}")
            else:
                QMessageBox.critical(self, "连接失败", f"无法连接到 {port_name}")

    def _log(self, message: str):
        self.log_text.append(message)

    def is_device_connected(self) -> bool:
        return self.transport.is_connected