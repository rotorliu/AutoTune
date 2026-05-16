from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QListWidget, QListWidgetItem, QPushButton,
    QTextEdit, QFileDialog, QMessageBox, QInputDialog,
)

from autotune.utils.profile_manager import ProfileManager


class ProfilePanel(QWidget):
    pid_load_request = Signal(dict)
    rate_load_request = Signal(dict)
    filter_load_request = Signal(dict)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.profile_manager = ProfileManager()
        self._init_ui()
        self._refresh_list()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("方案管理")
        title.setProperty("class", "header")
        layout.addWidget(title)

        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存当前配置")
        self.save_btn.clicked.connect(self._save_current)
        btn_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("加载选中方案")
        self.load_btn.clicked.connect(self._load_selected)
        btn_layout.addWidget(self.load_btn)

        self.delete_btn = QPushButton("删除选中方案")
        self.delete_btn.setProperty("class", "danger")
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_layout.addWidget(self.delete_btn)

        self.export_btn = QPushButton("导出方案")
        self.export_btn.clicked.connect(self._export_selected)
        btn_layout.addWidget(self.export_btn)

        self.import_btn = QPushButton("导入方案")
        self.import_btn.clicked.connect(self._import_profile)
        btn_layout.addWidget(self.import_btn)

        layout.addLayout(btn_layout)

        list_group = QGroupBox("已保存方案")
        list_layout = QVBoxLayout(list_group)

        self.profile_list = QListWidget()
        self.profile_list.itemSelectionChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.profile_list)

        layout.addWidget(list_group)

        detail_group = QGroupBox("方案详情")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(200)
        detail_layout.addWidget(self.detail_text)
        layout.addWidget(detail_group)

        action_group = QGroupBox("快捷操作")
        action_layout = QHBoxLayout(action_group)

        self.apply_to_fc_btn = QPushButton("应用当前方案到飞控")
        self.apply_to_fc_btn.setProperty("class", "success")
        self.apply_to_fc_btn.clicked.connect(self._apply_to_fc)
        action_layout.addWidget(self.apply_to_fc_btn)

        self.restore_btn = QPushButton("恢复飞控默认值")
        self.restore_btn.setProperty("class", "danger")
        self.restore_btn.clicked.connect(self._restore_defaults)
        action_layout.addWidget(self.restore_btn)

        layout.addWidget(action_group)

        layout.addStretch()

    def _refresh_list(self):
        self.profile_list.clear()
        profiles = self.profile_manager.list_profiles()
        for profile in profiles:
            item = QListWidgetItem(
                f"{profile.get('name', 'Unnamed')} - {profile.get('timestamp', 'Unknown')}"
            )
            item.setData(Qt.UserRole, profile)
            self.profile_list.addItem(item)

    def _save_current(self):
        name, ok = QInputDialog.getText(self, "保存方案", "方案名称:")
        if not ok or not name:
            return

        notes, ok2 = QInputDialog.getText(self, "备注", "备注 (可选):")
        if not ok2:
            notes = ""

        try:
            filepath = self.profile_manager.save_profile(
                name,
                self.controller.pid_profile,
                self.controller.rate_profile,
                notes,
            )
            self._refresh_list()
            self.status_message.emit(f"方案已保存: {name}")
            QMessageBox.information(self, "保存成功", f"方案已保存到:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _load_selected(self):
        item = self.profile_list.currentItem()
        if not item:
            QMessageBox.warning(self, "未选择", "请先选择一个方案")
            return

        profile_data = item.data(Qt.UserRole)
        filepath = profile_data.get("filepath", "")
        data = self.profile_manager.load_profile(filepath)

        if not data:
            return

        self._show_detail(data)

        self.pid_load_request.emit(data.get("pid", {}))
        self.rate_load_request.emit(data.get("rate", {}))
        filter_data = data.get("filter", {})
        if filter_data:
            self.filter_load_request.emit(filter_data)

    def _delete_selected(self):
        item = self.profile_list.currentItem()
        if not item:
            return

        reply = QMessageBox.question(
            self, "确认删除", "确定要删除选中的方案吗？",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        profile_data = item.data(Qt.UserRole)
        self.profile_manager.delete_profile(profile_data.get("filepath", ""))
        self._refresh_list()

    def _export_selected(self):
        item = self.profile_list.currentItem()
        if not item:
            QMessageBox.warning(self, "未选择", "请先选择一个方案")
            return

        profile_data = item.data(Qt.UserRole)
        src = profile_data.get("filepath", "")

        dest, _ = QFileDialog.getSaveFileName(self, "导出方案", "", "JSON 文件 (*.json)")
        if not dest:
            return

        import shutil
        shutil.copy2(src, dest)
        QMessageBox.information(self, "导出成功", f"方案已导出到:\n{dest}")

    def _import_profile(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "导入方案", "", "JSON 文件 (*.json)")
        if not filepath:
            return

        data = self.profile_manager.load_profile(filepath)
        if not data:
            QMessageBox.critical(self, "导入失败", "无法读取方案文件")
            return

        name = data.get("name", "Imported")
        pid = data.get("pid", {})
        rate = data.get("rate", {})

        from autotune.fc.pid import PIDProfile
        from autotune.fc.rate import RateProfile

        self.pid_load_request.emit(pid)
        self.rate_load_request.emit(rate)
        filter_data = data.get("filter", {})
        if filter_data:
            self.filter_load_request.emit(filter_data)

        self._refresh_list()
        QMessageBox.information(self, "导入成功", f"已导入方案: {name}")

    def _apply_to_fc(self):
        reply = QMessageBox.question(
            self, "确认应用",
            "确定要将当前显示的参数写入飞控吗？",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            self.controller.write_pid_profile(self.controller.pid_profile)
            self.controller.write_rate_profile(self.controller.rate_profile)
            QMessageBox.information(self, "成功", "参数已写入飞控！")
        except Exception as e:
            QMessageBox.critical(self, "写入失败", str(e))

    def _restore_defaults(self):
        reply = QMessageBox.question(
            self, "恢复默认值",
            "确定要恢复飞控默认参数吗？\n此操作会覆盖当前 PID 和 Rate 设置！",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            QMessageBox.information(
                self, "提示",
                "请使用 Betaflight Configurator 的 'Reset Settings' 功能恢复默认值。"
            )

    def _on_selection_changed(self):
        item = self.profile_list.currentItem()
        if not item:
            return

        profile_data = item.data(Qt.UserRole)
        filepath = profile_data.get("filepath", "")
        data = self.profile_manager.load_profile(filepath)

        if data:
            self._show_detail(data)

    def _show_detail(self, data: dict):
        text = f"方案名称: {data.get('name', 'Unknown')}\n"
        text += f"保存时间: {data.get('timestamp', 'Unknown')}\n"
        text += f"备注: {data.get('notes', '无')}\n\n"

        pid = data.get("pid", {})
        if pid:
            text += "--- PID 参数 ---\n"
            for axis in ["Roll", "Pitch", "Yaw"]:
                a = pid.get(axis, {})
                text += f"{axis}: P={a.get('P', 0):.0f}, I={a.get('I', 0):.0f}, D={a.get('D', 0):.0f}\n"
            text += "\n"

        rate = data.get("rate", {})
        if rate:
            text += "--- Rate 参数 ---\n"
            for axis in ["Roll", "Pitch", "Yaw"]:
                a = rate.get(axis, {})
                text += (
                    f"{axis}: RC_Rate={a.get('RC_Rate', 0):.2f}, "
                    f"Super_Rate={a.get('Super_Rate', 0):.3f}, "
                    f"RC_Expo={a.get('RC_Expo', 0):.2f}\n"
                )

        self.detail_text.setText(text)

    @property
    def status_message(self):
        return self.parent().status_message if hasattr(self.parent(), 'status_message') else None