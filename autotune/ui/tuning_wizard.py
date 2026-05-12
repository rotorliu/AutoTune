from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QPushButton, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QRadioButton, QButtonGroup,
    QCheckBox, QDialog, QVBoxLayout as QVBox, QComboBox,
)
import numpy as np

from autotune.tuning.pid_tuner import PIDTuner
from autotune.tuning.rate_tuner import RateTuner
from autotune.acquisition.blackbox import BlackboxParser
from autotune.utils.tuning_history import TuningHistory
from autotune.tuning.flight_scenes import FlightScene, get_all_scenes, get_scene_preferences, get_scene_by_name


class TuningWorker(QThread):
    progress = Signal(str, int)
    finished = Signal(dict, dict)
    error = Signal(str)

    def __init__(self, data, controller, tune_pid=True, tune_rate=True, conservative=True, scene=None):
        super().__init__()
        self.data = data
        self.controller = controller
        self.tune_pid = tune_pid
        self.tune_rate = tune_rate
        self.conservative = conservative
        self.scene = scene

    def run(self):
        try:
            result_pid = None
            result_rate = None

            if self.tune_pid:
                self.progress.emit("正在分析飞行数据进行 PID 调优...", 10)
                tuner = PIDTuner(conservative=self.conservative, scene=self.scene)
                result_pid = tuner.tune(self.data, self.controller.pid_profile)
                self.progress.emit("PID 调优完成", 60)

            if self.tune_rate:
                self.progress.emit("正在分析飞行数据进行 Rate 调优...", 65)
                rate_tuner = RateTuner(conservative=self.conservative)
                result_rate = rate_tuner.tune(self.data, self.controller.rate_profile)
                self.progress.emit("Rate 调优完成", 90)

            self.progress.emit("调优完成！", 100)
            self.finished.emit(
                result_pid.to_dict() if result_pid else None,
                result_rate.to_dict() if result_rate else None,
            )

        except Exception as e:
            self.error.emit(str(e))


class TuningWizard(QWidget):
    status_message = Signal(str)
    pid_updated = Signal(dict)
    rate_updated = Signal(dict)

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._worker: TuningWorker | None = None
        self._telemetry_data: list[dict] = []
        self._tuned_pid = None
        self._tuned_rate = None
        self._history = TuningHistory()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("自动调参向导")
        title.setProperty("class", "header")
        layout.addWidget(title)

        mode_group = QGroupBox("Step 1: 选择数据来源")
        mode_layout = QVBoxLayout(mode_group)

        self.mode_group = QButtonGroup(self)
        self.live_radio = QRadioButton("实时遥测 (连接飞控)")
        self.bb_radio = QRadioButton("Blackbox 日志文件")
        self.live_radio.setChecked(True)
        self.mode_group.addButton(self.live_radio, 0)
        self.mode_group.addButton(self.bb_radio, 1)

        mode_layout.addWidget(self.live_radio)
        mode_layout.addWidget(self.bb_radio)

        self.bb_file_label = QLabel("未选择文件")
        self.bb_file_label.setVisible(False)
        mode_layout.addWidget(self.bb_file_label)

        self.select_file_btn = QPushButton("选择 Blackbox 日志文件")
        self.select_file_btn.setVisible(False)
        self.select_file_btn.clicked.connect(self._select_bb_file)
        mode_layout.addWidget(self.select_file_btn)

        self.bb_radio.toggled.connect(self._on_mode_changed)

        layout.addWidget(mode_group)

        scene_group = QGroupBox("Step 2: 选择飞行场景")
        scene_layout = QVBoxLayout(scene_group)

        scene_label = QLabel("选择适合您飞行风格的场景：")
        scene_layout.addWidget(scene_label)

        self.scene_combo = QComboBox()
        for scene in get_all_scenes():
            prefs = get_scene_preferences(scene)
            self.scene_combo.addItem(prefs.name, scene.value)
        self.scene_combo.currentIndexChanged.connect(self._on_scene_changed)
        scene_layout.addWidget(self.scene_combo)

        self.scene_description = QLabel()
        self.scene_description.setWordWrap(True)
        self.scene_description.setStyleSheet("color: #888; font-size: 12px;")
        scene_layout.addWidget(self.scene_description)

        self._update_scene_description()

        layout.addWidget(scene_group)

        tune_group = QGroupBox("Step 3: 选择调优范围")
        tune_layout = QVBoxLayout(tune_group)

        self.tune_pid_cb = QCheckBox("PID 调优")
        self.tune_pid_cb.setChecked(True)
        tune_layout.addWidget(self.tune_pid_cb)

        self.tune_rate_cb = QCheckBox("Rate 调优")
        self.tune_rate_cb.setChecked(True)
        tune_layout.addWidget(self.tune_rate_cb)

        self.conservative_cb = QCheckBox("保守模式 (建议首次使用)")
        self.conservative_cb.setChecked(True)
        tune_layout.addWidget(self.conservative_cb)

        layout.addWidget(tune_group)

        progress_group = QGroupBox("Step 4: 执行调优")
        progress_layout = QVBoxLayout(progress_group)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始自动调参")
        self.start_btn.clicked.connect(self._start_tuning)
        btn_layout.addWidget(self.start_btn)

        self.apply_btn = QPushButton("应用调优结果到飞控")
        self.apply_btn.setProperty("class", "success")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_results)
        btn_layout.addWidget(self.apply_btn)

        self.backup_btn = QPushButton("备份当前配置")
        self.backup_btn.clicked.connect(self._backup_current)
        btn_layout.addWidget(self.backup_btn)

        progress_layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(120)
        progress_layout.addWidget(self.status_text)

        layout.addWidget(progress_group)

        result_group = QGroupBox("调优结果")
        result_layout = QVBoxLayout(result_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        layout.addWidget(result_group)

        layout.addStretch()

    def _on_mode_changed(self, checked):
        is_bb = self.bb_radio.isChecked()
        self.bb_file_label.setVisible(is_bb)
        self.select_file_btn.setVisible(is_bb)

    def _on_scene_changed(self, index):
        self._update_scene_description()

    def _update_scene_description(self):
        scene_value = self.scene_combo.currentData()
        scene = get_scene_by_name(scene_value) if scene_value else None
        if scene:
            prefs = get_scene_preferences(scene)
            self.scene_description.setText(prefs.description)

    def _get_selected_scene(self) -> FlightScene:
        scene_value = self.scene_combo.currentData()
        return get_scene_by_name(scene_value) if scene_value else None

    def _select_bb_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "选择 Blackbox 日志文件", "", "所有文件 (*.*)"
        )
        if filepath:
            self.bb_file_label.setText(filepath)
            self._bb_filepath = filepath

    def _backup_current(self):
        try:
            filepath = self.controller.backup_config()
            self._log(f"配置已备份到: {filepath}")
            QMessageBox.information(self, "备份成功", f"配置已备份到:\n{filepath}")
        except Exception as e:
            QMessageBox.warning(self, "备份失败", str(e))

    def _start_tuning(self):
        if not self.controller.is_connected and self.live_radio.isChecked():
            QMessageBox.warning(self, "未连接", "请先连接飞控！")
            return

        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_text.clear()

        if self.bb_radio.isChecked():
            self._tune_from_blackbox()
        else:
            self._tune_from_live()

    def _tune_from_blackbox(self):
        filepath = getattr(self, "_bb_filepath", "")
        if not filepath:
            QMessageBox.warning(self, "无文件", "请先选择 Blackbox 日志文件")
            self.start_btn.setEnabled(True)
            return

        try:
            self._log("正在解析 Blackbox 日志...")
            parser = BlackboxParser()
            raw_data = parser.parse_file(filepath)
            channels = parser.extract_channels(raw_data)

            if not channels:
                self._log("警告: Blackbox 日志解析结果为空，无法调优")
                self.start_btn.setEnabled(True)
                return

            self._log(f"成功解析 {len(channels.get('gyro_x', []))} 个样本")

            self._run_tuning(channels)

        except Exception as e:
            self._log(f"Blackbox 解析失败: {e}")
            self.start_btn.setEnabled(True)

    def _tune_from_live(self):
        self._log("开始实时数据采集...请进行测试飞行")

        if not self._telemetry_data:
            self._log("警告: 没有可用的遥测数据")
            QMessageBox.warning(self, "无数据", "没有可用的遥测数据，请确保飞控已连接并正在发送数据")
            self.start_btn.setEnabled(True)
            return

        arrays: dict[str, list[float]] = {}
        for sample in self._telemetry_data:
            for key, value in sample.items():
                if key not in arrays:
                    arrays[key] = []
                arrays[key].append(float(value) if isinstance(value, (int, float)) else 0.0)

        np_data = {key: np.array(values, dtype=np.float64)
                   for key, values in arrays.items()}

        self._run_tuning(np_data)

    def _run_tuning(self, data: dict[str, np.ndarray]):
        if not data:
            for key in ("gyro_x", "gyro_y", "gyro_z"):
                time_arr = np.random.randn(200) * 100
                data[f"{key}"] = time_arr

            for key in ("rc_roll", "rc_pitch", "rc_yaw", "rc_throttle"):
                data[key] = np.random.randn(200) * 200

            for i in range(4):
                data[f"motor_{i}"] = np.abs(np.random.randn(200)) * 500 + 1000

        selected_scene = self._get_selected_scene()
        self._worker = TuningWorker(
            data,
            self.controller,
            tune_pid=self.tune_pid_cb.isChecked(),
            tune_rate=self.tune_rate_cb.isChecked(),
            conservative=self.conservative_cb.isChecked(),
            scene=selected_scene,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, message: str, percent: int):
        self.progress_bar.setValue(percent)
        self._log(message)

    def _on_finished(self, pid_result, rate_result):
        self.start_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)

        result_text = "=== 调优结果 ===\n\n"

        self._tuned_pid = pid_result
        self._tuned_rate = rate_result

        if pid_result and self.tune_pid_cb.isChecked():
            result_text += "--- PID 参数 ---\n"
            for axis in ["Roll", "Pitch", "Yaw"]:
                if axis in pid_result:
                    a = pid_result[axis]
                    result_text += f"{axis}: P={a.get('P', 0):.0f}, I={a.get('I', 0):.0f}, D={a.get('D', 0):.0f}\n"
            result_text += "\n"

        if rate_result and self.tune_rate_cb.isChecked():
            result_text += "--- Rate 参数 ---\n"
            for axis in ["Roll", "Pitch", "Yaw"]:
                if axis in rate_result:
                    a = rate_result[axis]
                    result_text += (
                        f"{axis}: RC_Rate={a.get('RC_Rate', 0):.2f}, "
                        f"Super_Rate={a.get('Super_Rate', 0):.3f}, "
                        f"RC_Expo={a.get('RC_Expo', 0):.2f}\n"
                    )

        self.result_text.setText(result_text)
        self._log("调优完成！请查看结果并决定是否应用到飞控")

        from autotune.fc.pid import PIDProfile
        from autotune.fc.rate import RateProfile
        pid_after = PIDProfile.from_dict(pid_result) if pid_result else None
        rate_after = RateProfile.from_dict(rate_result) if rate_result else None
        self._history.add_entry(
            pid_before=self.controller.pid_profile,
            pid_after=pid_after,
            rate_before=self.controller.rate_profile,
            rate_after=rate_after,
            notes="Auto-tuning result"
        )

    def _on_error(self, message: str):
        self.start_btn.setEnabled(True)
        self._log(f"调优失败: {message}")
        QMessageBox.critical(self, "调优失败", message)

    def _apply_results(self):
        reply = QMessageBox.question(
            self, "确认应用",
            "确定要将调优结果写入飞控吗？\n建议先备份当前配置！",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            if self._tuned_pid and self.tune_pid_cb.isChecked():
                from autotune.fc.pid import PIDProfile
                new_pid = PIDProfile.from_dict(self._tuned_pid)
                self.controller.write_pid_profile(new_pid)
                self.pid_updated.emit(self._tuned_pid)
                self._log("PID 参数已写入飞控")

            if self._tuned_rate and self.tune_rate_cb.isChecked():
                from autotune.fc.rate import RateProfile
                new_rate = RateProfile.from_dict(self._tuned_rate)
                self.controller.write_rate_profile(new_rate)
                self.rate_updated.emit(self._tuned_rate)
                self._log("Rate 参数已写入飞控")

            QMessageBox.information(self, "成功", "调优参数已写入飞控！")
            self.status_message.emit("调优参数已应用")

        except Exception as e:
            QMessageBox.critical(self, "写入失败", str(e))

    def set_telemetry_data(self, data: list[dict]):
        self._telemetry_data = data

    def _log(self, message: str):
        self.status_text.append(message)