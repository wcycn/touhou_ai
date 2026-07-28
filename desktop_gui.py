#!/usr/bin/env python3
"""Touhou AI统一桌面控制台。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageDraw, ImageTk

from session_recorder import (
    DEFAULT_SESSIONS_DIR,
    iter_events,
    list_sessions,
    write_report,
)
from session_analysis import export_review_dataset, write_analysis


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR
ROOT_LAUNCHER = PROJECT_DIR / "touhou_ai.py"
SETTINGS_PATH = PROJECT_DIR / "settings.json"


def build_mode_arguments(
    command: str,
    *,
    mode: str,
    confidence: float,
    sensitivity: float,
    safe_margin: int,
    player_lost_timeout: float,
    allow_vertical: bool,
    recording_arguments: list[str] | None = None,
) -> list[str]:
    """为模拟观察和正式控制构造同一套AI参数。"""
    if command not in {"observe", "ai"}:
        raise ValueError(f"不支持的AI运行方式: {command}")
    arguments = [
        command,
        "--mode",
        mode,
        "--confidence",
        str(confidence),
        "--sensitivity",
        str(sensitivity),
        "--safe-margin",
        str(safe_margin),
        "--player-lost-timeout",
        str(player_lost_timeout),
        *(recording_arguments or []),
    ]
    if not allow_vertical:
        arguments.append("--no-vertical")
    return arguments


class ProcessManager:
    """后台运行命令并把输出安全送回Tk主线程。"""

    def __init__(self, event_queue: queue.Queue):
        self.event_queue = event_queue
        self.processes: dict[str, subprocess.Popen] = {}
        self.stopping: set[str] = set()
        self._lock = threading.Lock()

    def is_running(self, name: str) -> bool:
        with self._lock:
            process = self.processes.get(name)
            return bool(process and process.poll() is None)

    def start(self, name: str, arguments: list[str]) -> bool:
        with self._lock:
            existing = self.processes.get(name)
            if existing and existing.poll() is None:
                self.event_queue.put(
                    ("log", name, f"{name} 已在运行，忽略重复启动\n")
                )
                return False
            command = [sys.executable, str(ROOT_LAUNCHER), *arguments]
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
            except OSError as exc:
                self.event_queue.put(
                    ("log", name, f"启动失败: {exc}\n")
                )
                return False
            self.processes[name] = process

        self.event_queue.put(("state", name, "running"))
        self.event_queue.put(
            ("log", name, f"$ {' '.join(command)}\n")
        )
        threading.Thread(
            target=self._read_output,
            args=(name, process),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._watch,
            args=(name, process),
            daemon=True,
        ).start()
        return True

    def _read_output(self, name: str, process: subprocess.Popen) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            self.event_queue.put(("log", name, line))

    def _watch(self, name: str, process: subprocess.Popen) -> None:
        return_code = process.wait()
        with self._lock:
            requested_stop = name in self.stopping
            self.stopping.discard(name)
        state = "stopped" if requested_stop else (
            "completed" if return_code == 0 else "failed"
        )
        self.event_queue.put(("state", name, state))
        self.event_queue.put(
            ("log", name, f"[进程结束，退出码 {return_code}]\n")
        )
        self.event_queue.put(("refresh_sessions", None, None))

    def stop(self, name: str) -> bool:
        with self._lock:
            process = self.processes.get(name)
            if not process or process.poll() is not None:
                return False
            self.stopping.add(name)
        self.event_queue.put(("log", name, "正在请求安全停止...\n"))
        try:
            os.killpg(process.pid, signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.terminate()
            except OSError:
                return False

        def force_after_timeout():
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.event_queue.put(
                    ("log", name, "安全停止超时，正在终止进程...\n")
                )
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass

        threading.Thread(target=force_after_timeout, daemon=True).start()
        return True

    def stop_all(self) -> None:
        for name in list(self.processes):
            self.stop(name)


class TouhouControlCenter:
    STATUS_COLORS = {
        "idle": "#6b7280",
        "running": "#059669",
        "completed": "#2563eb",
        "stopped": "#6b7280",
        "failed": "#dc2626",
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Touhou AI 控制中心")
        self.root.geometry("1180x820")
        self.root.minsize(980, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.events: queue.Queue = queue.Queue()
        self.process_manager = ProcessManager(self.events)
        self.process_states = {
            name: "idle"
            for name in ("game", "locate", "observe", "ai", "control-test")
        }
        self.status_labels: dict[str, ttk.Label] = {}
        self.session_rows: dict[str, dict] = {}
        self.current_session_dir: Path | None = None
        self.current_frames: list[tuple[Path, dict]] = []
        self.current_frame_index = 0
        self.preview_photo = None

        settings = self.load_settings()
        self.mode_var = tk.StringVar(value=settings.get("mode", "balanced"))
        self.confidence_var = tk.StringVar(
            value=str(settings.get("confidence", "0.35"))
        )
        self.sensitivity_var = tk.StringVar(
            value=str(settings.get("sensitivity", "0.05"))
        )
        self.record_var = tk.BooleanVar(value=settings.get("record", True))
        self.record_fps_var = tk.StringVar(
            value=str(settings.get("record_fps", "2.0"))
        )
        self.observe_preview_var = tk.BooleanVar(
            value=settings.get("observe_preview", True)
        )
        self.safe_margin_var = tk.StringVar(
            value=str(settings.get("safe_margin", "36"))
        )
        self.player_lost_timeout_var = tk.StringVar(
            value=str(settings.get("player_lost_timeout", "0.70"))
        )
        self.vertical_movement_var = tk.BooleanVar(
            value=settings.get("vertical_movement", True)
        )

        self._setup_style()
        self._build_ui()
        self.refresh_sessions()
        self.root.after(100, self.poll_events)

    @staticmethod
    def load_settings() -> dict:
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save_settings(self) -> None:
        settings = {
            "mode": self.mode_var.get(),
            "confidence": self.confidence_var.get(),
            "sensitivity": self.sensitivity_var.get(),
            "record": self.record_var.get(),
            "record_fps": self.record_fps_var.get(),
            "observe_preview": self.observe_preview_var.get(),
            "safe_margin": self.safe_margin_var.get(),
            "player_lost_timeout": self.player_lost_timeout_var.get(),
            "vertical_movement": self.vertical_movement_var.get(),
        }
        temporary = SETTINGS_PATH.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(settings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(SETTINGS_PATH)
        except OSError as exc:
            self.append_log("GUI", f"无法保存界面设置: {exc}\n")

    def _setup_style(self):
        style = ttk.Style()
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 20, "bold"))
        style.configure("Heading.TLabel", font=("Sans", 12, "bold"))
        style.configure("Accent.TButton", font=("Sans", 10, "bold"))
        style.configure("Status.TLabel", font=("Sans", 9, "bold"))
        style.configure(
            "Observe.TButton",
            font=("Sans", 13, "bold"),
            padding=(16, 14),
            foreground="#064e3b",
            background="#a7f3d0",
        )
        style.map(
            "Observe.TButton",
            background=[("active", "#6ee7b7"), ("disabled", "#d1d5db")],
        )
        style.configure(
            "AIStart.TButton",
            font=("Sans", 14, "bold"),
            padding=(16, 14),
            foreground="white",
            background="#047857",
        )
        style.map(
            "AIStart.TButton",
            background=[("active", "#059669"), ("disabled", "#9ca3af")],
        )
        style.configure(
            "Emergency.TButton",
            font=("Sans", 11, "bold"),
            padding=(12, 9),
            foreground="white",
            background="#b91c1c",
        )
        style.map(
            "Emergency.TButton",
            background=[("active", "#dc2626"), ("disabled", "#9ca3af")],
        )
        style.configure("Mode.TLabel", font=("Sans", 13, "bold"))

    def _build_ui(self):
        header = ttk.Frame(self.root, padding=(18, 12))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Touhou AI 控制中心",
            style="Title.TLabel",
        ).pack(side="left")
        ttk.Label(
            header,
            text="游戏 · 观察 · 自动控制 · 记录 · 回放",
            foreground="#4b5563",
        ).pack(side="left", padx=18, pady=(8, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        self.control_tab = ttk.Frame(self.notebook, padding=14)
        self.sessions_tab = ttk.Frame(self.notebook, padding=10)
        self.logs_tab = ttk.Frame(self.notebook, padding=10)
        self.roadmap_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.control_tab, text="控制台")
        self.notebook.add(self.sessions_tab, text="会话与回放")
        self.notebook.add(self.logs_tab, text="实时日志")
        self.notebook.add(self.roadmap_tab, text="项目路线")

        self._build_control_tab()
        self._build_sessions_tab()
        self._build_logs_tab()
        self._build_roadmap_tab()

    def _build_control_tab(self):
        self.control_tab.columnconfigure(0, weight=1)
        self.control_tab.columnconfigure(1, weight=1)

        primary = ttk.LabelFrame(
            self.control_tab,
            text="选择 AI 运行方式",
            padding=14,
        )
        primary.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 12),
        )
        primary.columnconfigure(0, weight=1)
        primary.columnconfigure(1, weight=1)

        self.mode_status_label = ttk.Label(
            primary,
            text="● 当前：空闲（不会发送按键）",
            style="Mode.TLabel",
            foreground="#4b5563",
        )
        self.mode_status_label.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 10),
        )

        observe_card = ttk.Frame(primary, padding=8)
        observe_card.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        ttk.Label(
            observe_card,
            text="安全模式",
            style="Heading.TLabel",
            foreground="#047857",
        ).pack(anchor="w")
        ttk.Label(
            observe_card,
            text=(
                "运行完整AI：检测、自机/敌弹跟踪、轨迹预测和动作决策；"
                "最后一步强制禁用所有键鼠输入。"
            ),
            wraplength=470,
            foreground="#374151",
        ).pack(anchor="w", pady=(3, 8))
        self.observe_button = ttk.Button(
            observe_card,
            text="▶ 启动 AI 模拟观察（不按键）",
            command=self.start_observe,
            style="Observe.TButton",
        )
        self.observe_button.pack(fill="x")

        ai_card = ttk.Frame(primary, padding=8)
        ai_card.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        ttk.Label(
            ai_card,
            text="控制模式",
            style="Heading.TLabel",
            foreground="#b45309",
        ).pack(anchor="w")
        ttk.Label(
            ai_card,
            text=(
                "运行同一套AI，并把稳定后的移动、射击和炸弹动作发送给"
                "已确认获得焦点的游戏窗口。"
            ),
            wraplength=470,
            foreground="#374151",
        ).pack(anchor="w", pady=(3, 8))
        self.ai_button = ttk.Button(
            ai_card,
            text="▶ 启动 AI 自动控制（会按键）",
            command=self.start_ai,
            style="AIStart.TButton",
        )
        self.ai_button.pack(fill="x")

        self.stop_modes_button = ttk.Button(
            primary,
            text="■ 立即停止 AI / 模拟观察并释放按键",
            command=self.stop_control_modes,
            style="Emergency.TButton",
            state="disabled",
        )
        self.stop_modes_button.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(12, 0),
        )

        status_frame = ttk.LabelFrame(
            self.control_tab,
            text="组件状态",
            padding=12,
        )
        status_frame.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 12),
        )
        for column, (name, label) in enumerate(
            (
                ("game", "游戏启动"),
                ("locate", "窗口定位"),
                ("observe", "AI模拟观察"),
                ("ai", "AI自动控制"),
                ("control-test", "输入测试"),
            )
        ):
            card = ttk.Frame(status_frame, padding=8)
            card.grid(row=0, column=column, sticky="ew", padx=4)
            status_frame.columnconfigure(column, weight=1)
            ttk.Label(card, text=label).pack()
            status = ttk.Label(
                card,
                text="● 空闲",
                style="Status.TLabel",
                foreground=self.STATUS_COLORS["idle"],
            )
            status.pack(pady=(4, 0))
            self.status_labels[name] = status

        workflow = ttk.LabelFrame(
            self.control_tab,
            text="运行前准备",
            padding=14,
        )
        workflow.grid(row=2, column=0, sticky="nsew", padx=(0, 7))

        ttk.Button(
            workflow,
            text="启动游戏（vpatch）",
            command=self.start_game,
            style="Accent.TButton",
        ).pack(fill="x", pady=4)
        ttk.Button(
            workflow,
            text="检查窗口定位",
            command=lambda: self.start_process("locate", ["locate"]),
        ).pack(fill="x", pady=4)
        ttk.Button(
            workflow,
            text="测试左右输入",
            command=lambda: self.start_process(
                "control-test",
                ["control-test"],
            ),
        ).pack(fill="x", pady=4)
        ttk.Separator(workflow).pack(fill="x", pady=10)
        ttk.Button(
            workflow,
            text="运行环境检查",
            command=lambda: self.start_process("check", ["check"]),
        ).pack(fill="x", pady=4)
        ttk.Button(
            workflow,
            text="运行全部无按键测试",
            command=lambda: self.start_process("test", ["test"]),
        ).pack(fill="x", pady=4)
        ttk.Label(
            workflow,
            text=(
                "推荐首次顺序：启动游戏 → 检查定位 → "
                "AI模拟观察 → 输入测试 → AI自动控制"
            ),
            wraplength=430,
            foreground="#4b5563",
        ).pack(anchor="w", pady=(12, 0))

        config = ttk.LabelFrame(
            self.control_tab,
            text="检测与记录参数",
            padding=14,
        )
        config.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        config.columnconfigure(1, weight=1)

        ttk.Label(config, text="AI模式").grid(
            row=0, column=0, sticky="w", pady=5
        )
        ttk.Combobox(
            config,
            textvariable=self.mode_var,
            values=("balanced", "aggressive", "defensive"),
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=5)

        ttk.Label(config, text="检测置信度").grid(
            row=1, column=0, sticky="w", pady=5
        )
        ttk.Entry(config, textvariable=self.confidence_var).grid(
            row=1, column=1, sticky="ew", pady=5
        )
        ttk.Label(config, text="风险灵敏度").grid(
            row=2, column=0, sticky="w", pady=5
        )
        ttk.Entry(config, textvariable=self.sensitivity_var).grid(
            row=2, column=1, sticky="ew", pady=5
        )
        ttk.Checkbutton(
            config,
            text="记录检测、动作和抽样画面",
            variable=self.record_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 5))
        ttk.Label(config, text="保存画面FPS").grid(
            row=4, column=0, sticky="w", pady=5
        )
        ttk.Entry(config, textvariable=self.record_fps_var).grid(
            row=4, column=1, sticky="ew", pady=5
        )
        ttk.Checkbutton(
            config,
            text="观察模式显示实时预览",
            variable=self.observe_preview_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Label(config, text="边缘安全距离(px)").grid(
            row=6, column=0, sticky="w", pady=5
        )
        ttk.Entry(config, textvariable=self.safe_margin_var).grid(
            row=6, column=1, sticky="ew", pady=5
        )
        ttk.Label(config, text="自机漏检停控(秒)").grid(
            row=7, column=0, sticky="w", pady=5
        )
        ttk.Entry(config, textvariable=self.player_lost_timeout_var).grid(
            row=7, column=1, sticky="ew", pady=5
        )
        ttk.Checkbutton(
            config,
            text="允许上下方向风险规划",
            variable=self.vertical_movement_var,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=5)

        ttk.Separator(config).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=12
        )
        ttk.Button(
            config,
            text="打开会话目录",
            command=lambda: self.open_path(DEFAULT_SESSIONS_DIR),
        ).grid(row=10, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(
            config,
            text="选择标注数据集并评估模型",
            command=self.start_model_evaluation,
        ).grid(row=11, column=0, columnspan=2, sticky="ew", pady=4)

        help_frame = ttk.LabelFrame(
            self.control_tab,
            text="安全提示",
            padding=12,
        )
        help_frame.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(12, 0),
        )
        ttk.Label(
            help_frame,
            text=(
                "AI模拟观察会运行完整决策，但绝不发送按键。"
                "AI自动控制会持续聚焦游戏窗口；想操作其他程序前，"
                "请先点击上方红色停止按钮。"
            ),
            wraplength=950,
            foreground="#92400e",
        ).pack(anchor="w")

    def _build_sessions_tab(self):
        self.sessions_tab.rowconfigure(0, weight=1)
        self.sessions_tab.columnconfigure(0, weight=3)
        self.sessions_tab.columnconfigure(1, weight=2)

        left = ttk.Frame(self.sessions_tab)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(left)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(
            toolbar,
            text="刷新",
            command=self.refresh_sessions,
        ).pack(side="left")
        ttk.Button(
            toolbar,
            text="生成报告",
            command=self.generate_current_report,
        ).pack(side="left", padx=6)
        ttk.Button(
            toolbar,
            text="深入分析/审核清单",
            command=self.analyze_current_session,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            toolbar,
            text="导出标注候选",
            command=self.export_current_review_dataset,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            toolbar,
            text="打开目录",
            command=self.open_current_session,
        ).pack(side="left")

        columns = (
            "source",
            "status",
            "duration",
            "events",
            "frames",
            "player_rate",
            "danger",
            "risk",
            "switches",
        )
        self.session_tree = ttk.Treeview(
            left,
            columns=columns,
            show="tree headings",
            selectmode="browse",
        )
        self.session_tree.heading("#0", text="会话")
        self.session_tree.column("#0", width=210)
        headings = {
            "source": "来源",
            "status": "状态",
            "duration": "秒",
            "events": "事件",
            "frames": "画面",
            "player_rate": "自机率",
            "danger": "最高危险",
            "risk": "碰撞风险",
            "switches": "切向",
        }
        for name in columns:
            self.session_tree.heading(name, text=headings[name])
            self.session_tree.column(name, width=75, anchor="center")
        self.session_tree.grid(row=1, column=0, sticky="nsew")
        self.session_tree.bind("<<TreeviewSelect>>", self.on_session_select)

        right = ttk.Frame(self.sessions_tab)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.rowconfigure(1, weight=3)
        right.rowconfigure(3, weight=2)
        right.columnconfigure(0, weight=1)

        ttk.Label(
            right,
            text="抽样帧回放",
            style="Heading.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self.frame_label = ttk.Label(
            right,
            text="选择一个会话",
            anchor="center",
        )
        self.frame_label.grid(row=1, column=0, sticky="nsew", pady=6)

        navigation = ttk.Frame(right)
        navigation.grid(row=2, column=0, sticky="ew")
        ttk.Button(
            navigation,
            text="◀ 上一帧",
            command=lambda: self.change_frame(-1),
        ).pack(side="left")
        self.frame_counter = ttk.Label(navigation, text="0 / 0")
        self.frame_counter.pack(side="left", expand=True)
        ttk.Button(
            navigation,
            text="下一帧 ▶",
            command=lambda: self.change_frame(1),
        ).pack(side="right")

        self.session_details = scrolledtext.ScrolledText(
            right,
            height=12,
            wrap="word",
            font=("Monospace", 9),
        )
        self.session_details.grid(row=3, column=0, sticky="nsew", pady=(8, 0))

    def _build_logs_tab(self):
        toolbar = ttk.Frame(self.logs_tab)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(
            toolbar,
            text="清空日志",
            command=lambda: self.log_text.delete("1.0", "end"),
        ).pack(side="left")
        self.log_text = scrolledtext.ScrolledText(
            self.logs_tab,
            wrap="word",
            font=("Monospace", 10),
            bg="#111827",
            fg="#e5e7eb",
            insertbackground="white",
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_configure("system", foreground="#93c5fd")
        self.log_text.tag_configure("error", foreground="#fca5a5")

    def _build_roadmap_tab(self):
        text = scrolledtext.ScrolledText(
            self.roadmap_tab,
            wrap="word",
            font=("Sans", 10),
        )
        text.pack(fill="both", expand=True)
        roadmap_path = PROJECT_DIR / "docs" / "ROADMAP.md"
        try:
            content = roadmap_path.read_text(encoding="utf-8")
        except OSError as exc:
            content = f"无法读取路线文档：{exc}"
        text.insert("1.0", content)
        text.configure(state="disabled")

    def validated_parameters(
        self,
    ) -> tuple[str, float, float, float, int, float, bool] | None:
        try:
            confidence = float(self.confidence_var.get())
            sensitivity = float(self.sensitivity_var.get())
            record_fps = float(self.record_fps_var.get())
            safe_margin = int(self.safe_margin_var.get())
            player_lost_timeout = float(self.player_lost_timeout_var.get())
        except ValueError:
            messagebox.showerror(
                "参数错误",
                "置信度、敏感度、FPS、安全距离和停控时间必须是数字",
            )
            return None
        if not 0 <= confidence <= 1 or not 0 <= sensitivity <= 1:
            messagebox.showerror("参数错误", "置信度和敏感度必须在0到1之间")
            return None
        if not 0 <= record_fps <= 20:
            messagebox.showerror("参数错误", "记录FPS必须在0到20之间")
            return None
        if not 8 <= safe_margin <= 160:
            messagebox.showerror("参数错误", "边缘安全距离必须在8到160像素之间")
            return None
        if not 0.35 <= player_lost_timeout <= 3:
            messagebox.showerror("参数错误", "自机漏检停控时间必须在0.35到3秒之间")
            return None
        return (
            self.mode_var.get(),
            confidence,
            sensitivity,
            record_fps,
            safe_margin,
            player_lost_timeout,
            self.vertical_movement_var.get(),
        )

    def recording_arguments(self, record_fps: float) -> list[str]:
        if not self.record_var.get():
            return []
        return [
            "--record",
            "--record-dir",
            str(DEFAULT_SESSIONS_DIR),
            "--record-fps",
            str(record_fps),
        ]

    def start_process(self, name: str, arguments: list[str]) -> None:
        self.process_manager.start(name, arguments)
        if name not in {"observe", "ai"}:
            self.notebook.select(self.logs_tab)

    def stop_control_modes(self):
        stopped = False
        for name in ("ai", "observe"):
            stopped = self.process_manager.stop(name) or stopped
        if stopped:
            self.mode_status_label.configure(
                text="● 当前：正在安全停止并释放按键…",
                foreground="#b45309",
            )
        else:
            self.update_mode_banner()

    def update_mode_banner(self):
        ai_running = self.process_states.get("ai") == "running"
        observe_running = self.process_states.get("observe") == "running"
        if ai_running:
            self.mode_status_label.configure(
                text="● 当前：AI自动控制中（正在向游戏发送按键）",
                foreground="#b91c1c",
            )
            self.ai_button.configure(state="disabled")
            self.observe_button.configure(state="disabled")
            self.stop_modes_button.configure(state="normal")
        elif observe_running:
            self.mode_status_label.configure(
                text="● 当前：AI模拟观察中（完整AI运行，不发送任何按键）",
                foreground="#047857",
            )
            self.ai_button.configure(state="normal")
            self.observe_button.configure(state="disabled")
            self.stop_modes_button.configure(state="normal")
        else:
            self.mode_status_label.configure(
                text="● 当前：空闲（不会发送按键）",
                foreground="#4b5563",
            )
            self.ai_button.configure(state="normal")
            self.observe_button.configure(state="normal")
            self.stop_modes_button.configure(state="disabled")

    def start_game(self):
        if not any(
            (PROJECT_DIR / "game" / name).is_file()
            for name in (
                "vpatch.exe",
                "th06c.exe",
                "th06.exe",
                "東方紅魔郷.exe",
            )
        ):
            messagebox.showerror(
                "缺少游戏文件",
                "发布包不包含游戏本体。请先按照 game/README.md "
                "将合法取得的游戏文件放入 game/。",
            )
            return
        self.start_process("game", ["game"])

    def start_model_evaluation(self):
        data_yaml = filedialog.askopenfilename(
            title="选择YOLO数据集data.yaml",
            filetypes=(("YAML", "*.yaml *.yml"), ("所有文件", "*")),
        )
        if data_yaml:
            self.start_process(
                "model-eval",
                ["model-eval", data_yaml],
            )

    def start_observe(self):
        values = self.validated_parameters()
        if not values:
            return
        self.save_settings()
        (
            mode,
            confidence,
            sensitivity,
            record_fps,
            safe_margin,
            player_lost_timeout,
            allow_vertical,
        ) = values
        arguments = build_mode_arguments(
            "observe",
            mode=mode,
            confidence=confidence,
            sensitivity=sensitivity,
            safe_margin=safe_margin,
            player_lost_timeout=player_lost_timeout,
            allow_vertical=allow_vertical,
            recording_arguments=self.recording_arguments(record_fps),
        )
        if not self.observe_preview_var.get():
            arguments.extend(
                [
                    "--no-preview",
                    "--save-dir",
                    str(PROJECT_DIR / "runs" / "observe"),
                ]
            )
        self.start_process("observe", arguments)

    def start_ai(self):
        values = self.validated_parameters()
        if not values:
            return
        self.save_settings()
        if self.process_manager.is_running("observe"):
            if not messagebox.askyesno(
                "停止观察",
                "观察窗口可能遮挡游戏画面。是否先停止观察再启动AI？",
            ):
                return
            self.process_manager.stop("observe")
            self.root.after(900, self.start_ai)
            return

        (
            mode,
            confidence,
            sensitivity,
            record_fps,
            safe_margin,
            player_lost_timeout,
            allow_vertical,
        ) = values
        arguments = build_mode_arguments(
            "ai",
            mode=mode,
            confidence=confidence,
            sensitivity=sensitivity,
            safe_margin=safe_margin,
            player_lost_timeout=player_lost_timeout,
            allow_vertical=allow_vertical,
            recording_arguments=self.recording_arguments(record_fps),
        )
        self.start_process("ai", arguments)

    def append_log(self, name: str, line: str):
        prefix = f"[{name}] "
        tag = "error" if "❌" in line or "失败" in line else "system"
        self.log_text.insert("end", prefix, tag)
        self.log_text.insert("end", line)
        self.log_text.see("end")

    def set_status(self, name: str, state: str):
        self.process_states[name] = state
        label = self.status_labels.get(name)
        if label:
            display = {
                "idle": "● 空闲",
                "running": "● 运行中",
                "completed": "● 已完成",
                "stopped": "● 已停止",
                "failed": "● 失败",
            }.get(state, state)
            label.configure(
                text=display,
                foreground=self.STATUS_COLORS.get(state, "#6b7280"),
            )
        self.update_mode_banner()

    def poll_events(self):
        try:
            while True:
                event_type, name, payload = self.events.get_nowait()
                if event_type == "log":
                    self.append_log(name, payload)
                elif event_type == "state":
                    self.set_status(name, payload)
                elif event_type == "refresh_sessions":
                    self.refresh_sessions()
                    self.root.after(1200, self.refresh_sessions)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def refresh_sessions(self):
        selected = self.current_session_dir
        for item in self.session_tree.get_children():
            self.session_tree.delete(item)
        self.session_rows.clear()
        for metadata in list_sessions():
            summary = metadata.get("summary", {})
            session_id = metadata.get("session_id", "unknown")
            item = self.session_tree.insert(
                "",
                "end",
                text=session_id,
                values=(
                    metadata.get("source", ""),
                    metadata.get("status", ""),
                    metadata.get("duration_seconds", 0),
                    summary.get("event_count", 0),
                    summary.get("saved_frame_count", 0),
                    f"{summary.get('player_detection_rate', 0):.0%}",
                    summary.get("max_danger", 0),
                    f"{summary.get('max_collision_risk', 0):.2f}",
                    summary.get("direction_switches", 0),
                ),
            )
            self.session_rows[item] = metadata
            if selected and Path(metadata["session_dir"]) == selected:
                self.session_tree.selection_set(item)

    def on_session_select(self, _event=None):
        selection = self.session_tree.selection()
        if not selection:
            return
        metadata = self.session_rows.get(selection[0])
        if not metadata:
            return
        self.current_session_dir = Path(metadata["session_dir"])
        self.session_details.delete("1.0", "end")
        self.session_details.insert(
            "1.0",
            json.dumps(metadata, ensure_ascii=False, indent=2),
        )

        self.current_frames = []
        for event in iter_events(self.current_session_dir):
            frame_file = event.get("frame_file")
            if frame_file:
                path = self.current_session_dir / frame_file
                if path.is_file():
                    self.current_frames.append((path, event))
        self.current_frame_index = 0
        self.show_current_frame()

    def show_current_frame(self):
        if not self.current_frames:
            self.preview_photo = None
            self.frame_label.configure(image="", text="该会话没有保存画面")
            self.frame_counter.configure(text="0 / 0")
            return
        path, event = self.current_frames[self.current_frame_index]
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:
            self.frame_label.configure(image="", text=f"无法读取画面: {exc}")
            return

        draw = ImageDraw.Draw(image)
        for detection in event.get("detections", []):
            bbox = detection.get("bbox", [])
            if len(bbox) != 4:
                continue
            class_name = detection.get("class_name", "object")
            color = (
                "#22c55e"
                if class_name == "character"
                else "#ef4444"
                if "bullet_enemy" in class_name
                else "#f59e0b"
            )
            box = tuple(int(value) for value in bbox)
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], max(0, box[1] - 12)), class_name, fill=color)

        action = event.get("action", {})
        state = event.get("state", {})
        for bullet in state.get("bullet_tracks", []):
            start = (
                int(bullet.get("center_x", 0)),
                int(bullet.get("center_y", 0)),
            )
            end = (
                int(bullet.get("predicted_x", start[0])),
                int(bullet.get("predicted_y", start[1])),
            )
            color = (
                "#ef4444"
                if float(bullet.get("collision_risk", 0.0) or 0.0) >= 0.65
                else "#f59e0b"
            )
            draw.line((start, end), fill=color, width=2)
        header = (
            f"t={event.get('elapsed_seconds', 0):.2f}s "
            f"scene={state.get('scene_state')} "
            f"risk={state.get('max_collision_risk', 0):.2f} "
            f"action={action.get('movement')} "
            f"executed={action.get('executed')}"
        )
        draw.rectangle((0, 0, image.width, 24), fill="#111827")
        draw.text((6, 5), header, fill="white")
        image.thumbnail((500, 440), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.frame_label.configure(image=self.preview_photo, text="")
        self.frame_counter.configure(
            text=f"{self.current_frame_index + 1} / {len(self.current_frames)}"
        )

    def change_frame(self, delta: int):
        if not self.current_frames:
            return
        self.current_frame_index = max(
            0,
            min(
                len(self.current_frames) - 1,
                self.current_frame_index + delta,
            ),
        )
        self.show_current_frame()

    def generate_current_report(self):
        if not self.current_session_dir:
            messagebox.showinfo("会话报告", "请先选择一个会话")
            return
        try:
            json_path, markdown_path = write_report(self.current_session_dir)
        except Exception as exc:
            messagebox.showerror("报告失败", str(exc))
            return
        messagebox.showinfo(
            "报告完成",
            f"已生成：\n{json_path}\n{markdown_path}",
        )

    def analyze_current_session(self):
        if not self.current_session_dir:
            messagebox.showinfo("会话分析", "请先选择一个会话")
            return
        try:
            json_path, markdown_path, review_path = write_analysis(
                self.current_session_dir
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("分析失败", str(exc))
            return
        self.append_log(
            "analysis",
            (
                f"已生成 {json_path.name}、{markdown_path.name}、"
                f"{review_path.name}\n"
            ),
        )
        messagebox.showinfo(
            "分析完成",
            f"已生成：\n{json_path}\n{markdown_path}\n{review_path}",
        )

    def export_current_review_dataset(self):
        if not self.current_session_dir:
            messagebox.showinfo("标注候选", "请先选择一个会话")
            return
        try:
            output_dir = export_review_dataset(self.current_session_dir)
        except (OSError, ValueError) as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self.append_log("analysis", f"标注候选已导出: {output_dir}\n")
        messagebox.showinfo(
            "导出完成",
            "预标注来自模型预测，必须人工检查后才能用于训练或评估。\n"
            f"{output_dir}",
        )

    def open_path(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc))

    def open_current_session(self):
        if not self.current_session_dir:
            messagebox.showinfo("打开目录", "请先选择一个会话")
            return
        self.open_path(self.current_session_dir)

    def on_close(self):
        active = [
            name
            for name in ("observe", "ai")
            if self.process_manager.is_running(name)
        ]
        if active and not messagebox.askyesno(
            "退出",
            f"仍在运行：{', '.join(active)}。\n是否安全停止并退出？",
        ):
            return
        self.save_settings()
        self.process_manager.stop_all()
        self.root.after(700, self.root.destroy)

    def run(self):
        self.append_log("GUI", f"项目目录: {ROOT}\n")
        self.append_log("GUI", "统一桌面控制台已就绪。\n")
        self.root.mainloop()


def main() -> int:
    try:
        application = TouhouControlCenter()
        application.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except tk.TclError as exc:
        print(f"❌ 无法启动桌面GUI: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
