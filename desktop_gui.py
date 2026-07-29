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
                    ("log", name, f"{name} is already running; request ignored.\n")
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
                    ("log", name, f"Could not start process: {exc}\n")
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
            ("log", name, f"[Process ended with exit code {return_code}]\n")
        )
        self.event_queue.put(("refresh_sessions", None, None))

    def stop(self, name: str) -> bool:
        with self._lock:
            process = self.processes.get(name)
            if not process or process.poll() is not None:
                return False
            self.stopping.add(name)
        self.event_queue.put(("log", name, "Requesting a safe stop...\n"))
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
                    ("log", name, "Safe stop timed out; terminating process...\n")
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
    COLORS = {
        "canvas": "#f3f4f8",
        "surface": "#ffffff",
        "surface_alt": "#f8f8fb",
        "header": "#222333",
        "header_soft": "#303247",
        "text": "#222433",
        "muted": "#686b7c",
        "border": "#dfe1e8",
        "accent": "#c2385a",
        "accent_hover": "#a92d4b",
        "safe": "#4657a8",
        "safe_hover": "#37468e",
        "success": "#21875b",
        "warning": "#b36a19",
        "danger": "#b92743",
    }
    STATUS_COLORS = {
        "idle": "#7a7d8c",
        "running": "#21875b",
        "completed": "#4657a8",
        "stopped": "#7a7d8c",
        "failed": "#b92743",
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Touhou AI Control Center")
        self.root.geometry("1240x880")
        self.root.minsize(1040, 720)
        self.root.configure(background=self.COLORS["canvas"])
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
        self.mode_var = tk.StringVar(value=settings.get("mode", "defensive"))
        self.confidence_var = tk.StringVar(
            value=str(settings.get("confidence", "0.15"))
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
            self.append_log("GUI", f"Could not save GUI settings: {exc}\n")

    def _setup_style(self):
        style = ttk.Style()
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")
        font = "DejaVu Sans"
        colors = self.COLORS

        style.configure(".", font=(font, 10))
        style.configure("TFrame", background=colors["canvas"])
        style.configure(
            "Card.TFrame",
            background=colors["surface"],
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "SoftCard.TFrame",
            background=colors["surface_alt"],
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TLabel",
            background=colors["canvas"],
            foreground=colors["text"],
        )
        style.configure(
            "Card.TLabel",
            background=colors["surface"],
            foreground=colors["text"],
        )
        style.configure(
            "SoftCard.TLabel",
            background=colors["surface_alt"],
            foreground=colors["text"],
        )
        style.configure(
            "Title.TLabel",
            font=(font, 20, "bold"),
            foreground=colors["text"],
        )
        style.configure(
            "Heading.TLabel",
            font=(font, 12, "bold"),
            foreground=colors["text"],
        )
        style.configure(
            "Section.TLabel",
            font=(font, 14, "bold"),
            foreground=colors["text"],
        )
        style.configure("Muted.TLabel", foreground=colors["muted"])
        style.configure(
            "CardMuted.TLabel",
            background=colors["surface"],
            foreground=colors["muted"],
        )
        style.configure(
            "Status.TLabel",
            background=colors["surface"],
            font=(font, 9, "bold"),
        )
        style.configure(
            "Mode.TLabel",
            background=colors["surface"],
            font=(font, 12, "bold"),
        )
        style.configure(
            "TLabelframe",
            background=colors["surface"],
            bordercolor=colors["border"],
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=colors["canvas"],
            foreground=colors["text"],
            font=(font, 11, "bold"),
            padding=(4, 0),
        )
        style.configure(
            "TNotebook",
            background=colors["canvas"],
            borderwidth=0,
            tabmargins=(4, 4, 4, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background=colors["canvas"],
            foreground=colors["muted"],
            borderwidth=0,
            padding=(18, 10),
            font=(font, 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", colors["surface"]),
                ("active", colors["surface_alt"]),
            ],
            foreground=[
                ("selected", colors["accent"]),
                ("active", colors["text"]),
            ],
        )
        style.configure(
            "TButton",
            background=colors["surface_alt"],
            foreground=colors["text"],
            bordercolor=colors["border"],
            lightcolor=colors["surface_alt"],
            darkcolor=colors["surface_alt"],
            padding=(12, 8),
        )
        style.map(
            "TButton",
            background=[
                ("active", "#ececf3"),
                ("pressed", "#e4e5ed"),
                ("disabled", "#eeeef2"),
            ],
            foreground=[("disabled", "#a4a6b0")],
        )
        style.configure(
            "Accent.TButton",
            font=(font, 10, "bold"),
            foreground=colors["accent"],
        )
        style.configure(
            "Observe.TButton",
            font=(font, 12, "bold"),
            padding=(16, 13),
            foreground="white",
            background=colors["safe"],
            bordercolor=colors["safe"],
            lightcolor=colors["safe"],
            darkcolor=colors["safe"],
        )
        style.map(
            "Observe.TButton",
            background=[
                ("active", colors["safe_hover"]),
                ("pressed", colors["safe_hover"]),
                ("disabled", "#b5b7c3"),
            ],
            foreground=[("disabled", "#f1f1f3")],
        )
        style.configure(
            "AIStart.TButton",
            font=(font, 14, "bold"),
            padding=(18, 16),
            foreground="white",
            background=colors["accent"],
            bordercolor=colors["accent"],
            lightcolor=colors["accent"],
            darkcolor=colors["accent"],
        )
        style.map(
            "AIStart.TButton",
            background=[
                ("active", colors["accent_hover"]),
                ("pressed", colors["accent_hover"]),
                ("disabled", "#b5b7c3"),
            ],
            foreground=[("disabled", "#f1f1f3")],
        )
        style.configure(
            "Emergency.TButton",
            font=(font, 11, "bold"),
            padding=(12, 10),
            foreground=colors["danger"],
            background="#fff1f3",
            bordercolor="#efb7c1",
            lightcolor="#fff1f3",
            darkcolor="#fff1f3",
        )
        style.map(
            "Emergency.TButton",
            background=[
                ("active", "#ffe1e7"),
                ("pressed", "#ffd5de"),
                ("disabled", "#eeeef2"),
            ],
            foreground=[("disabled", "#a4a6b0")],
        )
        style.configure(
            "Treeview",
            background=colors["surface"],
            fieldbackground=colors["surface"],
            foreground=colors["text"],
            bordercolor=colors["border"],
            rowheight=28,
        )
        style.configure(
            "Treeview.Heading",
            background=colors["surface_alt"],
            foreground=colors["muted"],
            font=(font, 9, "bold"),
            padding=(6, 7),
        )
        style.map(
            "Treeview",
            background=[("selected", "#f4dce3")],
            foreground=[("selected", colors["accent_hover"])],
        )
        style.configure(
            "TEntry",
            fieldbackground=colors["surface_alt"],
            bordercolor=colors["border"],
            padding=(8, 6),
        )
        style.configure(
            "TCombobox",
            fieldbackground=colors["surface_alt"],
            bordercolor=colors["border"],
            padding=(8, 6),
        )
        style.configure(
            "TCheckbutton",
            background=colors["surface"],
            foreground=colors["text"],
        )

    def _build_ui(self):
        colors = self.COLORS
        header = tk.Frame(
            self.root,
            background=colors["header"],
            padx=24,
            pady=16,
        )
        header.pack(fill="x")

        tk.Label(
            header,
            text="TH",
            background=colors["accent"],
            foreground="white",
            font=("DejaVu Sans", 15, "bold"),
            padx=10,
            pady=6,
        ).pack(side="left", padx=(0, 14))

        title_group = tk.Frame(header, background=colors["header"])
        title_group.pack(side="left")
        tk.Label(
            title_group,
            text="Touhou AI",
            background=colors["header"],
            foreground="white",
            font=("DejaVu Sans", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_group,
            text="Embodiment of Scarlet Devil · Observe, run and review",
            background=colors["header"],
            foreground="#b9bbca",
            font=("DejaVu Sans", 9),
        ).pack(anchor="w")

        release = tk.Frame(header, background=colors["header"])
        release.pack(side="right")
        tk.Label(
            release,
            text="EXPERIMENTAL",
            background=colors["header_soft"],
            foreground="#f3bac8",
            font=("DejaVu Sans", 8, "bold"),
            padx=10,
            pady=3,
        ).pack(anchor="e")
        tk.Label(
            release,
            text="v1.1.0 · Final experimental build",
            background=colors["header"],
            foreground="#b9bbca",
            font=("DejaVu Sans", 9),
        ).pack(anchor="e", pady=(5, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(8, 16))

        self.control_tab = ttk.Frame(self.notebook, padding=12)
        self.sessions_tab = ttk.Frame(self.notebook, padding=12)
        self.logs_tab = ttk.Frame(self.notebook, padding=12)
        self.tools_tab = ttk.Frame(self.notebook, padding=12)
        self.roadmap_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.control_tab, text="  Control  ")
        self.notebook.add(self.sessions_tab, text="  Sessions  ")
        self.notebook.add(self.logs_tab, text="  Live Log  ")
        self.notebook.add(self.tools_tab, text="  Tools  ")
        self.notebook.add(self.roadmap_tab, text="  About  ")

        self._build_control_tab()
        self._build_sessions_tab()
        self._build_logs_tab()
        self._build_tools_tab()
        self._build_roadmap_tab()
        self.notebook.select(self.control_tab)

    def _build_control_tab(self):
        self.control_tab.columnconfigure(0, weight=1)
        self.control_tab.columnconfigure(1, weight=1)
        self.control_tab.rowconfigure(2, weight=1)

        primary = ttk.Frame(
            self.control_tab,
            style="Card.TFrame",
            padding=18,
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

        heading = ttk.Frame(primary, style="Card.TFrame")
        heading.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            heading,
            text="Start AI",
            style="Card.TLabel",
            font=("DejaVu Sans", 14, "bold"),
        ).pack(side="left")
        self.mode_status_label = ttk.Label(
            heading,
            text="● IDLE · No input is being sent",
            style="Mode.TLabel",
            foreground=self.COLORS["muted"],
        )
        self.mode_status_label.pack(side="right")
        ttk.Label(
            primary,
            text=(
                "Both modes use the same detection and decision pipeline. "
                "Only Control Mode sends input to the game."
            ),
            style="CardMuted.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 14))

        observe_card = ttk.Frame(primary, style="SoftCard.TFrame", padding=13)
        observe_card.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(
            observe_card,
            text="01  SAFE OBSERVATION",
            style="SoftCard.TLabel",
            font=("DejaVu Sans", 12, "bold"),
            foreground=self.COLORS["safe"],
        ).pack(anchor="w")
        ttk.Label(
            observe_card,
            text=(
                "Runs detection, tracking and action planning while keyboard "
                "and mouse input remain disabled."
            ),
            style="SoftCard.TLabel",
            wraplength=480,
            foreground=self.COLORS["muted"],
        ).pack(anchor="w", pady=(3, 8))
        self.observe_button = ttk.Button(
            observe_card,
            text="▶  Start Safe Observation",
            command=self.start_observe,
            style="Observe.TButton",
        )
        self.observe_button.pack(fill="x")

        ai_card = ttk.Frame(primary, style="SoftCard.TFrame", padding=13)
        ai_card.grid(row=2, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(
            ai_card,
            text="02  CONTROL MODE",
            style="SoftCard.TLabel",
            font=("DejaVu Sans", 12, "bold"),
            foreground=self.COLORS["accent"],
        ).pack(anchor="w")
        ttk.Label(
            ai_card,
            text=(
                "Sends movement, fire and bomb actions to the game window. "
                "Complete window and input checks first."
            ),
            style="SoftCard.TLabel",
            wraplength=480,
            foreground=self.COLORS["muted"],
        ).pack(anchor="w", pady=(3, 8))
        self.ai_button = ttk.Button(
            ai_card,
            text="▶  Start AI Control",
            command=self.start_ai,
            style="AIStart.TButton",
        )
        self.ai_button.pack(fill="x")

        self.stop_modes_button = ttk.Button(
            primary,
            text="■  STOP AI AND RELEASE ALL KEYS",
            command=self.stop_control_modes,
            style="Emergency.TButton",
            state="disabled",
        )
        self.stop_modes_button.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(14, 0),
        )

        status_frame = ttk.Frame(
            self.control_tab,
            style="Card.TFrame",
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
                ("game", "Game"),
                ("locate", "Window"),
                ("observe", "Observation"),
                ("ai", "AI Control"),
                ("control-test", "Input Test"),
            )
        ):
            card = ttk.Frame(status_frame, style="Card.TFrame", padding=7)
            card.grid(row=0, column=column, sticky="ew", padx=4)
            status_frame.columnconfigure(column, weight=1)
            ttk.Label(
                card,
                text=label,
                style="CardMuted.TLabel",
            ).pack()
            status = ttk.Label(
                card,
                text="● Idle",
                style="Status.TLabel",
                foreground=self.STATUS_COLORS["idle"],
            )
            status.pack(pady=(4, 0))
            self.status_labels[name] = status

        workflow = ttk.Frame(
            self.control_tab,
            style="Card.TFrame",
            padding=14,
        )
        workflow.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        ttk.Label(
            workflow,
            text="First-run checklist",
            style="Card.TLabel",
            font=("DejaVu Sans", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            workflow,
            text="Complete these three steps before enabling AI control.",
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(2, 8))

        ttk.Button(
            workflow,
            text="1  Launch game (vpatch)",
            command=self.start_game,
            style="Accent.TButton",
        ).pack(fill="x", pady=4)
        ttk.Button(
            workflow,
            text="2  Check game window",
            command=lambda: self.start_process("locate", ["locate"]),
        ).pack(fill="x", pady=4)
        ttk.Button(
            workflow,
            text="3  Test left / right input",
            command=lambda: self.start_process(
                "control-test",
                ["control-test"],
            ),
        ).pack(fill="x", pady=4)
        ttk.Label(
            workflow,
            text=(
                "Tip: verify the detection result with Safe Observation "
                "before running the input test."
            ),
            wraplength=430,
            style="CardMuted.TLabel",
        ).pack(anchor="w", pady=(12, 0))

        config = ttk.Frame(
            self.control_tab,
            style="Card.TFrame",
            padding=14,
        )
        config.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        config.columnconfigure(1, weight=1)
        config.columnconfigure(3, weight=1)
        ttk.Label(
            config,
            text="Runtime settings",
            style="Card.TLabel",
            font=("DejaVu Sans", 14, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(
            config,
            text="Defaults are tuned for the current Stage 1 experiment.",
            style="CardMuted.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 8))

        ttk.Label(
            config,
            text="AI profile",
            style="Card.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(
            config,
            textvariable=self.mode_var,
            values=("defensive", "balanced", "aggressive"),
            state="readonly",
        ).grid(
            row=2,
            column=1,
            columnspan=3,
            sticky="ew",
            pady=4,
            padx=(6, 0),
        )

        ttk.Label(config, text="Detection confidence", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=4
        )
        ttk.Entry(config, textvariable=self.confidence_var).grid(
            row=3, column=1, sticky="ew", pady=4, padx=(6, 12)
        )
        ttk.Label(config, text="Risk sensitivity", style="Card.TLabel").grid(
            row=3, column=2, sticky="w", pady=4
        )
        ttk.Entry(config, textvariable=self.sensitivity_var).grid(
            row=3, column=3, sticky="ew", pady=4, padx=(6, 0)
        )
        ttk.Label(
            config,
            text="Safe margin (px)",
            style="Card.TLabel",
        ).grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.safe_margin_var).grid(
            row=4, column=1, sticky="ew", pady=4, padx=(6, 12)
        )
        ttk.Label(
            config,
            text="Lost timeout (sec)",
            style="Card.TLabel",
        ).grid(row=4, column=2, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.player_lost_timeout_var).grid(
            row=4, column=3, sticky="ew", pady=4, padx=(6, 0)
        )
        ttk.Checkbutton(
            config,
            text="Record session and sample frames",
            variable=self.record_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(7, 4))
        ttk.Label(config, text="Recording FPS", style="Card.TLabel").grid(
            row=5, column=2, sticky="w", pady=(7, 4)
        )
        ttk.Entry(config, textvariable=self.record_fps_var).grid(
            row=5, column=3, sticky="ew", pady=(7, 4), padx=(6, 0)
        )
        ttk.Checkbutton(
            config,
            text="Show observation preview",
            variable=self.observe_preview_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(
            config,
            text="Allow vertical planning",
            variable=self.vertical_movement_var,
        ).grid(row=6, column=2, columnspan=2, sticky="w", pady=4)

        help_frame = ttk.Frame(
            self.control_tab,
            style="SoftCard.TFrame",
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
                "SAFETY  ·  Control Mode keeps focus on the game window. "
                "Press STOP before switching to another application."
            ),
            style="SoftCard.TLabel",
            wraplength=950,
            foreground=self.COLORS["warning"],
        ).pack(anchor="w")

    def _build_sessions_tab(self):
        self.sessions_tab.rowconfigure(1, weight=1)
        self.sessions_tab.columnconfigure(0, weight=3)
        self.sessions_tab.columnconfigure(1, weight=2)

        title = ttk.Frame(
            self.sessions_tab,
            style="Card.TFrame",
            padding=(16, 12),
        )
        title.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 12),
        )
        ttk.Label(
            title,
            text="Session Review",
            style="Card.TLabel",
            font=("DejaVu Sans", 14, "bold"),
        ).pack(side="left")
        ttk.Label(
            title,
            text="Review detection rate, collision risk, actions and frames",
            style="CardMuted.TLabel",
        ).pack(side="left", padx=14)

        left = ttk.Frame(
            self.sessions_tab,
            style="Card.TFrame",
            padding=12,
        )
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(left, style="Card.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(
            toolbar,
            text="↻ Refresh",
            command=self.refresh_sessions,
        ).pack(side="left")
        ttk.Button(
            toolbar,
            text="Build Report",
            command=self.generate_current_report,
        ).pack(side="left", padx=6)
        ttk.Button(
            toolbar,
            text="Analyze",
            command=self.analyze_current_session,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            toolbar,
            text="Export Review Set",
            command=self.export_current_review_dataset,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            toolbar,
            text="Open Folder",
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
        self.session_tree.heading("#0", text="Session")
        self.session_tree.column("#0", width=210)
        headings = {
            "source": "Source",
            "status": "Status",
            "duration": "Sec",
            "events": "Events",
            "frames": "Frames",
            "player_rate": "Player",
            "danger": "Danger",
            "risk": "Risk",
            "switches": "Turns",
        }
        for name in columns:
            self.session_tree.heading(name, text=headings[name])
            self.session_tree.column(name, width=75, anchor="center")
        self.session_tree.grid(row=1, column=0, sticky="nsew")
        self.session_tree.bind("<<TreeviewSelect>>", self.on_session_select)

        right = ttk.Frame(
            self.sessions_tab,
            style="Card.TFrame",
            padding=12,
        )
        right.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        right.rowconfigure(1, weight=3)
        right.rowconfigure(3, weight=2)
        right.columnconfigure(0, weight=1)

        ttk.Label(
            right,
            text="Sample Frame Review",
            style="Card.TLabel",
            font=("DejaVu Sans", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.frame_label = ttk.Label(
            right,
            text="Select a session to begin",
            anchor="center",
            style="SoftCard.TLabel",
        )
        self.frame_label.grid(row=1, column=0, sticky="nsew", pady=6)

        navigation = ttk.Frame(right, style="Card.TFrame")
        navigation.grid(row=2, column=0, sticky="ew")
        ttk.Button(
            navigation,
            text="◀ Previous",
            command=lambda: self.change_frame(-1),
        ).pack(side="left")
        self.frame_counter = ttk.Label(navigation, text="0 / 0")
        self.frame_counter.configure(style="CardMuted.TLabel")
        self.frame_counter.pack(side="left", expand=True)
        ttk.Button(
            navigation,
            text="Next ▶",
            command=lambda: self.change_frame(1),
        ).pack(side="right")

        self.session_details = scrolledtext.ScrolledText(
            right,
            height=12,
            wrap="word",
            font=("Monospace", 9),
            background=self.COLORS["surface_alt"],
            foreground=self.COLORS["text"],
            insertbackground=self.COLORS["text"],
            relief="flat",
            borderwidth=1,
            padx=10,
            pady=10,
        )
        self.session_details.grid(row=3, column=0, sticky="nsew", pady=(8, 0))

    def _build_logs_tab(self):
        toolbar = ttk.Frame(
            self.logs_tab,
            style="Card.TFrame",
            padding=(16, 10),
        )
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Label(
            toolbar,
            text="Live Log",
            style="Card.TLabel",
            font=("DejaVu Sans", 14, "bold"),
        ).pack(side="left")
        ttk.Label(
            toolbar,
            text="Commands, runtime status and errors appear here",
            style="CardMuted.TLabel",
        ).pack(side="left", padx=14)
        ttk.Button(
            toolbar,
            text="Clear Log",
            command=lambda: self.log_text.delete("1.0", "end"),
        ).pack(side="right")
        self.log_text = scrolledtext.ScrolledText(
            self.logs_tab,
            wrap="word",
            font=("Monospace", 10),
            bg="#191a26",
            fg="#e8e8ef",
            insertbackground="white",
            relief="flat",
            padx=14,
            pady=12,
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_configure("system", foreground="#9ca9ef")
        self.log_text.tag_configure("error", foreground="#ff9caf")

    def _build_tools_tab(self):
        self.tools_tab.columnconfigure(0, weight=1)
        self.tools_tab.columnconfigure(1, weight=1)

        header = ttk.Frame(
            self.tools_tab,
            style="Card.TFrame",
            padding=(16, 12),
        )
        header.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 12),
        )
        ttk.Label(
            header,
            text="Maintenance Tools",
            style="Card.TLabel",
            font=("DejaVu Sans", 14, "bold"),
        ).pack(side="left")
        ttk.Label(
            header,
            text="Diagnostics and dataset utilities for this experimental build",
            style="CardMuted.TLabel",
        ).pack(side="left", padx=14)

        diagnostics = ttk.Frame(
            self.tools_tab,
            style="Card.TFrame",
            padding=18,
        )
        diagnostics.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        ttk.Label(
            diagnostics,
            text="Diagnostics",
            style="Card.TLabel",
            font=("DejaVu Sans", 13, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            diagnostics,
            text=(
                "Check dependencies and run the complete test suite. "
                "Neither action sends input to the game."
            ),
            style="CardMuted.TLabel",
            wraplength=480,
        ).pack(anchor="w", pady=(3, 14))
        ttk.Button(
            diagnostics,
            text="Run Environment Check",
            command=lambda: self.start_process("check", ["check"]),
            style="Accent.TButton",
        ).pack(fill="x", pady=5)
        ttk.Button(
            diagnostics,
            text="Run All Tests (No Input)",
            command=lambda: self.start_process("test", ["test"]),
        ).pack(fill="x", pady=5)

        data_tools = ttk.Frame(
            self.tools_tab,
            style="Card.TFrame",
            padding=18,
        )
        data_tools.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        ttk.Label(
            data_tools,
            text="Recordings & Model",
            style="Card.TLabel",
            font=("DejaVu Sans", 13, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            data_tools,
            text=(
                "Download or verify the published YOLO model, open local "
                "session data, or evaluate a reviewed validation set."
            ),
            style="CardMuted.TLabel",
            wraplength=480,
        ).pack(anchor="w", pady=(3, 14))
        ttk.Button(
            data_tools,
            text="Download / Verify YOLO Model",
            command=lambda: self.start_process("model", ["model"]),
            style="Accent.TButton",
        ).pack(fill="x", pady=5)
        ttk.Button(
            data_tools,
            text="Open Recordings Folder",
            command=lambda: self.open_path(DEFAULT_SESSIONS_DIR),
        ).pack(fill="x", pady=5)
        ttk.Button(
            data_tools,
            text="Select Dataset and Evaluate Model",
            command=self.start_model_evaluation,
        ).pack(fill="x", pady=5)

    def _build_roadmap_tab(self):
        header = ttk.Frame(
            self.roadmap_tab,
            style="Card.TFrame",
            padding=(16, 12),
        )
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(
            header,
            text="About This Build",
            style="Card.TLabel",
            font=("DejaVu Sans", 14, "bold"),
        ).pack(side="left")
        ttk.Label(
            header,
            text="The control algorithm is frozen; this build focuses on a clean release.",
            style="CardMuted.TLabel",
        ).pack(side="left", padx=14)
        text = scrolledtext.ScrolledText(
            self.roadmap_tab,
            wrap="word",
            font=("DejaVu Sans", 10),
            background=self.COLORS["surface"],
            foreground=self.COLORS["text"],
            selectbackground="#f4dce3",
            selectforeground=self.COLORS["accent_hover"],
            relief="flat",
            padx=22,
            pady=18,
            spacing1=2,
            spacing3=4,
        )
        text.pack(fill="both", expand=True)
        content = """TOUHOU AI  ·  v1.1.0 EXPERIMENTAL

PURPOSE
An unofficial computer-vision and rule-based control experiment for
Touhou Koumakyou: the Embodiment of Scarlet Devil.

CURRENT CAPABILITIES
• Launch and locate the game window on Linux X11
• Detect the player, bullets, enemies, bosses and items with YOLO
• Track short player dropouts and estimate bullet trajectories
• Plan eight-direction movement with edge and collision protection
• Run the full AI safely without sending any keyboard input
• Record sessions, inspect sample frames and export review candidates

PROJECT STATUS
The end-to-end workflow is operational and the AI can reach the first boss
in the current test setup. It is not a reliable stage-clear or game-clear
system. Dense spell-card patterns, lasers and imperfect detections remain
known limitations.

This is the final experimental baseline. Future work would require a new,
carefully reviewed dataset and a substantially different control approach,
not more parameter tuning.

SAFETY
Control Mode focuses the game window and sends real keyboard events.
Always stop the AI before using another application.

LEGAL
This is an unofficial fan-made technical experiment. It is not affiliated
with, endorsed by, or sponsored by Team Shanghai Alice, ZUN, or the Touhou
Project. The game itself is not included.
"""
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
                "Invalid settings",
                "Confidence, sensitivity, FPS, margin and timeout must be numeric.",
            )
            return None
        if not 0 <= confidence <= 1 or not 0 <= sensitivity <= 1:
            messagebox.showerror(
                "Invalid settings",
                "Confidence and sensitivity must be between 0 and 1.",
            )
            return None
        if not 0 <= record_fps <= 20:
            messagebox.showerror(
                "Invalid settings",
                "Recording FPS must be between 0 and 20.",
            )
            return None
        if not 8 <= safe_margin <= 160:
            messagebox.showerror(
                "Invalid settings",
                "Safe edge margin must be between 8 and 160 pixels.",
            )
            return None
        if not 0.35 <= player_lost_timeout <= 3:
            messagebox.showerror(
                "Invalid settings",
                "Player-lost timeout must be between 0.35 and 3 seconds.",
            )
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
                text="● STOPPING · Releasing input…",
                foreground=self.COLORS["warning"],
            )
        else:
            self.update_mode_banner()

    def update_mode_banner(self):
        ai_running = self.process_states.get("ai") == "running"
        observe_running = self.process_states.get("observe") == "running"
        if ai_running:
            self.mode_status_label.configure(
                text="● CONTROL ACTIVE · Sending input to the game",
                foreground=self.COLORS["danger"],
            )
            self.ai_button.configure(state="disabled")
            self.observe_button.configure(state="disabled")
            self.stop_modes_button.configure(state="normal")
        elif observe_running:
            self.mode_status_label.configure(
                text="● OBSERVING · AI is running without input",
                foreground=self.COLORS["safe"],
            )
            self.ai_button.configure(state="normal")
            self.observe_button.configure(state="disabled")
            self.stop_modes_button.configure(state="normal")
        else:
            self.mode_status_label.configure(
                text="● IDLE · No input is being sent",
                foreground=self.COLORS["muted"],
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
                "Game files not found",
                "The game is not included. Follow game/README.md and copy "
                "your legally obtained game files into the game folder.",
            )
            return
        self.start_process("game", ["game"])

    def start_model_evaluation(self):
        data_yaml = filedialog.askopenfilename(
            title="Select YOLO dataset data.yaml",
            filetypes=(("YAML", "*.yaml *.yml"), ("All files", "*")),
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
                "Stop observation?",
                "The observation preview may cover the game. Stop observation "
                "before starting AI Control?",
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
        tag = (
            "error"
            if "❌" in line or "失败" in line or "error" in line.lower()
            else "system"
        )
        self.log_text.insert("end", prefix, tag)
        self.log_text.insert("end", line)
        self.log_text.see("end")

    def set_status(self, name: str, state: str):
        self.process_states[name] = state
        label = self.status_labels.get(name)
        if label:
            display = {
                "idle": "● Idle",
                "running": "● Running",
                "completed": "● Complete",
                "stopped": "● Stopped",
                "failed": "● Failed",
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
            self.frame_label.configure(image="", text="No saved frames in this session")
            self.frame_counter.configure(text="0 / 0")
            return
        path, event = self.current_frames[self.current_frame_index]
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:
            self.frame_label.configure(image="", text=f"Could not read frame: {exc}")
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
            messagebox.showinfo("Session report", "Select a session first.")
            return
        try:
            json_path, markdown_path = write_report(self.current_session_dir)
        except Exception as exc:
            messagebox.showerror("Report failed", str(exc))
            return
        messagebox.showinfo(
            "Report complete",
            f"Created:\n{json_path}\n{markdown_path}",
        )

    def analyze_current_session(self):
        if not self.current_session_dir:
            messagebox.showinfo("Session analysis", "Select a session first.")
            return
        try:
            json_path, markdown_path, review_path = write_analysis(
                self.current_session_dir
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Analysis failed", str(exc))
            return
        self.append_log(
            "analysis",
            (
                f"Created {json_path.name}, {markdown_path.name} and "
                f"{review_path.name}\n"
            ),
        )
        messagebox.showinfo(
            "Analysis complete",
            f"Created:\n{json_path}\n{markdown_path}\n{review_path}",
        )

    def export_current_review_dataset(self):
        if not self.current_session_dir:
            messagebox.showinfo("Review candidates", "Select a session first.")
            return
        try:
            output_dir = export_review_dataset(self.current_session_dir)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self.append_log("analysis", f"Review candidates exported: {output_dir}\n")
        messagebox.showinfo(
            "Export complete",
            "Predicted labels must be reviewed manually before training or evaluation.\n"
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
            messagebox.showerror("Could not open folder", str(exc))

    def open_current_session(self):
        if not self.current_session_dir:
            messagebox.showinfo("Open folder", "Select a session first.")
            return
        self.open_path(self.current_session_dir)

    def on_close(self):
        active = [
            name
            for name in ("observe", "ai")
            if self.process_manager.is_running(name)
        ]
        if active and not messagebox.askyesno(
            "Exit Touhou AI?",
            f"Still running: {', '.join(active)}.\nStop safely and exit?",
        ):
            return
        self.save_settings()
        self.process_manager.stop_all()
        self.root.after(700, self.root.destroy)

    def run(self):
        self.append_log("GUI", f"Project directory: {ROOT}\n")
        self.append_log("GUI", "Control Center is ready.\n")
        self.root.mainloop()


def main() -> int:
    try:
        application = TouhouControlCenter()
        application.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except tk.TclError as exc:
        print(f"❌ Could not start desktop GUI: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
