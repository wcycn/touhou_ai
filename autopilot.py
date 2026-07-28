#!/usr/bin/env python3
"""东方红魔乡视觉检测与规则控制核心。"""

import os
import sys
import time
import signal
import threading
import cv2
import numpy as np
import mss
import pyautogui
from pathlib import Path
import argparse

# 添加项目路径
def setup_project_paths():
    """设置项目路径"""
    project_dir = Path(__file__).parent.resolve()
    project_dir_str = str(project_dir)
    if project_dir_str not in sys.path:
        sys.path.insert(0, project_dir_str)

    os.environ['TOUHOU_AI_ROOT'] = project_dir_str
    os.environ['TOUHOU_AI_GAME_DIR'] = str(project_dir / "game")
    ultralytics_config_dir = project_dir / "runs" / "ultralytics"
    ultralytics_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config_dir)

    return project_dir

project_dir = setup_project_paths()

from session_recorder import SessionRecorder
from control_logic import (
    ActionStabilizer,
    BulletTracker,
    GameSceneStateMachine,
    InputStateMachine,
    PlayerTracker,
    RiskPlanner,
    collision_metrics,
)
from inference_device import select_inference_device

PLAYER_FALLBACK_CONFIDENCE = 0.03
PLAYER_FALLBACK_MIN_Y_RATIO = 0.68
PLAYER_FALLBACK_MAX_X_RATIO = 0.70

# YOLO推理由Ultralytics/PyTorch负责。
try:
    from ultralytics import YOLO, settings
    import torch

    settings.update(
        {
            "datasets_dir": str(project_dir / "datasets"),
            "weights_dir": str(project_dir / "runs" / "ultralytics" / "weights"),
            "runs_dir": str(project_dir / "runs"),
        }
    )
    INFERENCE_DEVICE = select_inference_device(torch)
    print("✅ YOLO模型库加载成功")
    print(
        f"✅ PyTorch版本: {torch.__version__} "
        f"(YOLO后端，设备: {INFERENCE_DEVICE.label})"
    )
    if INFERENCE_DEVICE.reason:
        print(f"⚠️ CUDA未启用，自动回退CPU: {INFERENCE_DEVICE.reason}")

except ImportError as e:
    print(f"❌ 必要库导入失败: {e}")
    sys.exit(1)

try:
    from window_controller import WindowController
    WINDOW_CONTROLLER_AVAILABLE = True
    print("✅ X11游戏窗口定位器导入成功")
except Exception as e:
    WINDOW_CONTROLLER_AVAILABLE = False
    print(f"⚠️ X11游戏窗口定位器不可用: {e}")

class TouhouAIController:
    """基于YOLO检测和规则决策的游戏控制器。"""

    def __init__(
        self,
        allow_no_game=False,
        record_session=False,
        record_dir=None,
        record_fps=2.0,
        safe_margin=36,
        player_lost_timeout=0.70,
        allow_vertical=True,
    ):
        # 设置PyAutoGUI
        pyautogui.PAUSE = 0.01
        pyautogui.FAILSAFE = True

        self.allow_no_game = allow_no_game
        self.record_session = record_session
        self.record_dir = Path(record_dir).expanduser() if record_dir else None
        self.record_fps = record_fps
        self.session_recorder = None
        self._cleanup_complete = False
        self.safe_margin = max(8, int(safe_margin))
        self.player_lost_timeout = max(0.35, float(player_lost_timeout))
        self.allow_vertical = bool(allow_vertical)

        # 截图区域
        self.screen_region = None
        self.last_window_region_refresh = 0.0
        self.window_region_refresh_interval = 1.0
        self.auto_focus_game = True
        self.last_focus_check = 0.0
        self.focus_check_interval = 0.20
        self.game_focus_ok = False

        # AI参数
        self.model_path = str(project_dir / "models" / "best.pt")
        self.confidence_threshold = 0.15
        self.bullet_threshold = 0.05
        self.detection_interval = 0.1

        # 游戏进程监控
        self.game_process_monitor_thread = None
        self.game_process_monitoring = False
        self.game_process_names = [
            "東方紅魔郷.exe", "th06.exe", "th06c.exe",
            "vpatch.exe"  # vpatch可能也会被使用
        ]
        self.last_game_check_time = 0
        self.game_check_interval = 2.0  # 每2秒检查一次游戏进程

        # 状态变量
        self.running = False
        self.paused = False
        self.auto_bomb = True
        self.ai_mode = "balanced"

        self.last_bomb_time = float("-inf")
        self.bomb_cooldown = 4.0

        # 统计信息
        self.stats = {
            'start_time': None,
            'game_launch_time': None,
            'ai_start_time': None,
            'detections': 0,
            'movements': 0,
            'shots_fired': 0,
            'bombs_used': 0,
            'dodge_count': 0,
            'direction_switches': 0,
            'blocked_switches': 0,
            'player_predicted_frames': 0,
            'safe_stop_frames': 0,
            'collision_warning_frames': 0,
            'focus_loss_frames': 0,
            'key_down_events': 0,
            'key_up_events': 0,
        }

        print("✅ 输入后端: PyAutoGUI")

        # 窗口定位始终独立初始化。
        if WINDOW_CONTROLLER_AVAILABLE:
            try:
                self.window_controller = WindowController()
            except Exception as e:
                print(f"⚠️ 窗口定位器初始化失败: {e}")
                self.window_controller = None
        else:
            self.window_controller = None

        self.model = None
        self.inference_device = INFERENCE_DEVICE.value
        self.last_detection_time = 0

        # 线程锁
        self.action_lock = threading.Lock()
        self.player_tracker = PlayerTracker(
            prediction_timeout=min(0.35, self.player_lost_timeout),
            stop_timeout=self.player_lost_timeout,
        )
        self.bullet_tracker = BulletTracker()
        self.risk_planner = RiskPlanner(
            safe_margin=self.safe_margin,
            allow_vertical=self.allow_vertical,
        )
        self.action_stabilizer = ActionStabilizer(
            safe_margin=self.safe_margin,
        )
        self.scene_machine = GameSceneStateMachine()
        self.input_state = InputStateMachine(pyautogui)
        self.last_input_transition = {
            "pressed": [],
            "released": [],
            "held": [],
        }

        print("🎯 Touhou AI控制器初始化")
        print("=" * 50)
        print(f"🔥 PyTorch版本: {torch.__version__} (YOLO后端)")
        print("📦 Ultralytics: 已加载")

    def is_game_process_running(self):
        """检查游戏进程是否在运行"""
        import subprocess

        for process_name in self.game_process_names:
            try:
                # 检查进程是否存在
                result = subprocess.run(['pgrep', '-f', process_name],
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    return True, process_name
            except Exception:
                continue

        return False, None

    def monitor_game_process(self):
        """监控游戏进程，如果游戏退出则自动停止AI"""
        print("🔍 开始监控游戏进程...")

        while self.game_process_monitoring and self.running:
            current_time = time.time()

            # 每隔一段时间检查一次
            if current_time - self.last_game_check_time >= self.game_check_interval:
                game_running, process_name = self.is_game_process_running()

                if not game_running:
                    print(f"⚠️ 游戏进程已退出！自动停止AI控制...")
                    print(f"🔓 释放键盘控制...")

                    # 立即停止AI并释放键盘
                    self.stop(reason="game_exited")
                    break

                self.last_game_check_time = current_time

            time.sleep(0.5)

        print("🔍 游戏进程监控结束")

    def start_game_process_monitor(self):
        """启动游戏进程监控"""
        if not self.game_process_monitoring:
            self.game_process_monitoring = True
            self.game_process_monitor_thread = threading.Thread(
                target=self.monitor_game_process,
                daemon=True
            )
            self.game_process_monitor_thread.start()
            print("✅ 游戏进程监控已启动")

    def stop_game_process_monitor(self):
        """停止游戏进程监控"""
        if self.game_process_monitoring:
            self.game_process_monitoring = False
            if (
                self.game_process_monitor_thread
                and threading.current_thread() is not self.game_process_monitor_thread
            ):
                self.game_process_monitor_thread.join(timeout=2)
            print("🛑 游戏进程监控已停止")

    def setup_signal_handlers(self):
        """设置信号处理器"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n📡 收到信号 {signum}，正在安全退出...")
        self.stop(reason=f"signal_{signum}")

    def load_model(self):
        """加载YOLO模型。"""
        try:
            # 尝试加载YOLO模型，如果失败则使用简化检测
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ YOLO模型加载成功: {self.model_path}")
                if hasattr(self.model, 'names'):
                    print(f"📊 模型类别: {len(self.model.names)} 种")
            else:
                print(f"❌ 模型文件不存在: {self.model_path}")
                print("💡 为避免伪检测驱动键盘，自动控制不会启用轮廓回退")
                self.model = None
                return False

            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 为避免错误输入，拒绝在无模型状态下继续")
            self.model = None
            return False

    def check_dependencies(self):
        """检查依赖项"""
        print("🔍 检查运行环境...")

        try:
            # 检查游戏进程
            import subprocess
            process_names = ['th06c.exe', 'th06', 'wine.*th06']
            found_process = False

            for name in process_names:
                result = subprocess.run(['pgrep', '-f', name], capture_output=True)
                if result.returncode == 0:
                    print(f"✅ 检测到游戏进程: {name}")
                    found_process = True
                    break

            if not found_process:
                print("⚠️ 未检测到游戏进程，但继续运行...")
                print("💡 如果游戏正在运行，AI仍可能正常工作")

            return True

        except Exception as e:
            print(f"⚠️ 依赖检查失败: {e}")
            print("💡 继续运行，AI会尝试直接截图")
            return True

    def detect_game(self):
        """检测游戏窗口和区域"""
        try:
            # 使用窗口控制器或使用默认值
            if self.window_controller:
                if self.window_controller.find_game_window():
                    screenshot_region = self.window_controller.get_screenshot_region()
                    if screenshot_region and len(screenshot_region) >= 4:
                        self.screen_region = {
                            'top': screenshot_region[1],
                            'left': screenshot_region[0],
                            'width': screenshot_region[2],
                            'height': screenshot_region[3]
                        }
                        self.game_found = True
                        print(f"✅ 检测到游戏窗口: {self.screen_region}")
                        return True

            # 如果没有找到游戏窗口，使用默认区域
            self.screen_region = {
                'top': 100, 'left': 100,
                'width': 640, 'height': 480
            }
            self.game_found = False
            print("⚠️ 使用默认截图区域")
            return True

        except Exception as e:
            print(f"⚠️ 游戏检测失败: {e}")
            self.screen_region = {
                'top': 100, 'left': 100,
                'width': 640, 'height': 480
            }
            self.game_found = False
            return False

    def refresh_game_region(self):
        """窗口移动或缩放后刷新截图区域。"""
        if not self.window_controller:
            return False
        try:
            if not self.window_controller.refresh():
                return False
            screenshot_region = self.window_controller.get_screenshot_region()
            if not screenshot_region or len(screenshot_region) < 4:
                return False
            new_region = {
                'top': screenshot_region[1],
                'left': screenshot_region[0],
                'width': screenshot_region[2],
                'height': screenshot_region[3]
            }
            if new_region != self.screen_region:
                print(f"🪟 游戏窗口位置已更新: {new_region}")
                self.screen_region = new_region
            self.game_found = True
            return True
        except Exception as e:
            print(f"⚠️ 刷新游戏窗口位置失败: {e}")
            return False

    def ensure_game_focus(self, force=False):
        """确保全局按键只会在游戏窗口激活时发送。"""
        if not self.window_controller or not self.game_found:
            return self.allow_no_game

        current_time = time.monotonic()
        if (
            not force
            and current_time - self.last_focus_check < self.focus_check_interval
        ):
            return self.game_focus_ok
        self.last_focus_check = current_time

        try:
            if self.window_controller.is_game_window_active():
                self.game_focus_ok = True
                return True

            self.game_focus_ok = False
            if not self.auto_focus_game:
                return False

            if self.window_controller.activate_window():
                print("🎯 已将输入焦点切换到游戏窗口")
                self.game_focus_ok = True
                return True

            print("⚠️ 无法激活游戏窗口，暂停发送按键")
            return False
        except Exception as e:
            self.game_focus_ok = False
            print(f"⚠️ 游戏窗口焦点检查失败: {e}")
            return False

    def take_screenshot(self):
        """截图功能"""
        try:
            if self.screen_region is None:
                self.detect_game()

            current_time = time.monotonic()
            if (
                self.window_controller
                and current_time - self.last_window_region_refresh
                >= self.window_region_refresh_interval
            ):
                self.refresh_game_region()
                self.last_window_region_refresh = current_time

            if not isinstance(self.screen_region, dict):
                print(f"❌ 截图区域格式错误: {type(self.screen_region)}")
                self.screen_region = {
                    'top': 0, 'left': 0,
                    'width': 640, 'height': 480
                }

            if not hasattr(self, '_last_printed_region') or self._last_printed_region != self.screen_region:
                print(f"🔍 使用截图区域: {self.screen_region}")
                self._last_printed_region = self.screen_region.copy()

            with mss.mss() as sct:
                screenshot = sct.grab(self.screen_region)
                image = np.array(screenshot)
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                return image
        except Exception as e:
            print(f"❌ 截图失败: {e}")
            return None

    def detect_objects(self, image):
        """检测游戏物体。"""
        if image is None:
            return []

        try:
            if self.model is not None:
                # 显式传入阈值和设备，避免 Ultralytics 自动选择到不兼容的 CUDA，
                # 也避免其默认 conf=0.25 吞掉用户要求的低置信度结果。
                try:
                    results = self.model.predict(
                        image,
                        verbose=False,
                        conf=min(
                            self.confidence_threshold,
                            PLAYER_FALLBACK_CONFIDENCE,
                        ),
                        device=self.inference_device,
                    )
                except Exception as inference_error:
                    if self.inference_device == "cpu":
                        raise
                    print(
                        "⚠️ CUDA推理失败，当前会话自动切换到CPU: "
                        f"{inference_error}"
                    )
                    self.inference_device = "cpu"
                    results = self.model.predict(
                        image,
                        verbose=False,
                        conf=min(
                            self.confidence_threshold,
                            PLAYER_FALLBACK_CONFIDENCE,
                        ),
                        device="cpu",
                    )
                detections = []

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            try:
                                # 处理张量数据
                                if hasattr(box.xyxy[0], 'cpu'):
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                    conf = box.conf[0].cpu().numpy()
                                    cls = int(box.cls[0].cpu().numpy())
                                else:
                                    xyxy = box.xyxy[0]
                                    if hasattr(xyxy, 'asnumpy'):
                                        xyxy_np = xyxy.asnumpy()
                                        conf_np = box.conf[0].asnumpy()
                                        cls_np = box.cls[0].asnumpy()
                                        x1, y1, x2, y2 = xyxy_np
                                        conf = float(conf_np)
                                        cls = int(cls_np)
                                    else:
                                        x1, y1, x2, y2 = xyxy
                                        conf = float(box.conf[0])
                                        cls = int(box.cls[0])

                                class_name = (
                                    self.model.names[cls]
                                    if hasattr(self.model, 'names')
                                    else f"object_{cls}"
                                )
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                box_width = int(x2 - x1)
                                box_height = int(y2 - y1)
                                image_height, image_width = image.shape[:2]
                                fallback_candidate = (
                                    class_name == "enemy_small_red"
                                    and float(conf)
                                    >= PLAYER_FALLBACK_CONFIDENCE
                                    and center_y
                                    >= image_height * PLAYER_FALLBACK_MIN_Y_RATIO
                                    and center_x
                                    <= image_width * PLAYER_FALLBACK_MAX_X_RATIO
                                    and 6 <= box_width <= 45
                                    and 6 <= box_height <= 45
                                )
                                if (
                                    conf < self.confidence_threshold
                                    and not fallback_candidate
                                ):
                                    continue

                                detection = {
                                    'bbox': [x1, y1, x2, y2],
                                    'x': int(x1),
                                    'y': int(y1),
                                    'center_x': center_x,
                                    'center_y': center_y,
                                    'width': box_width,
                                    'height': box_height,
                                    'confidence': float(conf),
                                    'class_id': cls,
                                    'class_name': (
                                        "character_fallback"
                                        if fallback_candidate
                                        else class_name
                                    ),
                                    'model_class_name': class_name,
                                    'player_fallback_candidate': (
                                        fallback_candidate
                                    ),
                                }
                                detections.append(detection)

                            except Exception as tensor_error:
                                continue

                self.stats['detections'] += len(detections)
                return detections
            else:
                return []

        except Exception as e:
            print(f"❌ 检测失败: {e}")
            return []

    def _ensure_control_components(self):
        """兼容纯逻辑测试通过 ``__new__`` 构造的轻量实例。"""
        safe_margin = getattr(self, "safe_margin", 36)
        lost_timeout = getattr(self, "player_lost_timeout", 0.70)
        allow_vertical = getattr(self, "allow_vertical", True)
        if not hasattr(self, "player_tracker"):
            self.player_tracker = PlayerTracker(
                prediction_timeout=min(0.35, lost_timeout),
                stop_timeout=lost_timeout,
            )
        if not hasattr(self, "bullet_tracker"):
            self.bullet_tracker = BulletTracker()
        if not hasattr(self, "risk_planner"):
            self.risk_planner = RiskPlanner(
                safe_margin=safe_margin,
                allow_vertical=allow_vertical,
            )
        if not hasattr(self, "action_stabilizer"):
            self.action_stabilizer = ActionStabilizer(
                safe_margin=safe_margin,
            )
        if not hasattr(self, "scene_machine"):
            self.scene_machine = GameSceneStateMachine()

    def analyze_game_state(self, detections, timestamp=None):
        """分类检测、跟踪自机/敌弹并计算未来碰撞风险。"""
        self._ensure_control_components()
        now = time.monotonic() if timestamp is None else timestamp
        bullets = []
        player_bullets = []
        enemies = []
        players = []
        fallback_players = []
        powerups = []

        width = self.screen_region["width"] if self.screen_region else 640
        height = self.screen_region["height"] if self.screen_region else 480
        screen_center_x = width // 2

        for detection in detections:
            class_name = str(detection.get("class_name", "")).lower()
            if detection.get("player_fallback_candidate") or (
                class_name == "enemy_small_red"
                and float(detection.get("center_y", 0))
                >= height * PLAYER_FALLBACK_MIN_Y_RATIO
                and float(detection.get("center_x", width))
                <= width * PLAYER_FALLBACK_MAX_X_RATIO
                and 6 <= float(detection.get("width", 0)) <= 45
                and 6 <= float(detection.get("height", 0)) <= 45
            ):
                fallback_players.append(detection)
            elif class_name == "character" or (
                "player" in class_name and "bullet" not in class_name
            ):
                players.append(detection)
            elif "bullet_player" in class_name:
                player_bullets.append(detection)
            elif "bullet_enemy" in class_name or class_name == "bullet":
                bullets.append(detection)
            elif "enemy" in class_name or "boss" in class_name:
                enemies.append(detection)
            elif "power" in class_name or "item" in class_name:
                powerups.append(detection)

        plausible_players = [
            candidate
            for candidate in players
            if float(candidate.get("center_y", 0)) >= height * 0.62
        ]
        player = max(
            plausible_players,
            key=lambda item: float(item.get("confidence", 0.0)),
            default=None,
        )
        fallback_player = max(
            fallback_players,
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                float(item.get("height", 0.0)),
            ),
            default=None,
        )
        if fallback_player is not None and (
            player is None
            or float(fallback_player["center_y"])
            > float(player["center_y"]) + height * 0.08
        ):
            player = dict(fallback_player)
            player.setdefault("model_class_name", player.get("class_name"))
            player["class_name"] = "character_fallback"
            player["player_detection_source"] = "red_sprite_fallback"
        elif player is not None:
            player = dict(player)
            player["player_detection_source"] = "model_character"
        player_track = self.player_tracker.update(
            player,
            width,
            height,
            now=now,
        )
        tracked_player = player_track.as_dict()
        tracked_bullets = self.bullet_tracker.update(bullets, now=now)

        danger_level = 0
        immediate_threats = []
        predicted_threats = []
        max_collision_risk = 0.0
        for bullet in tracked_bullets:
            distance = (
                (float(bullet["center_x"]) - player_track.x) ** 2
                + (float(bullet["center_y"]) - player_track.y) ** 2
            ) ** 0.5
            metrics = collision_metrics(
                bullet,
                player_track.x,
                player_track.y,
            )
            bullet.update(metrics)
            risk = float(metrics["collision_risk"])
            max_collision_risk = max(max_collision_risk, risk)
            if distance < 100:
                danger_level += 3
                immediate_threats.append(bullet)
            elif distance < 200:
                danger_level += 1
            danger_level += int(round(risk * 4))
            if risk >= 0.65:
                predicted_threats.append(bullet)

        scene_state = self.scene_machine.update(
            player_detected=player is not None,
            player_safe=player_track.safe_to_control,
            bullet_count=len(tracked_bullets),
            enemy_count=len(enemies),
            player_bullet_count=len(player_bullets),
        )
        return {
            "bullets": tracked_bullets,
            "player_bullets": player_bullets,
            "enemies": enemies,
            "player": player,
            "tracked_player": tracked_player,
            "player_source": player_track.source,
            "player_valid": player_track.valid,
            "safe_to_control": player_track.safe_to_control,
            "powerups": powerups,
            "danger_level": danger_level,
            "immediate_threats": immediate_threats,
            "predicted_threats": predicted_threats,
            "max_collision_risk": round(max_collision_risk, 4),
            "scene_state": scene_state,
            "action_allowed": (
                self.scene_machine.action_allowed
                and player_track.safe_to_control
            ),
            "screen_center_x": screen_center_x,
            "player_x": player_track.x,
            "player_y": player_track.y,
            "screen_bottom": height,
            "screen_width": width,
            "screen_height": height,
        }

    def make_decision(self, game_state):
        """根据候选方向未来风险、边界和场景状态生成稳定动作。"""
        self._ensure_control_components()
        now = time.monotonic()
        player_x = float(game_state["player_x"])
        player_y = float(game_state["player_y"])
        width = int(game_state.get("screen_width", 640))
        height = int(game_state.get("screen_height", 480))
        safe_to_control = bool(game_state.get("action_allowed", False))

        if not safe_to_control:
            movement, stabilization_reason = self.action_stabilizer.stabilize(
                "stay",
                player_x,
                player_y,
                width,
                height,
                False,
                now,
            )
            game_state["candidate_costs"] = {}
            game_state["decision_reason"] = (
                f"safe_stop:{game_state.get('scene_state', 'unknown')}:"
                f"{game_state.get('player_source', 'missing')}"
            )
            game_state["stabilization_reason"] = stabilization_reason
            if hasattr(self, "stats"):
                self.stats["safe_stop_frames"] += 1
            return movement, False, False

        enemies = game_state["enemies"]
        preferred_x = (
            min(
                enemies,
                key=lambda item: abs(float(item["center_x"]) - player_x),
            )["center_x"]
            if enemies
            else width / 2
        )
        proposed, costs = self.risk_planner.choose(
            game_state["bullets"],
            player_x,
            player_y,
            width,
            height,
            preferred_x=float(preferred_x),
        )

        max_risk = float(game_state.get("max_collision_risk", 0.0))
        mode_adjustment = {
            "defensive": -0.08,
            "balanced": 0.0,
            "aggressive": 0.08,
        }.get(self.ai_mode, 0.0)
        risk_trigger = max(
            0.15,
            min(
                0.65,
                0.55
                - float(getattr(self, "bullet_threshold", 0.05)) * 4.0
                + mode_adjustment,
            ),
        )
        game_state["risk_trigger"] = round(risk_trigger, 4)
        if max_risk < risk_trigger and self.ai_mode == "aggressive":
            proposed = self.calculate_aim_movement(enemies, player_x)
        elif max_risk < 0.15 and not enemies:
            proposed = "stay"

        movement, stabilization_reason = self.action_stabilizer.stabilize(
            proposed,
            player_x,
            player_y,
            width,
            height,
            True,
            now,
        )
        game_state["candidate_costs"] = costs
        game_state["decision_reason"] = (
            "trajectory_avoidance"
            if max_risk >= risk_trigger
            else "positioning"
        )
        game_state["stabilization_reason"] = stabilization_reason
        if hasattr(self, "stats"):
            self.stats["direction_switches"] = self.action_stabilizer.switch_count
            self.stats["blocked_switches"] = self.action_stabilizer.blocked_count
            if game_state.get("player_source") == "predicted":
                self.stats["player_predicted_frames"] += 1
            if max_risk >= risk_trigger:
                self.stats["collision_warning_frames"] += 1
                self.stats["dodge_count"] += 1

        bomb_ready = now - getattr(self, "last_bomb_time", float("-inf")) >= getattr(
            self,
            "bomb_cooldown",
            4.0,
        )
        if (
            getattr(self, "auto_bomb", True)
            and bomb_ready
            and max_risk >= 0.92
            and len(game_state.get("predicted_threats", [])) >= 2
        ):
            self.last_bomb_time = now
            game_state["decision_reason"] = "imminent_collision_bomb"
            return "bomb", False, False

        return movement, True, max_risk >= risk_trigger

    def calculate_aim_movement(self, enemies, screen_center_x):
        """计算瞄准移动"""
        if not enemies:
            return 'stay'

        nearest_enemy = min(enemies, key=lambda e: abs(e['center_x'] - screen_center_x))

        if nearest_enemy['center_x'] < screen_center_x - 50:
            return 'left'
        elif nearest_enemy['center_x'] > screen_center_x + 50:
            return 'right'
        else:
            return 'stay'

    def execute_action(self, movement, shooting, focused):
        """执行动作"""
        with self.action_lock:
            try:
                return self.execute_simple_action(
                    movement,
                    shooting,
                    focused,
                )
            except Exception as e:
                print(f"❌ 动作执行失败: {e}")
                return False

    def execute_simple_action(self, movement, shooting, focused):
        """只发送与上一动作相比发生变化的按键事件。"""
        try:
            if not hasattr(self, "input_state"):
                self.input_state = InputStateMachine(pyautogui)
            transition = self.input_state.apply(
                movement,
                bool(shooting),
                bool(focused),
            )
            self.last_input_transition = transition
            if transition["pressed"] or transition["released"]:
                if movement != "stay":
                    self.stats["movements"] += 1
                if "z" in transition["pressed"]:
                    self.stats["shots_fired"] += 1
            self.stats["key_down_events"] = self.input_state.key_down_count
            self.stats["key_up_events"] = self.input_state.key_up_count
            return True
        except Exception as e:
            print(f"❌ 简单动作执行失败: {e}")
            return False

    def stop_action(self):
        """停止当前动作"""
        with self.action_lock:
            try:
                if hasattr(self, "input_state"):
                    released = self.input_state.release_all()
                    self.last_input_transition = {
                        "pressed": [],
                        "released": released,
                        "held": [],
                    }
                    if hasattr(self, "stats"):
                        self.stats["key_up_events"] = (
                            self.input_state.key_up_count
                        )
                else:
                    for key in ("left", "right", "up", "down", "z", "shift"):
                        pyautogui.keyUp(key)
            except Exception as e:
                print(f"❌ 停止动作失败: {e}")

    def ai_loop(self):
        """AI主循环"""
        print("🤖 AI主循环启动")

        consecutive_errors = 0
        max_errors = 10

        while self.running:
            try:
                if self.paused:
                    time.sleep(0.1)
                    continue

                current_time = time.time()

                if current_time - self.last_detection_time < self.detection_interval:
                    time.sleep(0.001)
                    continue

                self.last_detection_time = current_time

                # 截图检测
                image = self.take_screenshot()
                if image is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        print("❌ 连续截图失败，停止AI")
                        break
                    time.sleep(0.1)
                    continue

                consecutive_errors = 0

                # 目标检测
                detections = self.detect_objects(image)

                # 游戏状态分析
                game_state = self.analyze_game_state(detections)

                # 决策
                movement, shooting, focused = self.make_decision(game_state)

                # PyAutoGUI向活动窗口发送全局按键；焦点不正确时绝不执行动作。
                focus_ok = self.ensure_game_focus()
                executed = False
                execution_reason = None
                if not focus_ok:
                    execution_reason = "focus_unavailable"
                    self.stats["focus_loss_frames"] += 1
                    # 焦点丢失时不能保留上一帧已经按下的方向。
                    self.stop_action()
                elif movement == 'bomb':
                    try:
                        self.input_state.apply("stay", False, False)
                        self.input_state.press_once("x")
                        self.stats['bombs_used'] += 1
                        print("💣 使用炸弹!")
                        executed = True
                    except Exception as e:
                        execution_reason = f"bomb_failed: {e}"
                        print(f"❌ 炸弹动作失败: {e}")
                else:
                    executed = bool(
                        self.execute_action(movement, shooting, focused)
                    )
                    if not executed:
                        execution_reason = "input_backend_failed"

                if self.session_recorder:
                    self.session_recorder.record(
                        frame=image,
                        detections=detections,
                        game_state=game_state,
                        action={
                            "movement": movement,
                            "shooting": bool(shooting),
                            "focused": bool(focused),
                            "executed": executed,
                            "focus_ok": focus_ok,
                            "reason": execution_reason,
                            "decision_reason": game_state.get(
                                "decision_reason"
                            ),
                            "stabilization_reason": game_state.get(
                                "stabilization_reason"
                            ),
                            "input_transition": self.last_input_transition,
                        },
                        screen_region=self.screen_region,
                    )

                if not focus_ok:
                    time.sleep(0.1)
                    continue

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ AI循环错误: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    print("❌ 连续错误过多，停止AI")
                    break
                time.sleep(0.1)

        print("🛑 AI主循环结束")
        if self.running:
            self.stop(reason="ai_loop_ended")

    def print_stats(self):
        """打印统计信息"""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            print(f"\n📊 运行统计:")
            print(f"⏱️  运行时间: {runtime:.1f}秒")
            print(f"👁️  总检测次数: {self.stats['detections']}")
            print(f"🕹️  移动次数: {self.stats['movements']}")
            print(f"🔫 射击次数: {self.stats['shots_fired']}")
            print(f"💣 炸弹使用: {self.stats['bombs_used']}")
            print(f"🏃 闪避次数: {self.stats['dodge_count']}")
            print(f"🔁 方向切换: {self.stats['direction_switches']}")
            print(f"🧊 防抖拦截: {self.stats['blocked_switches']}")
            print(f"📍 自机预测帧: {self.stats['player_predicted_frames']}")
            print(f"🛑 安全停控帧: {self.stats['safe_stop_frames']}")
            print(f"⚠️ 碰撞预警帧: {self.stats['collision_warning_frames']}")
            print(
                f"⌨️ 按键事件: ↓{self.stats['key_down_events']} "
                f"↑{self.stats['key_up_events']}"
            )
            print(f"🔥 推理引擎: Ultralytics/PyTorch {torch.__version__}")
            if runtime > 0:
                dps = self.stats['shots_fired'] / runtime
                print(f"📈 射击频率: {dps:.1f}/秒")

    def start(self):
        """启动AI系统"""
        print("🚀 启动Touhou AI自动控制...")

        if not self.check_dependencies():
            return False

        if not self.load_model():
            return False

        self.setup_signal_handlers()

        # 检测游戏窗口
        self.detect_game()

        if not self.game_found and not self.allow_no_game:
            print("❌ 找到了游戏进程，但没有找到可聚焦的游戏窗口")
            print("💡 请保持游戏为窗口模式，并先运行 python3 touhou_ai.py locate")
            return False

        # 默认不在未检测到游戏时发送全局按键
        game_running, process_name = self.is_game_process_running()
        if not game_running and not self.allow_no_game:
            print("❌ 未检测到游戏进程，拒绝启动全局键盘控制")
            print("💡 请先启动游戏；仅调试时可显式使用 --allow-no-game")
            return False

        if self.game_found and not self.ensure_game_focus(force=True):
            print("❌ 游戏窗口存在，但无法获得输入焦点；拒绝发送全局按键")
            return False

        if self.record_session:
            try:
                self.session_recorder = SessionRecorder(
                    base_dir=self.record_dir or (project_dir / "sessions"),
                    frame_sample_fps=self.record_fps,
                    source="ai",
                    config={
                        "mode": self.ai_mode,
                        "confidence": self.confidence_threshold,
                        "sensitivity": self.bullet_threshold,
                        "model_path": self.model_path,
                        "screen_region": self.screen_region,
                        "safe_margin": self.safe_margin,
                        "player_lost_timeout": self.player_lost_timeout,
                        "allow_vertical": self.allow_vertical,
                    },
                )
                print(f"📝 会话记录: {self.session_recorder.session_dir}")
            except Exception as e:
                print(f"❌ 无法创建会话记录: {e}")
                return False

        self.stats['start_time'] = time.time()
        self.running = True
        self.paused = False

        print(f"\n🎯 截图区域: {self.screen_region}")
        print(f"🧠 AI模式: {self.ai_mode}")
        print(f"📊 检测置信度: {self.confidence_threshold}")
        print(f"⚡ 检测周期: {self.detection_interval*1000:.0f}ms")
        print(f"🔥 推理引擎: Ultralytics/PyTorch {torch.__version__}")

        print("\n✅ AI系统启动成功!")
        print("🎮 已启用游戏窗口焦点守护")
        print("按 Ctrl+C 安全退出\n")

        if game_running:
            print(f"✅ 检测到游戏进程: {process_name}")
            self.start_game_process_monitor()
        else:
            print("⚠️ 调试模式：未检测到游戏进程，仍将发送全局按键")

        ai_thread = threading.Thread(target=self.ai_loop, daemon=True)
        ai_thread.start()

        try:
            while self.running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop(reason="main_loop_ended")

    def stop(self, reason="stopped"):
        """停止AI系统 - 增强版确保键盘释放"""
        if self._cleanup_complete:
            return
        self._cleanup_complete = True

        print("\n🛑 正在停止AI系统...")
        self.running = False
        self.stop_game_process_monitor()

        time.sleep(0.2)

        # 多次确保键盘释放
        print("🔓 释放键盘控制...")
        for i in range(3):
            try:
                self.stop_action()
                # 额外的pyautogui释放
                pyautogui.keyUp('left')
                pyautogui.keyUp('right')
                pyautogui.keyUp('up')
                pyautogui.keyUp('down')
                pyautogui.keyUp('z')
                pyautogui.keyUp('shift')
                pyautogui.keyUp('x')
                pyautogui.keyUp('c')
                time.sleep(0.1)
                print(f"🔓 键盘释放完成 {i+1}/3")
            except Exception as e:
                print(f"⚠️ 键盘释放错误 {i+1}: {e}")

        self.print_stats()

        if self.session_recorder:
            try:
                self.session_recorder.close(
                    end_reason=reason,
                    final_stats=self.stats,
                )
                print(
                    f"📝 会话记录已完成: "
                    f"{self.session_recorder.session_dir}"
                )
            except Exception as e:
                print(f"⚠️ 关闭会话记录失败: {e}")

        print("✅ AI系统已安全停止，键盘控制已释放")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Touhou AI视觉检测与规则控制')
    parser.add_argument('--mode', choices=['balanced', 'aggressive', 'defensive'],
                       default='balanced', help='AI模式')
    parser.add_argument('--confidence', type=float, default=0.15,
                       help='检测置信度阈值')
    parser.add_argument('--sensitivity', type=float, default=0.05,
                       help='弹幕敏感度')
    parser.add_argument('--ai-only', action='store_true',
                       help='仅启动AI（游戏已运行）')
    parser.add_argument('--allow-no-game', action='store_true',
                       help='未检测到游戏进程时仍启动（有误操作风险）')
    parser.add_argument('--record', action='store_true',
                       help='记录检测、决策、动作和抽样画面')
    parser.add_argument('--record-dir',
                       help='会话记录根目录')
    parser.add_argument('--record-fps', type=float, default=2.0,
                       help='每秒保存的画面数；事件仍逐帧记录')
    parser.add_argument('--safe-margin', type=int, default=36,
                       help='自机距离截图边缘的安全像素')
    parser.add_argument('--player-lost-timeout', type=float, default=0.70,
                       help='自机连续漏检超过此秒数后强制停控')
    parser.add_argument('--no-vertical', action='store_true',
                       help='只允许左右移动，不使用上下方向')

    args = parser.parse_args()

    # 创建AI实例
    ai = TouhouAIController(
        allow_no_game=args.allow_no_game,
        record_session=args.record,
        record_dir=args.record_dir,
        record_fps=args.record_fps,
        safe_margin=args.safe_margin,
        player_lost_timeout=args.player_lost_timeout,
        allow_vertical=not args.no_vertical,
    )

    # 设置参数
    ai.ai_mode = args.mode
    ai.confidence_threshold = args.confidence
    ai.bullet_threshold = args.sensitivity

    # 启动AI
    try:
        ai.start()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        ai.stop(reason="exception")

if __name__ == "__main__":
    main()
