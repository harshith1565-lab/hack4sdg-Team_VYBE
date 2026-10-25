"""GajaGuard desktop prototype UI - YOLO elephant detection + Arduino + ntfy.

Windows-oriented single-file Tkinter application. Requires:
    py -m pip install ultralytics opencv-python pyserial

The app uses the pretrained YOLOv8n COCO model for elephant detection.
The Arduino output should be a low-current indicator/driver input only.
"""

import threading
import queue
import time
import os
from datetime import datetime
from urllib.parse import quote
from urllib.request import Request, urlopen

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import serial
from ultralytics import YOLO
import winsound


# --------------------------- Default configuration ---------------------------
DEFAULT_COM_PORT = "COM10"  # Change in the UI if Device Manager shows another port.
DEFAULT_BAUD_RATE = 9600
DEFAULT_CONFIDENCE = 0.40
DEFAULT_REQUIRED_HITS = 3
DEFAULT_CLEAR_FRAMES = 30
DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_NTFY_TOPIC = "GajaGuard-sightings-9fK2vR7pQ4mX8dT6"

# Theme
BG = "#0b1220"
PANEL = "#121c2e"
PANEL_2 = "#17243a"
BORDER = "#283750"
TEXT = "#edf4ff"
MUTED = "#96a8c2"
TEAL = "#36d6b0"
RED = "#ff5d6c"
AMBER = "#f5b94c"


class GajaGuardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GajaGuard | Wildlife Early Warning")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 680)
        self.root.configure(bg=BG)

        self.events = queue.Queue()
        self.worker = None
        self.stop_event = threading.Event()
        self.alarm_stop = threading.Event()
        self.alarm_thread = None
        self.last_frame_lock = threading.Lock()
        self.last_frame = None
        self.frame_version = 0
        self.displayed_frame_version = -1
        self.preview_error_reported = False
        self.preview_photo = None
        self.preview_size = None

        self.running = False
        self.serial_device = None
        self.siren_on = False
        self.elephant_count = 0
        self.detection_count = 0
        self.alert_count = 0
        self.last_confidence = 0.0
        self.last_alert_time = "—"
        self.started_at = None

        self.com_var = tk.StringVar(value=DEFAULT_COM_PORT)
        self.topic_var = tk.StringVar(value=DEFAULT_NTFY_TOPIC)
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        self.status_vars = {
            "ai": tk.StringVar(value="STANDBY"),
            "camera": tk.StringVar(value="OFFLINE"),
            "arduino": tk.StringVar(value="NOT CONNECTED"),
            "notify": tk.StringVar(value="READY"),
            "alarm": tk.StringVar(value="ALARM OFF"),
            "confidence": tk.StringVar(value="—"),
            "sightings": tk.StringVar(value="0"),
            "alerts": tk.StringVar(value="0"),
            "uptime": tk.StringVar(value="00:00"),
        }

        self._setup_styles()
        self._build_ui()
        self._poll_events()
        self._refresh_preview()
        self._update_uptime()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------------- UI ------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Panel2.TFrame", background=PANEL_2)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 15, "bold"))
        style.configure("Header.TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 23, "bold"))
        style.configure("SubHeader.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=(12, 9))
        style.configure("Primary.TButton", background=TEAL, foreground="#06221d", borderwidth=0)
        style.map("Primary.TButton", background=[("active", "#63e7c8"), ("disabled", "#52756e")])
        style.configure("Stop.TButton", background="#3a2030", foreground="#ffdfe3", borderwidth=0)
        style.map("Stop.TButton", background=[("active", "#613040"), ("disabled", "#372b37")])
        style.configure("Secondary.TButton", background=PANEL_2, foreground=TEXT, borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#263853")])
        style.configure("TEntry", fieldbackground="#0e1727", foreground=TEXT, insertcolor=TEXT, padding=7)
        style.configure("Horizontal.TScale", background=PANEL, troughcolor="#26364d")

    def _panel(self, parent, **kwargs):
        return tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1, **kwargs)

    def _label(self, parent, text, **kwargs):
        return tk.Label(parent, text=text, bg=kwargs.pop("bg", PANEL), fg=kwargs.pop("fg", TEXT),
                        font=kwargs.pop("font", ("Segoe UI", 10)), **kwargs)

    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True, padx=22, pady=18)

        # Header
        header = tk.Frame(outer, bg=BG)
        header.pack(fill="x", pady=(0, 16))
        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left", fill="x", expand=True)
        self._label(brand, "GAJAGUARD", bg=BG, fg=TEAL, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self._label(brand, "Wildlife Early Warning", bg=BG, font=("Segoe UI", 24, "bold")).pack(anchor="w", pady=(2, 0))
        self._label(brand, "AI-powered sighting detection • Local alarm • Team notification", bg=BG, fg=MUTED).pack(anchor="w", pady=(3, 0))
        header_right = tk.Frame(header, bg=BG)
        header_right.pack(side="right", anchor="e")
        self.live_badge = self._label(header_right, "●  SYSTEM STANDBY", bg="#172b38", fg=TEAL, font=("Segoe UI", 10, "bold"), padx=14, pady=9)
        self.live_badge.pack(anchor="e")
        self._label(header_right, "DEMO PROTOTYPE  /  NOT FOR FIELD SAFETY USE", bg=BG, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="e", pady=(6, 0))

        # Summary cards
        cards = tk.Frame(outer, bg=BG)
        cards.pack(fill="x", pady=(0, 14))
        self._metric_card(cards, "AI DETECTOR", "ai", "●")
        self._metric_card(cards, "CAMERA", "camera", "◉")
        self._metric_card(cards, "ARDUINO", "arduino", "⌁")
        self._metric_card(cards, "TEAM ALERTS", "notify", "↗")

        # Main layout
        content = tk.Frame(outer, bg=BG)
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=7, uniform="content")
        content.grid_columnconfigure(1, weight=4, uniform="content")
        content.grid_rowconfigure(0, weight=1)

        # Left: camera
        left = tk.Frame(content, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        camera_panel = self._panel(left)
        camera_panel.pack(fill="both", expand=True)
        topbar = tk.Frame(camera_panel, bg=PANEL)
        topbar.pack(fill="x", padx=16, pady=(13, 9))
        self._label(topbar, "LIVE DETECTION FEED", font=("Segoe UI", 11, "bold")).pack(side="left")
        self._label(topbar, "COCO • ELEPHANT CLASS", fg=MUTED, font=("Segoe UI", 8, "bold")).pack(side="right")

        self.video_label = tk.Label(camera_panel, bg="#050911", fg=MUTED,
                                    text="CAMERA OFFLINE\n\nStart monitoring to view the live feed",
                                    font=("Segoe UI", 13), justify="center")
        self.video_label.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        video_footer = tk.Frame(camera_panel, bg=PANEL)
        video_footer.pack(fill="x", padx=15, pady=(0, 13))
        self._label(video_footer, "DETECTION CONFIDENCE", font=("Segoe UI", 8, "bold"), fg=MUTED).pack(side="left")
        self.confidence_value = self._label(video_footer, "—", font=("Segoe UI", 10, "bold"), fg=TEAL)
        self.confidence_value.pack(side="right")

        # Right controls + event feed
        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(2, weight=1)

        control_panel = self._panel(right)
        control_panel.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self._label(control_panel, "MONITORING CONTROLS", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=15, pady=(13, 10))

        btn_row = tk.Frame(control_panel, bg=PANEL)
        btn_row.pack(fill="x", padx=14)
        self.start_btn = ttk.Button(btn_row, text="▶  Start monitoring", style="Primary.TButton", command=self.start_monitoring)
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.stop_btn = ttk.Button(btn_row, text="■  Stop", style="Stop.TButton", command=self.stop_monitoring, state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))

        config = tk.Frame(control_panel, bg=PANEL)
        config.pack(fill="x", padx=15, pady=(15, 7))
        self._label(config, "Arduino COM port", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.port_entry = ttk.Entry(config, textvariable=self.com_var, width=12)
        self.port_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self._label(config, "Confidence threshold", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", pady=(0, 5))
        self.conf_text = self._label(config, "0.40", fg=TEAL, font=("Segoe UI", 10, "bold"))
        self.conf_text.grid(row=1, column=1, sticky="e")
        self.conf_scale = ttk.Scale(config, from_=0.20, to=0.80, variable=self.conf_var, command=self._on_conf_change)
        self.conf_scale.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        config.grid_columnconfigure(0, weight=1)
        config.grid_columnconfigure(1, weight=1)

        config2 = tk.Frame(control_panel, bg=PANEL)
        config2.pack(fill="x", padx=15, pady=(7, 3))
        self._label(config2, "ntfy topic", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 5))
        self.topic_entry = ttk.Entry(config2, textvariable=self.topic_var)
        self.topic_entry.pack(fill="x")
        actions = tk.Frame(control_panel, bg=PANEL)
        actions.pack(fill="x", padx=14, pady=(10, 14))
        ttk.Button(actions, text="Send test alert", style="Secondary.TButton", command=self.send_test_alert).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(actions, text="Save snapshot", style="Secondary.TButton", command=self.save_snapshot).pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Alarm overview
        alarm_panel = self._panel(right)
        alarm_panel.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        alarm_line = tk.Frame(alarm_panel, bg=PANEL)
        alarm_line.pack(fill="x", padx=15, pady=12)
        self.alarm_dot = self._label(alarm_line, "●", fg=MUTED, font=("Segoe UI", 20, "bold"))
        self.alarm_dot.pack(side="left", padx=(0, 10))
        alarm_text = tk.Frame(alarm_line, bg=PANEL)
        alarm_text.pack(side="left", fill="x", expand=True)
        self._label(alarm_text, "ALARM STATUS", font=("Segoe UI", 8, "bold"), fg=MUTED).pack(anchor="w")
        self.alarm_status_label = self._label(alarm_text, "STANDBY", font=("Segoe UI", 13, "bold"))
        self.alarm_status_label.pack(anchor="w", pady=(2, 0))
        self._label(alarm_line, "3 consecutive\nframes to trigger", font=("Segoe UI", 8), fg=MUTED, justify="right").pack(side="right")

        # Event feed
        feed_panel = self._panel(right)
        feed_panel.grid(row=2, column=0, sticky="nsew")
        self._label(feed_panel, "ACTIVITY LOG", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=15, pady=(13, 8))
        feed_wrap = tk.Frame(feed_panel, bg=PANEL)
        feed_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log_list = tk.Listbox(feed_wrap, bg="#0c1524", fg="#d5e2f5", selectbackground="#244b59",
                                   relief="flat", borderwidth=0, font=("Consolas", 9), height=8,
                                   activestyle="none", highlightthickness=0)
        scroll = ttk.Scrollbar(feed_wrap, orient="vertical", command=self.log_list.yview)
        self.log_list.configure(yscrollcommand=scroll.set)
        self.log_list.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        footer = tk.Frame(outer, bg=BG)
        footer.pack(fill="x", pady=(12, 0))
        self._label(footer, "GAJAGUARD  •  LOCAL AI + ARDUINO + TEAM ALERT", bg=BG, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(side="left")
        self._label(footer, "AI results are unverified sightings — confirm before taking action.", bg=BG, fg=AMBER, font=("Segoe UI", 8)).pack(side="right")
        self.log("System ready. Configure COM port and start monitoring.")

    def _metric_card(self, parent, title, key, icon):
        card = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        card.pack(side="left", fill="x", expand=True, padx=(0, 9))
        body = tk.Frame(card, bg=PANEL)
        body.pack(fill="x", padx=13, pady=12)
        self._label(body, icon, fg=TEAL, font=("Segoe UI", 17, "bold")).pack(side="left", padx=(0, 10))
        textcol = tk.Frame(body, bg=PANEL)
        textcol.pack(side="left", fill="x", expand=True)
        self._label(textcol, title, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w")
        value = self._label(textcol, self.status_vars[key].get(), font=("Segoe UI", 11, "bold"))
        value.pack(anchor="w", pady=(4, 0))
        setattr(self, f"metric_{key}", value)

    # ---------------------------- utilities --------------------------------
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events.put(("log", f"[{timestamp}] {message}"))

    def set_status(self, key, value):
        self.events.put(("status", key, value))

    def _on_conf_change(self, _value=None):
        self.conf_text.configure(text=f"{self.conf_var.get():.2f}")

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "log":
                    self.log_list.insert("end", event[1])
                    self.log_list.yview_moveto(1.0)
                elif event[0] == "status":
                    _, key, value = event
                    if key in self.status_vars:
                        self.status_vars[key].set(value)
                        metric = getattr(self, f"metric_{key}", None)
                        if metric is not None:
                            metric.configure(text=value)
                    if key == "alarm":
                        self.alarm_status_label.configure(text=value)
                        active = value == "ALARM ACTIVE"
                        self.alarm_dot.configure(fg=RED if active else MUTED)
                        self.live_badge.configure(
                            text="●  MONITORING LIVE" if self.running else "●  SYSTEM STANDBY",
                            bg="#38212b" if active else ("#17392f" if self.running else "#172b38"),
                            fg=RED if active else TEAL
                        )
                    elif key in ("ai", "camera", "arduino", "notify"):
                        pass
                elif event[0] == "running":
                    running = event[1]
                    self.running = running
                    self.start_btn.configure(state="disabled" if running else "normal")
                    self.stop_btn.configure(state="normal" if running else "disabled")
                    self.port_entry.configure(state="disabled" if running else "normal")
                    self.topic_entry.configure(state="disabled" if running else "normal")
                    self.conf_scale.configure(state="disabled" if running else "normal")
                    self.live_badge.configure(
                        text="●  MONITORING LIVE" if running else "●  SYSTEM STANDBY",
                        bg="#17392f" if running else "#172b38",
                        fg=TEAL
                    )
                elif event[0] == "confidence":
                    value = event[1]
                    self.confidence_value.configure(text=f"{value:.0%}")
                elif event[0] == "increment":
                    counter = event[1]
                    if counter == "sightings":
                        self.detection_count += 1
                        value = str(self.detection_count)
                        self.status_vars["sightings"].set(value)
                    elif counter == "alerts":
                        self.alert_count += 1
                        value = str(self.alert_count)
                        self.status_vars["alerts"].set(value)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_events)

    def _refresh_preview(self):
        # Render pixels directly through Tk PhotoImage.put(). This bypasses
        # Pillow/ImageTk and Tk's PNG/PPM file decoders, which may be blocked
        # or unavailable on some Windows-managed installations.
        with self.last_frame_lock:
            frame = self.last_frame.copy() if self.last_frame is not None else None
            version = self.frame_version

        if frame is not None and version != self.displayed_frame_version:
            try:
                label_width = max(240, self.video_label.winfo_width())
                label_height = max(135, self.video_label.winfo_height())
                h, w = frame.shape[:2]

                # Keep pixel conversion small enough for responsive UI updates.
                scale = min(label_width / w, label_height / h)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                image_height, image_width = rgb.shape[:2]

                desired_size = (image_width, image_height)
                if self.preview_photo is None or self.preview_size != desired_size:
                    self.preview_photo = tk.PhotoImage(
                        master=self.root,
                        width=image_width,
                        height=image_height
                    )
                    self.preview_size = desired_size
                    self.video_label.configure(
                        image=self.preview_photo,
                        text="",
                        compound="center"
                    )
                    self.video_label.image = self.preview_photo

                # Tk accepts nested lists of color values through PhotoImage.put.
                # ASCII hex pixels avoid all image-file decoding paths.
                rows = []
                for row in rgb:
                    rows.append("{" + " ".join(
                        f"#{int(pixel[0]):02x}{int(pixel[1]):02x}{int(pixel[2]):02x}"
                        for pixel in row
                    ) + "}")


                self.preview_photo.put(rows, to=(0, 0))
                self.video_label.configure(image=self.preview_photo, text="", compound="center")
                self.video_label.image = self.preview_photo
                self.displayed_frame_version = version
                self.preview_error_reported = False

            except Exception as exc:
                if not self.preview_error_reported:
                    self.preview_error_reported = True
                    self.log(f"Camera preview display error: {type(exc).__name__}: {exc}")

        # Direct pixel painting is heavier than native image decoding; update at
        # about 7-10 FPS, enough for a hackathon demo while keeping the UI responsive.
        self.root.after(120, self._refresh_preview)

    def _update_uptime(self):
        if self.started_at and self.running:
            elapsed = int(time.time() - self.started_at)
            self.status_vars["uptime"].set(f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        self.root.after(1000, self._update_uptime)

    # --------------------------- notifications ------------------------------
    def _post_ntfy(self, title, message, topic=None, timeout=5):
        topic = (topic if topic is not None else self.topic_var.get()).strip()
        if not topic:
            raise ValueError("ntfy topic is blank")
        encoded_topic = quote(topic, safe="-_")
        request = Request(
            f"https://ntfy.sh/{encoded_topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": "high", "Tags": "warning,elephant"},
            method="POST"
        )
        with urlopen(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"ntfy returned HTTP {response.status}")

    def send_test_alert(self):
        topic = self.topic_var.get().strip()
        if not topic:
            messagebox.showwarning("Topic required", "Enter the ntfy topic you and your teammate subscribed to.")
            return
        self.status_vars["notify"].set("SENDING TEST…")
        self.metric_notify.configure(text="SENDING TEST…")
        self.log("Sending test notification…")

        def task():
            try:
                self._post_ntfy("GajaGuard Test Alert",
                                "GajaGuard UI test notification. This is not a real wildlife sighting.",
                                topic=topic)
                self.set_status("notify", "TEST SENT")
                self.log("Test alert sent to ntfy topic.")
            except Exception as exc:
                self.set_status("notify", "SEND FAILED")
                self.log(f"Test notification failed: {type(exc).__name__}")
        threading.Thread(target=task, daemon=True).start()

    def send_sighting_alert(self, confidence):
        timestamp = datetime.now().astimezone().strftime("%d %b %Y, %I:%M:%S %p %Z")
        message = (
            "GAJAGUARD — POSSIBLE ELEPHANT SIGHTING\n\n"
            f"Detection confidence: {confidence:.1%}\n"
            f"Time: {timestamp}\n"
            "Status: AI-detected, not yet verified.\n"
            "Location: Not configured.\n\n"
            "Please verify the sighting before taking action."
        )
        try:
            self._post_ntfy("GajaGuard Wildlife Alert", message, timeout=5)
            self.set_status("notify", "ALERT SENT")
            self.log("Wildlife alert sent to teammate.")
            self.events.put(("increment", "alerts"))
        except Exception as exc:
            self.set_status("notify", "SEND FAILED")
            self.log(f"Wildlife alert failed: {type(exc).__name__}; monitoring continues.")

    # ---------------------------- controls ---------------------------------
    def start_monitoring(self):
        if self.worker and self.worker.is_alive():
            return
        port = self.com_var.get().strip()
        topic = self.topic_var.get().strip()
        if not port:
            messagebox.showwarning("COM port required", "Enter the Arduino COM port, such as COM10.")
            return
        if not topic:
            messagebox.showwarning("ntfy topic required", "Enter the ntfy topic name.")
            return
        self.stop_event.clear()
        self.alarm_stop.clear()
        self.started_at = time.time()
        self.running = True
        self.events.put(("running", True))
        self.set_status("ai", "LOADING…")
        self.set_status("camera", "STARTING…")
        self.set_status("arduino", "CONNECTING…")
        self.set_status("notify", "READY")
        self.set_status("alarm", "ALARM OFF")
        self.log("Starting GajaGuard monitoring…")
        self.worker = threading.Thread(target=self._monitor_loop, args=(port, topic, float(self.conf_var.get())), daemon=True)
        self.worker.start()

    def stop_monitoring(self):
        if self.running:
            self.log("Stop requested. Shutting down devices…")
            self.stop_event.set()
            self.alarm_stop.set()
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="disabled")

    def _start_alarm_sound(self):
        self.alarm_stop.clear()
        self.alarm_thread = threading.Thread(target=self._alarm_sound_loop, name="GajaGuardAlarm", daemon=True)
        self.alarm_thread.start()

    def _alarm_sound_loop(self):
        tones = [(1000, 500), (1400, 500), (800, 500)]
        try:
            while not self.alarm_stop.is_set():
                for frequency, duration in tones:
                    if self.alarm_stop.is_set():
                        return
                    winsound.Beep(frequency, duration)
                    if self.alarm_stop.wait(0.15):
                        return
        except Exception as exc:
            self.log(f"Laptop alarm sound unavailable: {type(exc).__name__}")

    def _send_arduino(self, command):
        device = self.serial_device
        if device is not None and device.is_open:
            try:
                device.write((command + "\n").encode("ascii"))
                device.flush()
                return True
            except Exception as exc:
                self.log(f"Arduino command failed: {type(exc).__name__}")
        return False

    def _monitor_loop(self, port, topic, confidence_threshold):
        camera = None
        serial_dev = None
        model = None
        siren_on = False
        hits = 0
        missing = 0
        version = 0
        try:
            self.log("Loading YOLO model (first launch may download weights)…")
            model = YOLO(DEFAULT_MODEL_PATH)
            elephant_ids = [i for i, name in model.names.items() if name.lower() == "elephant"]
            if not elephant_ids:
                raise RuntimeError("This YOLO model has no elephant class.")
            self.set_status("ai", "READY")
            self.log("YOLO loaded. Elephant class identified.")

            try:
                serial_dev = serial.Serial(port, DEFAULT_BAUD_RATE, timeout=1)
                self.serial_device = serial_dev
                time.sleep(2)  # Nano can reset after opening serial.
                self.set_status("arduino", f"CONNECTED · {port}")
                self.log(f"Arduino connected on {port}.")
            except (serial.SerialException, OSError) as exc:
                serial_dev = None
                self.serial_device = None
                self.set_status("arduino", "DISCONNECTED")
                self.log(f"Arduino unavailable ({type(exc).__name__}); AI/notifications will continue.")

            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError("Could not open webcam. Close other apps using the camera.")

            self.set_status("camera", "LIVE")
            self.log("Webcam live. Monitoring started.")

            while not self.stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    self.log("Webcam frame read failed; stopping monitor.")
                    break

                result = model.predict(
                    source=frame,
                    classes=elephant_ids,
                    conf=confidence_threshold,
                    imgsz=640,
                    verbose=False
                )[0]

                detected = result.boxes is not None and len(result.boxes) > 0
                if detected:
                    hits += 1
                    missing = 0
                    confidence = float(result.boxes.conf.max().item())
                    self.last_confidence = confidence
                    if hits % 10 == 0:
                        self.events.put(("confidence", confidence))
                else:
                    hits = 0
                    missing += 1

                if not siren_on and hits >= DEFAULT_REQUIRED_HITS:
                    confidence = float(result.boxes.conf.max().item()) if detected else self.last_confidence
                    siren_on = True
                    self.siren_on = True
                    self._send_arduino("ON")
                    self._start_alarm_sound()
                    self.set_status("alarm", "ALARM ACTIVE")
                    self.events.put(("increment", "sightings"))
                    self.log(f"Possible elephant confirmed across {DEFAULT_REQUIRED_HITS} consecutive frames ({confidence:.0%}).")
                    threading.Thread(target=self.send_sighting_alert, args=(confidence,), daemon=True).start()

                elif siren_on and missing >= DEFAULT_CLEAR_FRAMES:
                    siren_on = False
                    self.siren_on = False
                    self.alarm_stop.set()
                    self._send_arduino("OFF")
                    self.set_status("alarm", "ALARM OFF")
                    self.log("No elephant detected for 30 consecutive frames. Alarm cleared.")

                annotated = result.plot()
                label = "POSSIBLE ELEPHANT · ALERT ACTIVE" if siren_on else "MONITORING · ALARM STANDBY"
                color = (70, 70, 255) if siren_on else (54, 214, 176)
                cv2.rectangle(annotated, (10, 10), (annotated.shape[1] - 10, 54), (12, 18, 32), -1)
                cv2.putText(annotated, label, (22, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
                cv2.putText(annotated, f"CONF {self.last_confidence:.0%}  |  HITS {hits}/{DEFAULT_REQUIRED_HITS}",
                            (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

                version += 1
                # Keep only the newest frame so a fast webcam cannot flood the UI queue.
                with self.last_frame_lock:
                    self.last_frame = annotated
                    self.frame_version = version

        except Exception as exc:
            self.log(f"Monitoring error: {type(exc).__name__}: {exc}")
            self.set_status("ai", "ERROR" if model is None else "STOPPED")
        finally:
            self.stop_event.set()
            self.alarm_stop.set()
            try:
                if serial_dev is not None and serial_dev.is_open:
                    try:
                        serial_dev.write(b"OFF\n")
                        serial_dev.flush()
                        time.sleep(0.05)
                    except Exception:
                        pass
                    serial_dev.close()
            except Exception:
                pass
            self.serial_device = None
            if camera is not None:
                camera.release()
            self.set_status("camera", "OFFLINE")
            self.set_status("alarm", "ALARM OFF")
            if serial_dev is not None:
                self.set_status("arduino", "DISCONNECTED")
            self.events.put(("running", False))
            self.log("Monitoring stopped. Camera released and alarm shutdown requested.")

    def save_snapshot(self):
        with self.last_frame_lock:
            frame = self.last_frame.copy() if self.last_frame is not None else None
        if frame is None:
            messagebox.showinfo("No frame yet", "Start monitoring first to capture a snapshot.")
            return
        default_name = f"GajaGuard_sighting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = filedialog.asksaveasfilename(
            title="Save GajaGuard snapshot", defaultextension=".jpg",
            initialfile=default_name, filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")]
        )
        if not path:
            return
        if cv2.imwrite(path, frame):
            self.log(f"Snapshot saved: {os.path.basename(path)}")
            messagebox.showinfo("Snapshot saved", f"Saved image to:\n{path}")
        else:
            messagebox.showerror("Save failed", "Could not save the snapshot.")

    def on_close(self):
        self.stop_event.set()
        self.alarm_stop.set()
        try:
            self._send_arduino("OFF")
        except Exception:
            pass
        self.root.after(150, self.root.destroy)


if __name__ == "__main__":
    app_root = tk.Tk()
    app = GajaGuardApp(app_root)
    app_root.mainloop()
