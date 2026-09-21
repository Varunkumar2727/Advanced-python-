"""
Webcam Stopwatch
================
A Tkinter + OpenCV app that shows a live webcam feed with a
stopwatch/session timer overlaid on top of the video (burned into the
displayed frame), plus Start / Pause / Reset / Lap controls and the
ability to save a timestamped snapshot.

Dependencies:
    pip install opencv-python pillow

Run:
    python webcam_stopwatch.py
"""

import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwatch_snapshots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DISPLAY_MAX_W = 760
DISPLAY_MAX_H = 560
FRAME_INTERVAL_MS = 30


def format_elapsed(seconds):
    """seconds (float) -> 'HH:MM:SS.mmm'"""
    total_ms = int(seconds * 1000)
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


class WebcamStopwatch:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Stopwatch")
        self.root.geometry("980x720")
        self.root.minsize(760, 600)

        self.cap = None
        self.running_feed = False
        self.last_frame = None

        # Stopwatch state
        self.timer_running = False
        self.start_time = None
        self.accumulated = 0.0   # seconds accumulated across pauses
        self.laps = []

        self._build_ui()

    # ----------------------------------------------------------------- #
    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Stop Webcam", command=self.stop_webcam).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Save Snapshot", command=self.save_snapshot).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Click 'Start Webcam' to begin.")
        ttk.Label(toolbar, textvariable=self.status_var, foreground="gray").pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(self.root, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Stopwatch panel
        sw_frame = ttk.Frame(self.root, padding=10)
        sw_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.time_var = tk.StringVar(value="00:00:00.000")
        tk.Label(sw_frame, textvariable=self.time_var, font=("Consolas", 28, "bold")).pack(side=tk.LEFT, padx=(0, 20))

        btn_frame = ttk.Frame(sw_frame)
        btn_frame.pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Start", command=self.start_timer).grid(row=0, column=0, padx=3)
        ttk.Button(btn_frame, text="Pause", command=self.pause_timer).grid(row=0, column=1, padx=3)
        ttk.Button(btn_frame, text="Reset", command=self.reset_timer).grid(row=0, column=2, padx=3)
        ttk.Button(btn_frame, text="Lap", command=self.record_lap).grid(row=0, column=3, padx=3)

        self.laps_list = tk.Listbox(sw_frame, height=4, width=30)
        self.laps_list.pack(side=tk.RIGHT, padx=(20, 0))
        ttk.Label(sw_frame, text="Laps:").pack(side=tk.RIGHT)

    # ----------------------------------------------------------------- #
    # Webcam
    # ----------------------------------------------------------------- #
    def start_webcam(self):
        self.stop_webcam()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access the webcam.")
            return
        self.cap = cap
        self.running_feed = True
        self.status_var.set("Webcam live.")
        self._loop()

    def stop_webcam(self):
        self.running_feed = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.canvas.delete("all")
        self.status_var.set("Webcam stopped.")

    def _loop(self):
        if not self.running_feed or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self._draw_overlay(frame)
            self.last_frame = frame
            self._render_to_canvas(frame)
        self._update_time_label()
        self.root.after(FRAME_INTERVAL_MS, self._loop)

    # ----------------------------------------------------------------- #
    # Stopwatch logic
    # ----------------------------------------------------------------- #
    def _current_elapsed(self):
        if self.timer_running and self.start_time is not None:
            return self.accumulated + (time.time() - self.start_time)
        return self.accumulated

    def start_timer(self):
        if not self.timer_running:
            self.timer_running = True
            self.start_time = time.time()

    def pause_timer(self):
        if self.timer_running:
            self.accumulated = self._current_elapsed()
            self.timer_running = False
            self.start_time = None

    def reset_timer(self):
        self.timer_running = False
        self.start_time = None
        self.accumulated = 0.0
        self.laps = []
        self.laps_list.delete(0, tk.END)
        self.time_var.set("00:00:00.000")

    def record_lap(self):
        elapsed = self._current_elapsed()
        self.laps.append(elapsed)
        self.laps_list.insert(tk.END, f"Lap {len(self.laps)}: {format_elapsed(elapsed)}")
        self.laps_list.see(tk.END)

    def _update_time_label(self):
        self.time_var.set(format_elapsed(self._current_elapsed()))

    # ----------------------------------------------------------------- #
    # Overlay + rendering
    # ----------------------------------------------------------------- #
    def _draw_overlay(self, frame):
        text = format_elapsed(self._current_elapsed())
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        x, y = 16, th + 24
        cv2.rectangle(frame, (x - 10, 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
        color = (60, 220, 60) if self.timer_running else (60, 160, 255)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

        status_text = "RUNNING" if self.timer_running else "PAUSED"
        cv2.putText(frame, status_text, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _render_to_canvas(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        w, h = pil_img.size
        scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self.display_photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width() or DISPLAY_MAX_W
        canvas_h = self.canvas.winfo_height() or DISPLAY_MAX_H
        x = max(canvas_w // 2, new_w // 2)
        y = max(canvas_h // 2, new_h // 2)
        self.canvas.create_image(x, y, image=self.display_photo, anchor="center")

    # ----------------------------------------------------------------- #
    def save_snapshot(self):
        if self.last_frame is None:
            messagebox.showwarning("No Frame", "Start the webcam first.")
            return
        filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(path, self.last_frame)
        self.status_var.set(f"Saved: {filename}")

    def on_close(self):
        self.stop_webcam()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = WebcamStopwatch(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
