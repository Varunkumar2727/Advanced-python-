"""
Vision Studio
=============
A combined Tkinter + OpenCV desktop app that merges:

  1) Live Filter Studio  -- an Instagram/Snapchat-style filter pipeline
     (Grayscale, Sepia, Vintage, Cool, Warm, Invert, Cartoon, Blur,
     Vignette, Motion Mask) with a strength slider, snapshot gallery,
     and video recording.

  2) Real-Time Image Analyzer -- toggleable analysis overlays
     (brightness/contrast stats, RGB histogram, Canny edge detection,
     contour detection & counting, face/eye detection, motion
     detection with bounding boxes).

Pipeline per frame:
    raw frame -> [stats + histogram from raw frame]
              -> [color filter + strength blend]  => "looked" frame
              -> [analysis overlays drawn on top of the looked frame]
              -> displayed / recorded / snapshotted

Sources supported: live webcam, an uploaded image, or an uploaded video
(the same source can be filtered AND analyzed at the same time).

Dependencies:
    pip install opencv-python pillow numpy

Run:
    python vision_studio.py
"""

import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
COLOR_BG         = "#0d0f1e"
COLOR_PANEL      = "#141733"
COLOR_PANEL_ALT  = "#1b1e42"
COLOR_ACCENT     = "#3fa9f5"
COLOR_ACCENT_2   = "#7b2ff7"
COLOR_TEXT       = "#e8eaf6"
COLOR_TEXT_MUTED = "#8b8fb8"
COLOR_CANVAS_BG  = "#0a0c18"
COLOR_DANGER     = "#e63b6b"

GALLERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gallery")
os.makedirs(GALLERY_DIR, exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# --------------------------------------------------------------------------- #
# Filter functions -- each takes a BGR frame and returns a BGR frame
# --------------------------------------------------------------------------- #
def filter_grayscale(frame, bg_subtractor=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filter_sepia(frame, bg_subtractor=None):
    kernel = np.array([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame.astype(np.float32), kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)


def _add_vignette(frame, strength=1.0):
    h, w = frame.shape[:2]
    kx = cv2.getGaussianKernel(w, w * 0.5)
    ky = cv2.getGaussianKernel(h, h * 0.5)
    mask = ky @ kx.T
    mask = mask / mask.max()
    mask = mask ** (1.0 / max(strength, 0.01) * 0.6)
    vignette = frame.astype(np.float32) * mask[:, :, None]
    return np.clip(vignette, 0, 255).astype(np.uint8)


def filter_vintage(frame, bg_subtractor=None):
    sepia = filter_sepia(frame)
    hsv = cv2.cvtColor(sepia, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)
    desat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _add_vignette(desat, strength=1.2)


def filter_cool(frame, bg_subtractor=None):
    b, g, r = cv2.split(frame.astype(np.int32))
    b = np.clip(b + 25, 0, 255)
    r = np.clip(r - 15, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def filter_warm(frame, bg_subtractor=None):
    b, g, r = cv2.split(frame.astype(np.int32))
    r = np.clip(r + 25, 0, 255)
    b = np.clip(b - 15, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def filter_invert(frame, bg_subtractor=None):
    return cv2.bitwise_not(frame)


def filter_cartoon(frame, bg_subtractor=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(frame, d=9, sigmaColor=200, sigmaSpace=200)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_bgr)


def filter_blur(frame, bg_subtractor=None):
    return cv2.GaussianBlur(frame, (15, 15), 0)


def filter_vignette_only(frame, bg_subtractor=None):
    return _add_vignette(frame, strength=1.0)


def filter_motion_mask(frame, bg_subtractor=None):
    """Detect motion; keep moving regions in color, dim the rest to grayscale."""
    if bg_subtractor is None:
        return frame

    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.dilate(fg_mask, np.ones((7, 7), np.uint8), iterations=2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dimmed_gray = cv2.cvtColor((gray * 0.35).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) > 0
    result = np.where(mask_3ch, frame, dimmed_gray)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) >= 800:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 140), 2)

    return result


FILTERS = {
    "Normal": None,
    "Grayscale": filter_grayscale,
    "Sepia": filter_sepia,
    "Vintage": filter_vintage,
    "Cool": filter_cool,
    "Warm": filter_warm,
    "Invert": filter_invert,
    "Cartoon": filter_cartoon,
    "Blur": filter_blur,
    "Vignette": filter_vignette_only,
    "Motion Mask": filter_motion_mask,
}


def blend(original, filtered, strength):
    """strength: 0.0 (all original) .. 1.0 (all filtered)."""
    if filtered is None:
        return original
    strength = max(0.0, min(1.0, strength))
    return cv2.addWeighted(filtered, strength, original, 1 - strength, 0)


# --------------------------------------------------------------------------- #
# Analysis helper functions
# --------------------------------------------------------------------------- #
def compute_basic_stats(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    return brightness, contrast


def draw_edges(frame, t1, t2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    overlay = frame.copy()
    overlay[edges > 0] = (255, 255, 0)
    blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    return blended, edges


def draw_contours(frame, edges, min_area=150):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) >= min_area]
    cv2.drawContours(frame, significant, -1, (0, 255, 0), 2)
    return frame, len(significant)


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
        cv2.putText(frame, "Face", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 255), 1)
    return frame, len(faces)


def detect_motion(frame, bg_subtractor, min_area=800):
    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moving_regions = 0
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            moving_regions += 1
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame, moving_regions


def build_histogram_image(frame, width=300, height=150):
    """Return a small BGR image containing an RGB channel histogram plot."""
    hist_img = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [(255, 80, 80), (80, 255, 80), (80, 80, 255)]  # B, G, R (as drawn)
    bgr_channels = cv2.split(frame)

    for channel, color in zip(bgr_channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX)
        points = []
        for i in range(256):
            x = int(i * (width / 256))
            y = height - int(hist[i][0])
            points.append((x, y))
        for i in range(1, len(points)):
            cv2.line(hist_img, points[i - 1], points[i], color, 1)

    return hist_img


# --------------------------------------------------------------------------- #
# Reusable flat-styled button
# --------------------------------------------------------------------------- #
def make_flat_button(parent, text, command, accent=COLOR_ACCENT, fg="#ffffff", small=False):
    return tk.Button(
        parent, text=text, command=command, bg=accent, fg=fg,
        activebackground=COLOR_ACCENT_2, activeforeground="#ffffff",
        relief="flat", bd=0, padx=(8 if small else 14), pady=(3 if small else 6),
        font=("Segoe UI", 8 if small else 9, "bold"), cursor="hand2",
    )


# --------------------------------------------------------------------------- #
# Main Application
# --------------------------------------------------------------------------- #
class VisionStudio:
    DISPLAY_MAX_W = 680
    DISPLAY_MAX_H = 500
    FRAME_INTERVAL_MS = 30

    def __init__(self, root):
        self.root = root
        self.root.title("Vision Studio -- Filters + Real-Time Analysis")
        self.root.geometry("1300x820")
        self.root.minsize(1050, 660)
        self.root.configure(bg=COLOR_BG)
        self._configure_ttk_style()

        # Source state
        self.cap = None
        self.static_frame = None
        self.mode = None           # 'webcam' | 'video' | 'image'
        self.running = False

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=40, detectShadows=True)

        # Recording state
        self.video_writer = None
        self.recording = False

        self.last_time = time.time()
        self.fps = 0.0
        self.last_frame_output = None   # final rendered/recorded frame
        self.display_photo = None
        self.hist_photo = None
        self.gallery_thumbnails = []

        self._build_toolbar()
        self._build_body()
        self._refresh_gallery()

    # ----------------------------------------------------------------- #
    def _configure_ttk_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=COLOR_PANEL)
        style.configure("TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 9))
        style.configure("TScale", background=COLOR_PANEL, troughcolor=COLOR_PANEL_ALT)
        style.configure("Vertical.TScrollbar", background=COLOR_PANEL_ALT, troughcolor=COLOR_PANEL)
        style.configure("TRadiobutton", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 9))
        style.map("TRadiobutton", foreground=[("selected", COLOR_ACCENT)])
        style.configure("TCheckbutton", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 9))
        style.map("TCheckbutton", foreground=[("selected", COLOR_ACCENT)])

    # ----------------------------------------------------------------- #
    # Layout
    # ----------------------------------------------------------------- #
    def _build_toolbar(self):
        toolbar = tk.Frame(self.root, bg=COLOR_PANEL, height=54)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        inner = tk.Frame(toolbar, bg=COLOR_PANEL)
        inner.pack(side=tk.LEFT, padx=10, pady=8)

        make_flat_button(inner, "Start Webcam", self.start_webcam).pack(side=tk.LEFT, padx=3)
        make_flat_button(inner, "Upload Image", self.upload_image, accent=COLOR_PANEL_ALT).pack(side=tk.LEFT, padx=3)
        make_flat_button(inner, "Upload Video", self.upload_video, accent=COLOR_PANEL_ALT).pack(side=tk.LEFT, padx=3)
        make_flat_button(inner, "Stop", self.stop_source, accent=COLOR_DANGER).pack(side=tk.LEFT, padx=3)
        make_flat_button(inner, "Save Snapshot", self.save_snapshot, accent=COLOR_ACCENT_2).pack(side=tk.LEFT, padx=3)
        self.record_button = make_flat_button(inner, "Start Recording", self.toggle_recording, accent=COLOR_ACCENT_2)
        self.record_button.pack(side=tk.LEFT, padx=3)

        self.status_var = tk.StringVar(value="No source loaded.")
        tk.Label(toolbar, textvariable=self.status_var, bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=14)

    def _build_body(self):
        body = tk.Frame(self.root, bg=COLOR_BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: preview + gallery strip
        left_frame = tk.Frame(body, bg=COLOR_BG)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(left_frame, bg=COLOR_CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        gallery_frame = tk.Frame(left_frame, bg=COLOR_PANEL, height=110)
        gallery_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        gallery_frame.pack_propagate(False)

        tk.Label(gallery_frame, text="Gallery", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(4, 0))

        gallery_scroll = tk.Canvas(gallery_frame, bg=COLOR_PANEL, highlightthickness=0, height=80)
        gallery_hscroll = ttk.Scrollbar(gallery_frame, orient=tk.HORIZONTAL, command=gallery_scroll.xview)
        self.gallery_inner = tk.Frame(gallery_scroll, bg=COLOR_PANEL)

        self.gallery_inner.bind(
            "<Configure>", lambda e: gallery_scroll.configure(scrollregion=gallery_scroll.bbox("all"))
        )
        gallery_scroll.create_window((0, 0), window=self.gallery_inner, anchor="nw")
        gallery_scroll.configure(xscrollcommand=gallery_hscroll.set)
        gallery_scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8)
        gallery_hscroll.pack(side=tk.BOTTOM, fill=tk.X, padx=8)

        # Right: scrollable control panel (Filters + Analysis + Stats + Histogram)
        right_container = tk.Frame(body, width=340, bg=COLOR_PANEL)
        right_container.pack(side=tk.RIGHT, fill=tk.Y)
        right_container.pack_propagate(False)

        control_canvas = tk.Canvas(right_container, bg=COLOR_PANEL, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_container, orient=tk.VERTICAL, command=control_canvas.yview)
        self.controls_frame = tk.Frame(control_canvas, bg=COLOR_PANEL, padx=14, pady=14)

        self.controls_frame.bind(
            "<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        control_canvas.create_window((0, 0), window=self.controls_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_mousewheel(event):
            delta = -1 * (event.delta // 120) if event.delta else 0
            control_canvas.yview_scroll(int(delta), "units")

        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        control_canvas.bind_all("<Button-4>", lambda e: control_canvas.yview_scroll(-1, "units"))
        control_canvas.bind_all("<Button-5>", lambda e: control_canvas.yview_scroll(1, "units"))

        self._build_controls()

    def _build_controls(self):
        cf = self.controls_frame

        # ---- Filters section ----
        tk.Label(cf, text="Look / Filter", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self.filter_var = tk.StringVar(value="Normal")
        for name in FILTERS.keys():
            ttk.Radiobutton(
                cf, text=name, variable=self.filter_var, value=name,
                command=lambda: None
            ).pack(anchor="w", pady=1)

        tk.Label(cf, text="Filter Strength", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
        self.strength_var = tk.DoubleVar(value=1.0)
        ttk.Scale(cf, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.strength_var).pack(fill=tk.X)

        ttk.Separator(cf, orient="horizontal").pack(fill=tk.X, pady=12)

        # ---- Analysis section ----
        tk.Label(cf, text="Analysis Overlays", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self.show_edges_var = tk.BooleanVar(value=False)
        self.show_contours_var = tk.BooleanVar(value=False)
        self.show_faces_var = tk.BooleanVar(value=False)
        self.show_motion_var = tk.BooleanVar(value=False)
        self.show_histogram_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(cf, text="Edge Detection (Canny)", variable=self.show_edges_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(cf, text="Contour Detection", variable=self.show_contours_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(cf, text="Face / Eye Detection", variable=self.show_faces_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(cf, text="Motion Detection", variable=self.show_motion_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(cf, text="Show Histogram", variable=self.show_histogram_var).pack(anchor="w", pady=2)

        tk.Label(cf, text="Canny Threshold 1", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
        self.canny_t1 = tk.IntVar(value=80)
        ttk.Scale(cf, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.canny_t1).pack(fill=tk.X)

        tk.Label(cf, text="Canny Threshold 2", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        self.canny_t2 = tk.IntVar(value=160)
        ttk.Scale(cf, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.canny_t2).pack(fill=tk.X)

        tk.Label(cf, text="Min Contour Area", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        self.contour_min_area = tk.IntVar(value=150)
        ttk.Scale(cf, from_=10, to=2000, orient=tk.HORIZONTAL, variable=self.contour_min_area).pack(fill=tk.X)

        tk.Label(cf, text="Motion Sensitivity (min area)", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        self.motion_min_area = tk.IntVar(value=800)
        ttk.Scale(cf, from_=100, to=5000, orient=tk.HORIZONTAL, variable=self.motion_min_area).pack(fill=tk.X)

        ttk.Separator(cf, orient="horizontal").pack(fill=tk.X, pady=12)

        # ---- Live stats ----
        tk.Label(cf, text="Live Stats", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))

        self.stats_labels = {}
        for key, text in [
            ("resolution", "Resolution:"),
            ("fps", "FPS:"),
            ("brightness", "Brightness:"),
            ("contrast", "Contrast:"),
            ("faces", "Faces:"),
            ("contours", "Contours:"),
            ("motion", "Moving regions:"),
            ("recording", "Recording:"),
        ]:
            row = tk.Frame(cf, bg=COLOR_PANEL)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=text, bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, width=15, anchor="w",
                     font=("Segoe UI", 9)).pack(side=tk.LEFT)
            val = tk.Label(row, text="--", bg=COLOR_PANEL, fg=COLOR_ACCENT, font=("Segoe UI", 9, "bold"))
            val.pack(side=tk.LEFT)
            self.stats_labels[key] = val

        ttk.Separator(cf, orient="horizontal").pack(fill=tk.X, pady=12)

        # ---- Histogram ----
        tk.Label(cf, text="Histogram (raw frame)", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.hist_canvas = tk.Canvas(cf, width=300, height=150, bg=COLOR_CANVAS_BG, highlightthickness=0)
        self.hist_canvas.pack(pady=4)

    # ----------------------------------------------------------------- #
    # Source management
    # ----------------------------------------------------------------- #
    def start_webcam(self):
        self.stop_source()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access the webcam.")
            return
        self.cap = cap
        self.mode = "webcam"
        self.running = True
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=40, detectShadows=True)
        self.status_var.set("Webcam live.")
        self.last_time = time.time()
        self.fps = 0.0
        self._schedule_next_frame()

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the selected image file.")
            return
        self.stop_source()
        self.static_frame = img
        self.mode = "image"
        self.running = True
        self.status_var.set(f"Image loaded: {os.path.basename(path)}")
        self.last_time = time.time()
        self.fps = 0.0
        self._schedule_next_frame()

    def upload_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")],
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open the selected video file.")
            return
        self.stop_source()
        self.cap = cap
        self.mode = "video"
        self.running = True
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=40, detectShadows=True)
        self.status_var.set(f"Video loaded: {os.path.basename(path)}")
        self.last_time = time.time()
        self.fps = 0.0
        self._schedule_next_frame()

    def stop_source(self):
        self.running = False
        if self.recording:
            self._stop_recording()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.static_frame = None
        self.mode = None
        self.canvas.delete("all")
        self.hist_canvas.delete("all")
        self.status_var.set("Stopped.")

    # ----------------------------------------------------------------- #
    # Frame loop
    # ----------------------------------------------------------------- #
    def _schedule_next_frame(self):
        if self.running:
            self.root.after(self.FRAME_INTERVAL_MS, self._process_and_render)

    def _process_and_render(self):
        if not self.running:
            return

        frame = None
        if self.mode in ("webcam", "video") and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if self.mode == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret:
                    self.status_var.set("Stream ended.")
                    self.running = False
                    return
        elif self.mode == "image" and self.static_frame is not None:
            frame = self.static_frame.copy()

        if frame is None:
            self._schedule_next_frame()
            return

        if self.mode == "webcam":
            frame = cv2.flip(frame, 1)

        output = self._run_pipeline(frame)

        self.last_frame_output = output
        self._render_to_canvas(output)
        self._update_fps()
        self.stats_labels["recording"].config(text="YES" if self.recording else "No")

        if self.recording and self.video_writer is not None:
            self.video_writer.write(output)

        self._schedule_next_frame()

    def _run_pipeline(self, raw_frame):
        """raw_frame -> stats/histogram -> filter -> analysis overlays -> output."""
        # 1) Stats + histogram from the untouched raw frame
        brightness, contrast = compute_basic_stats(raw_frame)
        self.stats_labels["resolution"].config(text=f"{raw_frame.shape[1]}x{raw_frame.shape[0]}")
        self.stats_labels["brightness"].config(text=f"{brightness:.1f}")
        self.stats_labels["contrast"].config(text=f"{contrast:.1f}")

        if self.show_histogram_var.get():
            self._render_histogram(build_histogram_image(raw_frame))
        else:
            self.hist_canvas.delete("all")

        # 2) Apply the selected color/style filter, blended by strength
        filter_fn = FILTERS.get(self.filter_var.get())
        filtered = filter_fn(raw_frame, self.bg_subtractor) if filter_fn else None
        display = blend(raw_frame, filtered, self.strength_var.get()) if filtered is not None else raw_frame.copy()

        # 3) Analysis overlays drawn on top of the filtered/blended frame
        edges = None
        if self.show_edges_var.get() or self.show_contours_var.get():
            display, edges = draw_edges(display, self.canny_t1.get(), self.canny_t2.get())
            if not self.show_edges_var.get():
                # edges were only needed for contour extraction; keep the look clean
                filter_fn2 = FILTERS.get(self.filter_var.get())
                filtered2 = filter_fn2(raw_frame, self.bg_subtractor) if filter_fn2 else None
                display = blend(raw_frame, filtered2, self.strength_var.get()) if filtered2 is not None else raw_frame.copy()

        if self.show_contours_var.get() and edges is not None:
            display, contour_count = draw_contours(display, edges, self.contour_min_area.get())
            self.stats_labels["contours"].config(text=str(contour_count))
        else:
            self.stats_labels["contours"].config(text="--")

        if self.show_faces_var.get():
            display, face_count = detect_faces(display)
            self.stats_labels["faces"].config(text=str(face_count))
        else:
            self.stats_labels["faces"].config(text="--")

        if self.show_motion_var.get() and self.mode in ("webcam", "video"):
            display, motion_count = detect_motion(display, self.bg_subtractor, self.motion_min_area.get())
            self.stats_labels["motion"].config(text=str(motion_count))
        else:
            self.stats_labels["motion"].config(text="--")

        return display

    def _update_fps(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = self.fps * 0.8 + instant_fps * 0.2 if self.fps else instant_fps
        self.stats_labels["fps"].config(text=f"{self.fps:.1f}")

    # ----------------------------------------------------------------- #
    # Rendering
    # ----------------------------------------------------------------- #
    def _render_to_canvas(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        w, h = pil_img.size
        scale = min(self.DISPLAY_MAX_W / w, self.DISPLAY_MAX_H / h, 1.0)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self.display_photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width() or self.DISPLAY_MAX_W
        canvas_h = self.canvas.winfo_height() or self.DISPLAY_MAX_H
        x = max(canvas_w // 2, new_w // 2)
        y = max(canvas_h // 2, new_h // 2)
        self.canvas.create_image(x, y, image=self.display_photo, anchor="center")

    def _render_histogram(self, hist_bgr_img):
        rgb = cv2.cvtColor(hist_bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.hist_photo = ImageTk.PhotoImage(pil_img)
        self.hist_canvas.delete("all")
        self.hist_canvas.create_image(0, 0, image=self.hist_photo, anchor="nw")

    # ----------------------------------------------------------------- #
    # Snapshot / Recording / Gallery
    # ----------------------------------------------------------------- #
    def save_snapshot(self):
        if self.last_frame_output is None:
            messagebox.showwarning("No Frame", "There is no frame to save yet.")
            return
        filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(GALLERY_DIR, filename)
        cv2.imwrite(path, self.last_frame_output)
        self.status_var.set(f"Saved: {filename}")
        self._refresh_gallery()

    def toggle_recording(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if self.last_frame_output is None:
            messagebox.showwarning("No Frame", "Start a webcam or video source before recording.")
            return
        filename = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        path = os.path.join(GALLERY_DIR, filename)
        h, w = self.last_frame_output.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
        self.recording = True
        self.record_button.config(text="Stop Recording", bg=COLOR_DANGER)
        self.status_var.set(f"Recording to {filename}...")

    def _stop_recording(self):
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.record_button.config(text="Start Recording", bg=COLOR_ACCENT_2)
        self.status_var.set("Recording stopped.")
        self._refresh_gallery()

    def _refresh_gallery(self):
        for widget in self.gallery_inner.winfo_children():
            widget.destroy()
        self.gallery_thumbnails.clear()

        files = sorted(
            [f for f in os.listdir(GALLERY_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            reverse=True,
        )

        if not files:
            tk.Label(self.gallery_inner, text="No snapshots yet.", bg=COLOR_PANEL,
                     fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8, "italic")).pack(side=tk.LEFT, padx=8, pady=8)
            return

        for fname in files:
            self._add_gallery_item(fname)

    def _add_gallery_item(self, fname):
        full_path = os.path.join(GALLERY_DIR, fname)
        try:
            img = Image.open(full_path)
            img.thumbnail((90, 60))
            photo = ImageTk.PhotoImage(img)
        except Exception:
            return
        self.gallery_thumbnails.append(photo)

        item_frame = tk.Frame(self.gallery_inner, bg=COLOR_PANEL_ALT, padx=2, pady=2)
        item_frame.pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(item_frame, image=photo, bg=COLOR_PANEL_ALT).pack()
        make_flat_button(
            item_frame, "Delete", lambda f=fname: self._delete_gallery_item(f),
            accent=COLOR_DANGER, small=True
        ).pack(fill=tk.X, pady=(2, 0))

    def _delete_gallery_item(self, fname):
        full_path = os.path.join(GALLERY_DIR, fname)
        if messagebox.askyesno("Delete Snapshot", f"Delete '{fname}' permanently?"):
            try:
                os.remove(full_path)
            except OSError:
                pass
            self._refresh_gallery()

    def on_close(self):
        self.stop_source()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = VisionStudio(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()