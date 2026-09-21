"""
Snap Studio -- Photo & Video Editor
====================================
A Tkinter + OpenCV desktop app that behaves like the "Edit" screen in the
Android Photos app, but works on:

  - A live webcam feed (capture a photo from it, or record a video)
  - An uploaded photo
  - An uploaded video

Editing tools (Android-Photos style):
  - Rotate 90 CW / 90 CCW / 180, plus a fine-rotate slider (-180..180)
  - Flip Horizontal / Flip Vertical
  - Crop (drag a rectangle directly on the preview, then Apply Crop)
  - Adjust: Brightness, Contrast, Saturation
  - Filters: Normal, Grayscale, Sepia, Vintage, Cool, Warm, Invert,
    Blur, Vignette -- with an intensity slider
  - Undo (steps back through crop operations) and Reset to original
  - Save the edited photo, or export the edited/trimmed video to .mp4

Dependencies:
    pip install opencv-python pillow numpy

Run:
    python snap_studio.py
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
COLOR_CROP_BOX   = "#ffd23f"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snap_studio_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Filter functions -- each takes a BGR frame and returns a BGR frame
# --------------------------------------------------------------------------- #
def filter_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filter_sepia(frame):
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


def filter_vintage(frame):
    sepia = filter_sepia(frame)
    hsv = cv2.cvtColor(sepia, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)
    desat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _add_vignette(desat, strength=1.2)


def filter_cool(frame):
    b, g, r = cv2.split(frame.astype(np.int32))
    b = np.clip(b + 25, 0, 255)
    r = np.clip(r - 15, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def filter_warm(frame):
    b, g, r = cv2.split(frame.astype(np.int32))
    r = np.clip(r + 25, 0, 255)
    b = np.clip(b - 15, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def filter_invert(frame):
    return cv2.bitwise_not(frame)


def filter_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)


def filter_vignette_only(frame):
    return _add_vignette(frame, strength=1.0)


FILTERS = {
    "Normal": None,
    "Grayscale": filter_grayscale,
    "Sepia": filter_sepia,
    "Vintage": filter_vintage,
    "Cool": filter_cool,
    "Warm": filter_warm,
    "Invert": filter_invert,
    "Blur": filter_blur,
    "Vignette": filter_vignette_only,
}


def blend(original, filtered, strength):
    if filtered is None:
        return original
    strength = max(0.0, min(1.0, strength))
    return cv2.addWeighted(filtered, strength, original, 1 - strength, 0)


# --------------------------------------------------------------------------- #
# Transform helpers
# --------------------------------------------------------------------------- #
def rotate_bound(image, angle):
    """Rotate by an arbitrary angle, expanding the canvas so nothing is cut off."""
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = max(int((h * sin) + (w * cos)), 1)
    nH = max(int((h * cos) + (w * sin)), 1)
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))


def adjust_frame(frame, brightness, contrast, saturation):
    out = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    if abs(saturation - 1.0) > 0.001:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


class EditParams:
    """Live (non-destructive) edit parameters applied on top of a base image."""

    def __init__(self):
        self.rot90_count = 0     # multiples of 90 deg clockwise
        self.fine_angle = 0.0    # additional free-angle rotation
        self.flip_h = False
        self.flip_v = False
        self.brightness = 0      # -100..100
        self.contrast = 1.0      # 0.2..3.0
        self.saturation = 1.0    # 0.0..3.0
        self.filter_name = "Normal"
        self.filter_strength = 1.0

    def reset(self):
        self.__init__()


def apply_pipeline(base_img, p: EditParams):
    """Apply rotate -> flip -> adjust -> filter, in that order, non-destructively."""
    out = base_img
    r = p.rot90_count % 4
    if r == 1:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif r == 2:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif r == 3:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if abs(p.fine_angle) > 0.01:
        out = rotate_bound(out, p.fine_angle)

    if p.flip_h:
        out = cv2.flip(out, 1)
    if p.flip_v:
        out = cv2.flip(out, 0)

    out = adjust_frame(out, p.brightness, p.contrast, p.saturation)

    filt = FILTERS.get(p.filter_name)
    if filt is not None:
        out = blend(out, filt(out), p.filter_strength)

    return out


# --------------------------------------------------------------------------- #
# Reusable flat-styled button
# --------------------------------------------------------------------------- #
def make_flat_button(parent, text, command, accent=COLOR_ACCENT, fg="#ffffff", small=False):
    return tk.Button(
        parent, text=text, command=command, bg=accent, fg=fg,
        activebackground=COLOR_ACCENT_2, activeforeground="#ffffff",
        relief="flat", bd=0, padx=(8 if small else 12), pady=(3 if small else 6),
        font=("Segoe UI", 8 if small else 9, "bold"), cursor="hand2",
    )


# --------------------------------------------------------------------------- #
# Main Application
# --------------------------------------------------------------------------- #
class SnapStudio:
    DISPLAY_MAX_W = 700
    DISPLAY_MAX_H = 520
    FRAME_INTERVAL_MS = 30

    def __init__(self, root):
        self.root = root
        self.root.title("Snap Studio -- Photo & Video Editor")
        self.root.geometry("1340x840")
        self.root.minsize(1080, 680)
        self.root.configure(bg=COLOR_BG)
        self._configure_ttk_style()

        # Source / mode state
        self.mode = None            # 'webcam' | 'photo' | 'video'
        self.cap = None             # webcam VideoCapture
        self.video_path = None
        self.video_total_frames = 0
        self.video_fps = 25.0
        self.running_live = False   # webcam loop active

        # Photo edit state
        self.initial_image = None   # never mutated -- true original for Reset
        self.base_image = None      # current baked-in base (post-crop) for photo edits
        self.undo_stack = []

        # Live edit parameters (shared by photo/video/webcam preview)
        self.params = EditParams()

        # Recording state (webcam -> video)
        self.video_writer = None
        self.recording = False

        # Crop state
        self.crop_mode_active = False
        self.crop_rect_id = None
        self.crop_start_xy = None
        self.crop_end_xy = None
        self.render_scale = 1.0
        self.render_offset = (0, 0)
        self.render_img_size = (0, 0)  # size of the (post-pipeline) image that was rendered

        self.last_time = time.time()
        self.fps = 0.0
        self.display_photo = None

        self._build_toolbar()
        self._build_body()
        self._set_mode_ui_state()

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

    # ----------------------------------------------------------------- #
    # Layout
    # ----------------------------------------------------------------- #
    def _build_toolbar(self):
        toolbar = tk.Frame(self.root, bg=COLOR_PANEL, height=54)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        inner = tk.Frame(toolbar, bg=COLOR_PANEL)
        inner.pack(side=tk.LEFT, padx=10, pady=8)

        make_flat_button(inner, "Start Webcam", self.start_webcam).pack(side=tk.LEFT, padx=3)
        self.capture_btn = make_flat_button(inner, "Capture Photo", self.capture_photo, accent=COLOR_ACCENT_2)
        self.capture_btn.pack(side=tk.LEFT, padx=3)
        self.record_btn = make_flat_button(inner, "Start Recording", self.toggle_recording, accent=COLOR_ACCENT_2)
        self.record_btn.pack(side=tk.LEFT, padx=3)
        make_flat_button(inner, "Upload Photo", self.upload_photo, accent=COLOR_PANEL_ALT).pack(side=tk.LEFT, padx=3)
        make_flat_button(inner, "Upload Video", self.upload_video, accent=COLOR_PANEL_ALT).pack(side=tk.LEFT, padx=3)
        make_flat_button(inner, "Stop Source", self.stop_source, accent=COLOR_DANGER).pack(side=tk.LEFT, padx=3)
        self.save_btn = make_flat_button(inner, "Save Photo", self.save_or_export, accent=COLOR_ACCENT)
        self.save_btn.pack(side=tk.LEFT, padx=3)

        self.status_var = tk.StringVar(value="No source loaded. Start the webcam or upload a photo/video.")
        tk.Label(toolbar, textvariable=self.status_var, bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=14)

    def _build_body(self):
        body = tk.Frame(self.root, bg=COLOR_BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: preview canvas + below-canvas contextual bar (crop hint / video trim)
        left_frame = tk.Frame(body, bg=COLOR_BG)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(left_frame, bg=COLOR_CANVAS_BG, highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # Video trim / scrub bar (shown only in video mode)
        self.video_bar = tk.Frame(left_frame, bg=COLOR_PANEL, height=110)
        tk.Label(self.video_bar, text="Preview Position", bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w", padx=8, pady=(6, 0))
        self.position_var = tk.IntVar(value=0)
        self.position_scale = ttk.Scale(self.video_bar, from_=0, to=1, orient=tk.HORIZONTAL,
                                         variable=self.position_var, command=self._on_position_change)
        self.position_scale.pack(fill=tk.X, padx=8)

        trim_row = tk.Frame(self.video_bar, bg=COLOR_PANEL)
        trim_row.pack(fill=tk.X, padx=8, pady=(6, 4))
        tk.Label(trim_row, text="Trim Start", bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        make_flat_button(trim_row, "Set = Position", self._set_trim_start, accent=COLOR_PANEL_ALT, small=True).pack(side=tk.LEFT, padx=6)
        tk.Label(trim_row, text="Trim End", bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(18, 0))
        make_flat_button(trim_row, "Set = Position", self._set_trim_end, accent=COLOR_PANEL_ALT, small=True).pack(side=tk.LEFT, padx=6)
        self.trim_label_var = tk.StringVar(value="Trim: full clip")
        tk.Label(self.video_bar, textvariable=self.trim_label_var, bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=8, pady=(0, 6))

        # Crop hint bar (shown only in photo mode)
        self.crop_bar = tk.Frame(left_frame, bg=COLOR_PANEL, height=44)
        tk.Label(self.crop_bar, text="Crop tools are in the right panel -> drag on the preview to draw a box, then Apply Crop.",
                 bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 8, "italic")).pack(padx=8, pady=10, anchor="w")

        # Right: scrollable control panel
        right_container = tk.Frame(body, width=360, bg=COLOR_PANEL)
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

        # ---- Rotate & Flip ----
        tk.Label(cf, text="Rotate & Flip", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        row1 = tk.Frame(cf, bg=COLOR_PANEL)
        row1.pack(fill=tk.X, pady=2)
        make_flat_button(row1, "Rotate Left 90°", lambda: self._rotate90(-1), accent=COLOR_PANEL_ALT, small=True).pack(side=tk.LEFT, padx=(0, 4))
        make_flat_button(row1, "Rotate Right 90°", lambda: self._rotate90(1), accent=COLOR_PANEL_ALT, small=True).pack(side=tk.LEFT, padx=4)
        make_flat_button(row1, "180°", lambda: self._rotate90(2), accent=COLOR_PANEL_ALT, small=True).pack(side=tk.LEFT, padx=4)

        row2 = tk.Frame(cf, bg=COLOR_PANEL)
        row2.pack(fill=tk.X, pady=4)
        make_flat_button(row2, "Flip Horizontal", self._flip_h, accent=COLOR_PANEL_ALT, small=True).pack(side=tk.LEFT, padx=(0, 4))
        make_flat_button(row2, "Flip Vertical", self._flip_v, accent=COLOR_PANEL_ALT, small=True).pack(side=tk.LEFT, padx=4)

        tk.Label(cf, text="Fine Rotate (°)", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
        self.angle_var = tk.DoubleVar(value=0.0)
        ttk.Scale(cf, from_=-180, to=180, orient=tk.HORIZONTAL, variable=self.angle_var,
                  command=self._on_angle_change).pack(fill=tk.X)

        ttk.Separator(cf, orient="horizontal").pack(fill=tk.X, pady=12)

        # ---- Adjust ----
        tk.Label(cf, text="Adjust", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        tk.Label(cf, text="Brightness", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 9)).pack(anchor="w")
        self.brightness_var = tk.IntVar(value=0)
        ttk.Scale(cf, from_=-100, to=100, orient=tk.HORIZONTAL, variable=self.brightness_var,
                  command=self._on_adjust_change).pack(fill=tk.X)

        tk.Label(cf, text="Contrast", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        self.contrast_var = tk.DoubleVar(value=1.0)
        ttk.Scale(cf, from_=0.2, to=3.0, orient=tk.HORIZONTAL, variable=self.contrast_var,
                  command=self._on_adjust_change).pack(fill=tk.X)

        tk.Label(cf, text="Saturation", bg=COLOR_PANEL, fg=COLOR_TEXT, font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        self.saturation_var = tk.DoubleVar(value=1.0)
        ttk.Scale(cf, from_=0.0, to=3.0, orient=tk.HORIZONTAL, variable=self.saturation_var,
                  command=self._on_adjust_change).pack(fill=tk.X)

        ttk.Separator(cf, orient="horizontal").pack(fill=tk.X, pady=12)

        # ---- Filters ----
        tk.Label(cf, text="Filters", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        self.filter_var = tk.StringVar(value="Normal")
        for name in FILTERS.keys():
            ttk.Radiobutton(cf, text=name, variable=self.filter_var, value=name,
                             command=self._on_filter_change).pack(anchor="w", pady=1)

        tk.Label(cf, text="Filter Strength", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
        self.strength_var = tk.DoubleVar(value=1.0)
        ttk.Scale(cf, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.strength_var,
                  command=self._on_adjust_change).pack(fill=tk.X)

        ttk.Separator(cf, orient="horizontal").pack(fill=tk.X, pady=12)

        # ---- Crop (photo only) ----
        tk.Label(cf, text="Crop (photo only)", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))
        row3 = tk.Frame(cf, bg=COLOR_PANEL)
        row3.pack(fill=tk.X, pady=2)
        self.crop_toggle_btn = make_flat_button(row3, "Start Crop", self._toggle_crop_mode, accent=COLOR_PANEL_ALT, small=True)
        self.crop_toggle_btn.pack(side=tk.LEFT, padx=(0, 4))
        make_flat_button(row3, "Apply Crop", self._apply_crop, accent=COLOR_ACCENT_2, small=True).pack(side=tk.LEFT, padx=4)
        make_flat_button(row3, "Clear Box", self._clear_crop_box, accent=COLOR_DANGER, small=True).pack(side=tk.LEFT, padx=4)

        ttk.Separator(cf, orient="horizontal").pack(fill=tk.X, pady=12)

        # ---- Actions ----
        tk.Label(cf, text="Actions", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))
        row4 = tk.Frame(cf, bg=COLOR_PANEL)
        row4.pack(fill=tk.X, pady=2)
        make_flat_button(row4, "Undo", self._undo, accent=COLOR_PANEL_ALT, small=True).pack(side=tk.LEFT, padx=(0, 4))
        make_flat_button(row4, "Reset to Original", self._reset_all, accent=COLOR_DANGER, small=True).pack(side=tk.LEFT, padx=4)

        ttk.Separator(cf, orient="horizontal").pack(fill=tk.X, pady=12)

        # ---- Stats ----
        tk.Label(cf, text="Info", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.stats_labels = {}
        for key, text in [("resolution", "Resolution:"), ("fps", "FPS:"), ("recording", "Recording:")]:
            row = tk.Frame(cf, bg=COLOR_PANEL)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=text, bg=COLOR_PANEL, fg=COLOR_TEXT_MUTED, width=13, anchor="w",
                     font=("Segoe UI", 9)).pack(side=tk.LEFT)
            val = tk.Label(row, text="--", bg=COLOR_PANEL, fg=COLOR_ACCENT, font=("Segoe UI", 9, "bold"))
            val.pack(side=tk.LEFT)
            self.stats_labels[key] = val

    # ----------------------------------------------------------------- #
    # Mode-dependent UI visibility
    # ----------------------------------------------------------------- #
    def _set_mode_ui_state(self):
        self.video_bar.pack_forget()
        self.crop_bar.pack_forget()

        if self.mode == "video":
            self.video_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
            self.save_btn.config(text="Export Video")
        elif self.mode == "photo":
            self.crop_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
            self.save_btn.config(text="Save Photo")
        else:
            self.save_btn.config(text="Save Photo")

        self.capture_btn.config(state=(tk.NORMAL if self.mode == "webcam" else tk.DISABLED))
        self.record_btn.config(state=(tk.NORMAL if self.mode == "webcam" else tk.DISABLED))

    # ----------------------------------------------------------------- #
    # Parameter change callbacks (all just trigger a redraw)
    # ----------------------------------------------------------------- #
    def _sync_params_from_widgets(self):
        self.params.brightness = self.brightness_var.get()
        self.params.contrast = self.contrast_var.get()
        self.params.saturation = self.saturation_var.get()
        self.params.filter_name = self.filter_var.get()
        self.params.filter_strength = self.strength_var.get()
        self.params.fine_angle = self.angle_var.get()

    def _on_adjust_change(self, *_):
        self._sync_params_from_widgets()
        self._redraw_static_if_needed()

    def _on_filter_change(self):
        self._sync_params_from_widgets()
        self._redraw_static_if_needed()

    def _on_angle_change(self, *_):
        self._sync_params_from_widgets()
        self._redraw_static_if_needed()

    def _rotate90(self, steps):
        self.params.rot90_count = (self.params.rot90_count + steps) % 4
        self._redraw_static_if_needed()

    def _flip_h(self):
        self.params.flip_h = not self.params.flip_h
        self._redraw_static_if_needed()

    def _flip_v(self):
        self.params.flip_v = not self.params.flip_v
        self._redraw_static_if_needed()

    def _redraw_static_if_needed(self):
        """For photo/video (paused) modes, re-render the current frame immediately."""
        if self.mode == "photo" and self.base_image is not None:
            out = apply_pipeline(self.base_image, self.params)
            self._render_to_canvas(out)
        elif self.mode == "video" and self.cap is not None:
            self._render_video_position(self.position_var.get())

    # ----------------------------------------------------------------- #
    # Source management: Webcam
    # ----------------------------------------------------------------- #
    def start_webcam(self):
        self.stop_source()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access the webcam.")
            return
        self.cap = cap
        self.mode = "webcam"
        self.running_live = True
        self.params.reset()
        self._sync_widgets_from_params()
        self.status_var.set("Webcam live. Capture a photo or start recording.")
        self.last_time = time.time()
        self.fps = 0.0
        self._set_mode_ui_state()
        self._webcam_loop()

    def _webcam_loop(self):
        if not self.running_live or self.mode != "webcam":
            return
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self._sync_params_from_widgets()
            out = apply_pipeline(frame, self.params)
            self._last_webcam_frame = out
            self._render_to_canvas(out)
            self._update_fps()
            self.stats_labels["resolution"].config(text=f"{out.shape[1]}x{out.shape[0]}")
            self.stats_labels["recording"].config(text="YES" if self.recording else "No")
            if self.recording and self.video_writer is not None:
                self.video_writer.write(out)
        self.root.after(self.FRAME_INTERVAL_MS, self._webcam_loop)

    def capture_photo(self):
        if self.mode != "webcam" or not hasattr(self, "_last_webcam_frame"):
            messagebox.showwarning("No Frame", "Start the webcam first.")
            return
        frozen = self._last_webcam_frame.copy()
        self.running_live = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._load_photo_into_editor(frozen)
        self.status_var.set("Photo captured from webcam. Edit and Save when ready.")

    def toggle_recording(self):
        if self.mode != "webcam":
            return
        if not self.recording:
            filename = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            path = os.path.join(OUTPUT_DIR, filename)
            self._sync_params_from_widgets()
            sample = apply_pipeline(self._last_webcam_frame if hasattr(self, "_last_webcam_frame")
                                     else np.zeros((480, 640, 3), np.uint8), self.params)
            h, w = sample.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
            self.recording = True
            self.record_btn.config(text="Stop Recording", bg=COLOR_DANGER)
            self.status_var.set(f"Recording to {filename} ...")
        else:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.config(text="Start Recording", bg=COLOR_ACCENT_2)
            self.status_var.set(f"Recording saved to {OUTPUT_DIR}")

    # ----------------------------------------------------------------- #
    # Source management: Photo
    # ----------------------------------------------------------------- #
    def upload_photo(self):
        path = filedialog.askopenfilename(
            title="Select a photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the selected image file.")
            return
        self.stop_source()
        self._load_photo_into_editor(img)
        self.status_var.set(f"Photo loaded: {os.path.basename(path)}")

    def _load_photo_into_editor(self, img):
        self.mode = "photo"
        self.initial_image = img.copy()
        self.base_image = img.copy()
        self.undo_stack = []
        self.params.reset()
        self._sync_widgets_from_params()
        self._set_mode_ui_state()
        out = apply_pipeline(self.base_image, self.params)
        self.stats_labels["resolution"].config(text=f"{out.shape[1]}x{out.shape[0]}")
        self._render_to_canvas(out)

    # ----------------------------------------------------------------- #
    # Source management: Video
    # ----------------------------------------------------------------- #
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
        self.video_path = path
        self.cap = cap
        self.mode = "video"
        self.video_total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 1)
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.params.reset()
        self._sync_widgets_from_params()

        self.position_scale.config(to=self.video_total_frames)
        self.position_var.set(0)
        self.trim_start_frame = 0
        self.trim_end_frame = self.video_total_frames
        self.trim_label_var.set(f"Trim: 0 -> {self.trim_end_frame} (of {self.video_total_frames})")

        self._set_mode_ui_state()
        self._render_video_position(0)
        self.status_var.set(f"Video loaded: {os.path.basename(path)} ({self.video_total_frames + 1} frames)")

    def _on_position_change(self, *_):
        if self.mode == "video":
            self._render_video_position(int(self.position_var.get()))

    def _render_video_position(self, frame_idx):
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self._sync_params_from_widgets()
        out = apply_pipeline(frame, self.params)
        self._render_to_canvas(out)
        self.stats_labels["resolution"].config(text=f"{out.shape[1]}x{out.shape[0]}")
        self.stats_labels["fps"].config(text=f"{self.video_fps:.1f}")

    def _set_trim_start(self):
        if self.mode != "video":
            return
        self.trim_start_frame = min(int(self.position_var.get()), getattr(self, "trim_end_frame", self.video_total_frames))
        self.trim_label_var.set(f"Trim: {self.trim_start_frame} -> {self.trim_end_frame} (of {self.video_total_frames})")

    def _set_trim_end(self):
        if self.mode != "video":
            return
        self.trim_end_frame = max(int(self.position_var.get()), getattr(self, "trim_start_frame", 0))
        self.trim_label_var.set(f"Trim: {self.trim_start_frame} -> {self.trim_end_frame} (of {self.video_total_frames})")

    # ----------------------------------------------------------------- #
    # Stop
    # ----------------------------------------------------------------- #
    def stop_source(self):
        self.running_live = False
        if self.recording:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_btn.config(text="Start Recording", bg=COLOR_ACCENT_2)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mode = None
        self.video_path = None
        self.canvas.delete("all")
        self.crop_mode_active = False
        self.crop_rect_id = None
        self.status_var.set("Stopped.")
        self._set_mode_ui_state()

    # ----------------------------------------------------------------- #
    # Crop (photo mode -- drag on the canvas)
    # ----------------------------------------------------------------- #
    def _toggle_crop_mode(self):
        if self.mode != "photo":
            messagebox.showinfo("Crop", "Crop is only available in photo mode. Capture or upload a photo first.")
            return
        self.crop_mode_active = not self.crop_mode_active
        self.canvas.config(cursor="crosshair" if self.crop_mode_active else "arrow")
        self.crop_toggle_btn.config(
            text="Stop Crop" if self.crop_mode_active else "Start Crop",
            bg=COLOR_DANGER if self.crop_mode_active else COLOR_PANEL_ALT,
        )

    def _on_canvas_press(self, event):
        if not (self.crop_mode_active and self.mode == "photo"):
            return
        self.crop_start_xy = (event.x, event.y)
        self.crop_end_xy = (event.x, event.y)

    def _on_canvas_drag(self, event):
        if not (self.crop_mode_active and self.mode == "photo" and self.crop_start_xy):
            return
        self.crop_end_xy = (event.x, event.y)
        self.canvas.delete("crop_box")
        x0, y0 = self.crop_start_xy
        x1, y1 = self.crop_end_xy
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=COLOR_CROP_BOX, width=2, tags="crop_box")

    def _on_canvas_release(self, event):
        if not (self.crop_mode_active and self.mode == "photo"):
            return
        self.crop_end_xy = (event.x, event.y)

    def _clear_crop_box(self):
        self.crop_start_xy = None
        self.crop_end_xy = None
        self.canvas.delete("crop_box")

    def _apply_crop(self):
        if self.mode != "photo" or self.base_image is None:
            messagebox.showinfo("Crop", "Crop is only available in photo mode.")
            return
        if not (self.crop_start_xy and self.crop_end_xy):
            messagebox.showinfo("Crop", "Drag a box on the preview first (click 'Start Crop', then drag).")
            return

        # Map canvas coords -> coords on the rendered (post-pipeline) image
        ox, oy = self.render_offset
        img_w, img_h = self.render_img_size
        x0 = (min(self.crop_start_xy[0], self.crop_end_xy[0]) - ox) / self.render_scale
        y0 = (min(self.crop_start_xy[1], self.crop_end_xy[1]) - oy) / self.render_scale
        x1 = (max(self.crop_start_xy[0], self.crop_end_xy[0]) - ox) / self.render_scale
        y1 = (max(self.crop_start_xy[1], self.crop_end_xy[1]) - oy) / self.render_scale

        x0 = int(max(0, min(x0, img_w)))
        x1 = int(max(0, min(x1, img_w)))
        y0 = int(max(0, min(y0, img_h)))
        y1 = int(max(0, min(y1, img_h)))

        if x1 - x0 < 5 or y1 - y0 < 5:
            messagebox.showwarning("Crop", "Crop box is too small.")
            return

        self._sync_params_from_widgets()
        full = apply_pipeline(self.base_image, self.params)
        cropped = full[y0:y1, x0:x1].copy()

        # Bake in: cropped result becomes the new base; all live params reset
        self.undo_stack.append(self.base_image.copy())
        self.base_image = cropped
        self.params.reset()
        self._sync_widgets_from_params()

        self._clear_crop_box()
        self.crop_mode_active = False
        self.canvas.config(cursor="arrow")
        self.crop_toggle_btn.config(text="Start Crop", bg=COLOR_PANEL_ALT)

        self._render_to_canvas(self.base_image)
        self.stats_labels["resolution"].config(text=f"{self.base_image.shape[1]}x{self.base_image.shape[0]}")
        self.status_var.set("Crop applied.")

    # ----------------------------------------------------------------- #
    # Undo / Reset
    # ----------------------------------------------------------------- #
    def _sync_widgets_from_params(self):
        self.brightness_var.set(self.params.brightness)
        self.contrast_var.set(self.params.contrast)
        self.saturation_var.set(self.params.saturation)
        self.filter_var.set(self.params.filter_name)
        self.strength_var.set(self.params.filter_strength)
        self.angle_var.set(self.params.fine_angle)

    def _undo(self):
        if self.mode != "photo":
            return
        if self.undo_stack:
            self.base_image = self.undo_stack.pop()
            self.params.reset()
            self._sync_widgets_from_params()
            self._render_to_canvas(self.base_image)
            self.status_var.set("Undo applied.")
        else:
            messagebox.showinfo("Undo", "Nothing to undo.")

    def _reset_all(self):
        if self.mode == "photo" and self.initial_image is not None:
            self.base_image = self.initial_image.copy()
            self.undo_stack = []
            self.params.reset()
            self._sync_widgets_from_params()
            self._render_to_canvas(self.base_image)
            self.status_var.set("Reset to original photo.")
        elif self.mode == "video":
            self.params.reset()
            self._sync_widgets_from_params()
            self._render_video_position(self.position_var.get())
            self.status_var.set("Video edits reset.")
        elif self.mode == "webcam":
            self.params.reset()
            self._sync_widgets_from_params()

    # ----------------------------------------------------------------- #
    # Rendering
    # ----------------------------------------------------------------- #
    def _update_fps(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = self.fps * 0.8 + instant_fps * 0.2 if self.fps else instant_fps
        self.stats_labels["fps"].config(text=f"{self.fps:.1f}")

    def _render_to_canvas(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        w, h = pil_img.size
        scale = min(self.DISPLAY_MAX_W / w, self.DISPLAY_MAX_H / h, 1.0)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        pil_img_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self.display_photo = ImageTk.PhotoImage(pil_img_resized)
        self.canvas.delete("all")
        canvas_w = self.canvas.winfo_width() or self.DISPLAY_MAX_W
        canvas_h = self.canvas.winfo_height() or self.DISPLAY_MAX_H
        x = max(canvas_w // 2, new_w // 2)
        y = max(canvas_h // 2, new_h // 2)
        self.canvas.create_image(x, y, image=self.display_photo, anchor="center")

        # Remember mapping for crop coordinate conversion
        self.render_scale = scale
        self.render_offset = (x - new_w // 2, y - new_h // 2)
        self.render_img_size = (w, h)

    # ----------------------------------------------------------------- #
    # Save / Export
    # ----------------------------------------------------------------- #
    def save_or_export(self):
        if self.mode == "photo":
            self._save_photo()
        elif self.mode == "video":
            self._export_video()
        else:
            messagebox.showinfo("Nothing to save", "Capture/upload a photo, or upload a video, first.")

    def _save_photo(self):
        if self.base_image is None:
            return
        self._sync_params_from_widgets()
        out = apply_pipeline(self.base_image, self.params)
        path = filedialog.asksaveasfilename(
            title="Save edited photo",
            initialdir=OUTPUT_DIR,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")],
        )
        if not path:
            return
        if cv2.imwrite(path, out):
            messagebox.showinfo("Saved", f"Photo saved to:\n{path}")
            self.status_var.set(f"Saved: {os.path.basename(path)}")
        else:
            messagebox.showerror("Error", "Failed to save the photo.")

    def _export_video(self):
        if self.video_path is None:
            return
        start_f = min(getattr(self, "trim_start_frame", 0), getattr(self, "trim_end_frame", self.video_total_frames))
        end_f = max(getattr(self, "trim_start_frame", 0), getattr(self, "trim_end_frame", self.video_total_frames))

        out_path = filedialog.asksaveasfilename(
            title="Export edited video",
            initialdir=OUTPUT_DIR,
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All files", "*.*")],
        )
        if not out_path:
            return

        self._sync_params_from_widgets()
        params_snapshot = self.params

        export_cap = cv2.VideoCapture(self.video_path)
        export_cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        ret, first_frame = export_cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not read video for export.")
            export_cap.release()
            return

        sample_out = apply_pipeline(first_frame, params_snapshot)
        h, w = sample_out.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, self.video_fps, (w, h))
        writer.write(sample_out)

        total = max(end_f - start_f, 1)
        frame_idx = start_f + 1
        self.status_var.set(f"Exporting frame 1/{total} ...")
        self.root.update_idletasks()

        while frame_idx <= end_f:
            ret, frame = export_cap.read()
            if not ret:
                break
            out = apply_pipeline(frame, params_snapshot)
            if out.shape[:2] != (h, w):
                out = cv2.resize(out, (w, h))
            writer.write(out)

            if (frame_idx - start_f) % 10 == 0:
                self.status_var.set(f"Exporting frame {frame_idx - start_f + 1}/{total} ...")
                self.root.update_idletasks()

            frame_idx += 1

        writer.release()
        export_cap.release()
        messagebox.showinfo("Exported", f"Video exported to:\n{out_path}")
        self.status_var.set(f"Exported: {os.path.basename(out_path)}")

    # ----------------------------------------------------------------- #
    def on_close(self):
        self.stop_source()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SnapStudio(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()