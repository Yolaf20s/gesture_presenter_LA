"""Tkinter GUI for the Gesture Presenter demo app."""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from src.pipeline import Pipeline
from src.pca_classifier import PCAClassifier


# ============================================================
# Design tokens
# ============================================================
COLOR_BG               = "#FFFFFF"
COLOR_BG_SECONDARY     = "#F8F9FA"
COLOR_BG_TERTIARY      = "#F1F3F5"
COLOR_BG_DARK          = "#15171c"
COLOR_BG_FRAME         = "#E5E7EB"

COLOR_TEXT_PRIMARY     = "#1A1A1A"
COLOR_TEXT_SECONDARY   = "#6C757D"
COLOR_TEXT_MUTED       = "#ADB5BD"

COLOR_ACCENT           = "#0D6EFD"
COLOR_SUCCESS          = "#198754"
COLOR_DANGER           = "#DC3545"
COLOR_DANGER_HOVER     = "#C82333"
COLOR_DANGER_ACTIVE    = "#A71D2A"
COLOR_INACTIVE         = "#ADB5BD"

COLOR_BORDER           = "#DEE2E6"
COLOR_BORDER_DARK      = "#6C757D"

COLOR_DISABLED_BG      = "#E9ECEF"
COLOR_DISABLED_FG      = "#9CA3AF"


GESTURE_COLORS = {
    'right_arm':    {'main': '#3B82F6', 'light': '#DBEAFE', 'dark': '#1D4ED8'},
    'left_arm':     {'main': '#8B5CF6', 'light': '#EDE9FE', 'dark': '#6D28D9'},
    'thumb_up':     {'main': '#22C55E', 'light': '#DCFCE7', 'dark': '#15803D'},
    'five_fingers': {'main': '#F59E0B', 'light': '#FEF3C7', 'dark': '#B45309'},
    'idle':         {'main': '#9CA3AF', 'light': '#F3F4F6', 'dark': '#4B5563'},
}

GESTURE_SYMBOLS = {
    'right_arm':    '→',
    'left_arm':     '←',
    'thumb_up':     '↑',
    'five_fingers': '5',
    'idle':         '·',
}

GESTURE_LABELS = {
    'right_arm':    'Next slide',
    'left_arm':     'Previous slide',
    'thumb_up':     'Confetti',
    'five_fingers': 'Countdown 5s',
    'idle':         'No gesture',
}

GESTURE_KEYS_ORDER = ['right_arm', 'left_arm', 'thumb_up', 'five_fingers']


FONT_FAMILY            = "Segoe UI"
FONT_TAB               = (FONT_FAMILY, 11)
FONT_REGULAR           = (FONT_FAMILY, 10)
FONT_SMALL             = (FONT_FAMILY, 9)
FONT_LABEL             = (FONT_FAMILY, 8, "bold")
FONT_BOLD              = (FONT_FAMILY, 11, "bold")
FONT_GESTURE           = (FONT_FAMILY, 22, "bold")
FONT_METRIC            = (FONT_FAMILY, 18, "bold")
FONT_TITLE             = (FONT_FAMILY, 18, "bold")
FONT_SUBTITLE          = (FONT_FAMILY, 11)
FONT_BUTTON            = (FONT_FAMILY, 12, "bold")


def _round_rect_points(x1, y1, x2, y2, r):
    return [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]


# ============================================================
# PillButton
# ============================================================
class PillButton(tk.Canvas):
    def __init__(self, parent, text="", icon_type=None,
                 variant="primary",
                 width=180, height=54, command=None,
                 parent_bg=COLOR_BG_SECONDARY):
        super().__init__(parent, width=width, height=height,
                         highlightthickness=0, bg=parent_bg)
        self.w = width
        self.h = height
        self.text = text
        self.icon_type = icon_type
        self.variant = variant
        self.command = command
        self.disabled = False

        if variant == 'primary':
            self.bg_normal = COLOR_DANGER
            self.bg_hover  = COLOR_DANGER_HOVER
            self.bg_active = COLOR_DANGER_ACTIVE
            self.fg = "white"
            self.border = ""
        else:
            self.bg_normal = COLOR_BG
            self.bg_hover  = COLOR_BG_TERTIARY
            self.bg_active = COLOR_BORDER
            self.fg = COLOR_TEXT_PRIMARY
            self.border = COLOR_BORDER_DARK

        self._draw(self.bg_normal, self.fg, self.border)

        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _draw(self, fill, text_color, border):
        self.delete("all")
        r = (self.h - 4) // 2
        pts_shadow = _round_rect_points(2, 4, self.w - 2, self.h - 0, r)
        self.create_polygon(pts_shadow, fill=COLOR_BORDER, smooth=True)
        pts = _round_rect_points(2, 2, self.w - 2, self.h - 4, r)
        self.create_polygon(pts, fill=fill,
                            outline=border if border else "",
                            width=1.5 if border else 0,
                            smooth=True)
        text_offset = 0
        if self.icon_type == 'play':
            cx, cy = 38, (self.h - 2) // 2
            s = 8
            self.create_polygon([cx - s, cy - s, cx - s, cy + s, cx + s + 1, cy],
                                fill=text_color, outline="")
            text_offset = 14
        elif self.icon_type == 'stop':
            cx, cy = 38, (self.h - 2) // 2
            s = 7
            self.create_rectangle(cx - s, cy - s, cx + s, cy + s,
                                  fill=text_color, outline="")
            text_offset = 14
        self.create_text(self.w // 2 + text_offset, (self.h - 2) // 2,
                         text=self.text, fill=text_color, font=FONT_BUTTON)

    def _on_press(self, e):
        if not self.disabled:
            self._draw(self.bg_active, self.fg, self.border)
    def _on_release(self, e):
        if not self.disabled and self.command:
            self._draw(self.bg_hover, self.fg, self.border)
            self.command()
    def _on_enter(self, e):
        if not self.disabled:
            self._draw(self.bg_hover, self.fg, self.border)
            self.config(cursor="hand2")
    def _on_leave(self, e):
        if not self.disabled:
            self._draw(self.bg_normal, self.fg, self.border)
            self.config(cursor="")
    def set_enabled(self, enabled):
        self.disabled = not enabled
        if enabled:
            self._draw(self.bg_normal, self.fg, self.border)
        else:
            self._draw(COLOR_DISABLED_BG, COLOR_DISABLED_FG, "")


# ============================================================
# GestureBadge
# ============================================================
class GestureBadge(tk.Canvas):
    def __init__(self, parent, gesture_key, size=36, parent_bg=COLOR_BG):
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, bg=parent_bg)
        self.gesture_key = gesture_key
        self.size = size
        self._draw()

    def _draw(self):
        self.delete("all")
        colors = GESTURE_COLORS.get(self.gesture_key, GESTURE_COLORS['idle'])
        symbol = GESTURE_SYMBOLS.get(self.gesture_key, '·')
        self.create_oval(2, 2, self.size - 2, self.size - 2,
                         fill=colors['main'], outline="")
        font_size = max(10, int(self.size * 0.45))
        self.create_text(self.size // 2, self.size // 2 - 1,
                         text=symbol, fill="white",
                         font=(FONT_FAMILY, font_size, "bold"))

    def set_bg(self, bg):
        self.configure(bg=bg)

    def set_gesture(self, gesture_key):
        self.gesture_key = gesture_key
        self._draw()


# ============================================================
# StatCard
# ============================================================
class StatCard(tk.Frame):
    def __init__(self, parent, label, value="—"):
        super().__init__(parent, bg=COLOR_BG_TERTIARY, padx=14, pady=10)
        tk.Label(self, text=label.upper(),
                 fg=COLOR_TEXT_SECONDARY, bg=COLOR_BG_TERTIARY,
                 font=FONT_LABEL).pack(anchor=tk.W)
        self.value_label = tk.Label(self, text=value,
                 fg=COLOR_TEXT_PRIMARY, bg=COLOR_BG_TERTIARY,
                 font=FONT_METRIC)
        self.value_label.pack(anchor=tk.W, pady=(2, 0))
    def set_value(self, text):
        self.value_label.configure(text=text)


# ============================================================
# GestureRow
# ============================================================
class GestureRow(tk.Frame):
    def __init__(self, parent, gesture_key):
        super().__init__(parent, bg=COLOR_BG, padx=10, pady=6)
        self.gesture_key = gesture_key
        self.active = False
        self.icon = GestureBadge(self, gesture_key, size=32, parent_bg=COLOR_BG)
        self.icon.pack(side=tk.LEFT, padx=(0, 12))
        self.text_label = tk.Label(
            self, text=GESTURE_LABELS[gesture_key],
            fg=GESTURE_COLORS[gesture_key]['dark'], bg=COLOR_BG,
            font=FONT_REGULAR,
        )
        self.text_label.pack(side=tk.LEFT)

    def set_active(self, active):
        if active == self.active:
            return
        self.active = active
        colors = GESTURE_COLORS[self.gesture_key]
        if active:
            bg = colors['light']
            self.configure(bg=bg)
            self.icon.set_bg(bg)
            self.text_label.configure(bg=bg, fg=colors['dark'], font=FONT_BOLD)
        else:
            self.configure(bg=COLOR_BG)
            self.icon.set_bg(COLOR_BG)
            self.text_label.configure(bg=COLOR_BG,
                                      fg=colors['dark'],
                                      font=FONT_REGULAR)


# ============================================================
# Tab 1: Live
# ============================================================
class LiveTab(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=COLOR_BG)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        bottom = tk.Frame(self, bg=COLOR_BG_SECONDARY, height=110)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)
        bottom.pack_propagate(False)

        center = tk.Frame(bottom, bg=COLOR_BG_SECONDARY)
        center.pack(expand=True)

        self.start_button = PillButton(
            center, text="Start", icon_type='play',
            variant='primary', width=180, height=54,
            command=self.app.start, parent_bg=COLOR_BG_SECONDARY,
        )
        self.start_button.pack(side=tk.LEFT, padx=12)

        self.stop_button = PillButton(
            center, text="Stop", icon_type='stop',
            variant='secondary', width=180, height=54,
            command=self.app.stop, parent_bg=COLOR_BG_SECONDARY,
        )
        self.stop_button.pack(side=tk.LEFT, padx=12)
        self.stop_button.set_enabled(False)

        body = tk.Frame(self, bg=COLOR_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        left = tk.Frame(body, bg=COLOR_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        video_outer = tk.Frame(left, bg=COLOR_BG_FRAME, padx=8, pady=8)
        video_outer.pack(fill=tk.BOTH, expand=True)
        video_inner = tk.Frame(video_outer, bg=COLOR_BORDER_DARK, padx=2, pady=2)
        video_inner.pack(fill=tk.BOTH, expand=True)
        self.video_canvas = tk.Label(
            video_inner, bg=COLOR_BG_DARK,
            text="Press START to begin",
            fg="#6c757d", font=FONT_REGULAR,
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        right_container = tk.Frame(body, bg=COLOR_BG, width=380)
        right_container.pack(side=tk.RIGHT, fill=tk.Y)
        right_container.pack_propagate(False)

        scroll_canvas = tk.Canvas(right_container, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_container, orient=tk.VERTICAL,
                                  command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        right = tk.Frame(scroll_canvas, bg=COLOR_BG)
        canvas_window = scroll_canvas.create_window((0, 0), window=right, anchor='nw')
        right.bind("<Configure>",
                   lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.bind("<Configure>",
                           lambda e: scroll_canvas.itemconfig(canvas_window, width=e.width))

        def _on_mw(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        scroll_canvas.bind("<Enter>", lambda e: scroll_canvas.bind_all("<MouseWheel>", _on_mw))
        scroll_canvas.bind("<Leave>", lambda e: scroll_canvas.unbind_all("<MouseWheel>"))

        self._build_right_content(right)

    def _build_right_content(self, parent):
        status_row = tk.Frame(parent, bg=COLOR_BG)
        status_row.pack(fill=tk.X, pady=(0, 18), padx=4)
        self.status_dot = tk.Canvas(status_row, width=12, height=12,
                                     highlightthickness=0, bg=COLOR_BG)
        self.status_dot.pack(side=tk.LEFT, padx=(0, 8))
        self._draw_dot(COLOR_INACTIVE)
        self.status_label = tk.Label(status_row, text="Stopped",
            fg=COLOR_TEXT_SECONDARY, bg=COLOR_BG, font=FONT_BOLD)
        self.status_label.pack(side=tk.LEFT)

        tk.Label(parent, text="CURRENT GESTURE",
                 fg=COLOR_TEXT_MUTED, bg=COLOR_BG,
                 font=FONT_LABEL).pack(anchor=tk.W, padx=4)

        gesture_row = tk.Frame(parent, bg=COLOR_BG)
        gesture_row.pack(fill=tk.X, pady=(6, 16), padx=4)
        self.gesture_badge = GestureBadge(gesture_row, 'idle', size=46, parent_bg=COLOR_BG)
        self.gesture_badge.pack(side=tk.LEFT, padx=(0, 12))
        self.gesture_label = tk.Label(gesture_row, text="No gesture",
                 fg=COLOR_TEXT_SECONDARY, bg=COLOR_BG, font=FONT_GESTURE)
        self.gesture_label.pack(side=tk.LEFT)

        tk.Label(parent, text="CONFIDENCE",
                 fg=COLOR_TEXT_MUTED, bg=COLOR_BG,
                 font=FONT_LABEL).pack(anchor=tk.W, padx=4)
        conf_row = tk.Frame(parent, bg=COLOR_BG)
        conf_row.pack(fill=tk.X, pady=(6, 16), padx=4)
        self.conf_progress = ttk.Progressbar(conf_row, mode='determinate', maximum=100,
            style='Accent.Horizontal.TProgressbar')
        self.conf_progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.conf_label = tk.Label(conf_row, text="0%",
                 fg=COLOR_TEXT_PRIMARY, bg=COLOR_BG,
                 font=FONT_BOLD, width=5)
        self.conf_label.pack(side=tk.LEFT, padx=(10, 0))

        stats_row = tk.Frame(parent, bg=COLOR_BG)
        stats_row.pack(fill=tk.X, pady=(0, 18), padx=4)
        self.fps_card = StatCard(stats_row, "FPS", "0.0")
        self.fps_card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.cooldown_card = StatCard(stats_row, "Cooldown", "0.0s")
        self.cooldown_card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        tk.Label(parent, text="ALL GESTURES",
                 fg=COLOR_TEXT_MUTED, bg=COLOR_BG,
                 font=FONT_LABEL).pack(anchor=tk.W, pady=(0, 4), padx=4)
        self.gesture_rows = {}
        for key in GESTURE_KEYS_ORDER:
            row = GestureRow(parent, key)
            row.pack(fill=tk.X, padx=4, pady=2)
            self.gesture_rows[key] = row

        tk.Label(parent, text="LAST ACTION",
                 fg=COLOR_TEXT_MUTED, bg=COLOR_BG,
                 font=FONT_LABEL).pack(anchor=tk.W, pady=(18, 4), padx=4)
        self.last_action_label = tk.Label(parent, text="—",
                 fg=COLOR_TEXT_SECONDARY, bg=COLOR_BG,
                 font=FONT_SMALL, wraplength=340,
                 justify=tk.LEFT, anchor=tk.W)
        self.last_action_label.pack(anchor=tk.W, fill=tk.X, padx=4, pady=(0, 14))

    def _draw_dot(self, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(0, 0, 12, 12, fill=color, outline="")

    def update_video(self, photo):
        self.video_canvas.configure(image=photo, text="")
        self.video_canvas.image = photo
    def clear_video(self, message="Press START to begin"):
        self.video_canvas.configure(image="", text=message)
        self.video_canvas.image = None
    def update_status(self, text, color):
        self.status_label.configure(text=text, fg=color)
        self._draw_dot(color)
    def update_gesture(self, gesture_key, tracking_state="ready"):
        display_key = gesture_key if tracking_state == "ready" else "idle"
        self.gesture_badge.set_gesture(display_key)
        if tracking_state != "ready":
            label = config.TRACKING_DISPLAY_NAME.get(tracking_state, tracking_state)
            self.gesture_label.configure(text=label, fg=COLOR_TEXT_SECONDARY)
        else:
            label = GESTURE_LABELS.get(gesture_key, gesture_key)
            if gesture_key == 'idle':
                self.gesture_label.configure(text=label, fg=COLOR_TEXT_SECONDARY)
            else:
                self.gesture_label.configure(text=label, fg=GESTURE_COLORS[gesture_key]['dark'])
        for key, row in self.gesture_rows.items():
            row.set_active(tracking_state == "ready" and key == gesture_key)
    def update_confidence(self, percent):
        self.conf_progress['value'] = percent
        self.conf_label.configure(text=f"{percent:.0f}%")
    def update_metrics(self, fps, cooldown):
        self.fps_card.set_value(f"{fps:.1f}")
        self.cooldown_card.set_value(f"{cooldown:.1f}s")
    def update_last_action(self, message):
        if message:
            self.last_action_label.configure(text=message)
    def set_running(self, is_running):
        if is_running:
            self.start_button.set_enabled(False)
            self.stop_button.set_enabled(True)
        else:
            self.start_button.set_enabled(True)
            self.stop_button.set_enabled(False)


class PCASpaceSection(tk.Frame):
    """Live position of gesture in PCA eigenspace (2D projection)."""

    def __init__(self, parent):
        super().__init__(parent, bg=COLOR_BG)
        self._last_update = 0.0
        self.canvas = None
        self.current_scatter = None

        try:
            self._init_chart()
        except Exception as e:
            self._show_error(f"{type(e).__name__}: {e}")

    def _show_error(self, msg):
        for w in self.winfo_children():
            w.destroy()
        wrap = tk.Frame(self, bg=COLOR_BG)
        wrap.pack(expand=True, padx=40, pady=40)
        tk.Label(wrap, text="PCA Space unavailable",
                 fg=COLOR_DANGER, bg=COLOR_BG,
                 font=FONT_TITLE).pack(pady=(0, 10))
        tk.Label(wrap, text=msg,
                 fg=COLOR_TEXT_SECONDARY, bg=COLOR_BG,
                 font=FONT_SMALL, wraplength=600,
                 justify=tk.LEFT).pack(pady=(0, 14))
        tk.Label(wrap, text="Make sure pca_model.npz exists "
                            "and data/*.npy training files are present.",
                 fg=COLOR_TEXT_MUTED, bg=COLOR_BG,
                 font=FONT_SMALL, wraplength=600,
                 justify=tk.LEFT).pack()

    def _init_chart(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        classifier = PCAClassifier.load()
        if classifier.train_projections is None or classifier.train_labels is None:
            raise ValueError("Loaded model does not contain training projections.")
        if classifier.train_projections.shape[1] < 2:
            raise ValueError("The loaded PCA model needs at least 2 components.")

        X_proj = classifier.train_projections
        y_all = classifier.train_labels
        ratios = classifier.explained_variance_ratio
        pc1_var = ratios[0] * 100
        pc2_var = ratios[1] * 100

        # Header
        header = tk.Frame(self, bg=COLOR_BG)
        header.pack(fill=tk.X, padx=24, pady=(20, 6))
        tk.Label(header, text="Gesture in PCA Eigenspace",
                 fg=COLOR_TEXT_PRIMARY, bg=COLOR_BG,
                 font=FONT_TITLE, anchor=tk.W).pack(anchor=tk.W)
        tk.Label(header,
                 text=f"Each point is one training sample from the loaded model "
                      f"projected onto PC1 ({pc1_var:.1f}% var) × "
                      f"PC2 ({pc2_var:.1f}% var). "
                      f"⭐ = your current gesture, live.",
                 fg=COLOR_TEXT_SECONDARY, bg=COLOR_BG,
                 font=FONT_SUBTITLE, anchor=tk.W,
                 wraplength=900, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

        # Matplotlib figure
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=COLOR_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLOR_BG_SECONDARY)

        for gesture in GESTURE_KEYS_ORDER + ['idle']:
            mask = (y_all == gesture)
            if not mask.any():
                continue
            color = GESTURE_COLORS[gesture]['main']
            label = GESTURE_LABELS.get(gesture, gesture)
            ax.scatter(X_proj[mask, 0], X_proj[mask, 1],
                       c=color, label=label, alpha=0.5, s=40,
                       edgecolors='white', linewidths=0.5)

        self.current_scatter = ax.scatter([], [], c='#FF1744', s=380,
                                          marker='*',
                                          edgecolors='black', linewidths=1.5,
                                          label='Current', zorder=10)

        ax.set_xlabel(f'PC1 ({pc1_var:.1f}% variance)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pc2_var:.1f}% variance)', fontsize=10)
        ax.legend(loc='upper right', frameon=True, facecolor=COLOR_BG,
                  edgecolor=COLOR_BORDER, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()

        self.fig = fig
        self.ax = ax
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                          padx=24, pady=(0, 20))

    def update_current(self, features_proj):
        if self.current_scatter is None or self.canvas is None:
            return

        now = time.time()
        if now - self._last_update < 0.1:
            return
        self._last_update = now

        if features_proj is None or len(features_proj) < 2:
            self.current_scatter.set_offsets(np.empty((0, 2)))
        else:
            self.current_scatter.set_offsets(
                np.array([[features_proj[0], features_proj[1]]])
            )
        self.canvas.draw_idle()

class PlaceholderSection(tk.Frame):
    def __init__(self, parent, title, subtitle, phase="Phase 2"):
        super().__init__(parent, bg=COLOR_BG)
        wrap = tk.Frame(self, bg=COLOR_BG)
        wrap.pack(expand=True)
        tk.Label(wrap, text=title, fg=COLOR_TEXT_PRIMARY,
                 bg=COLOR_BG, font=FONT_TITLE).pack(pady=(0, 10))
        tk.Label(wrap, text=subtitle, fg=COLOR_TEXT_SECONDARY,
                 bg=COLOR_BG, font=FONT_REGULAR).pack(pady=(0, 18))
        tk.Label(wrap, text=f"Coming in {phase}",
                 fg=COLOR_ACCENT, bg=COLOR_BG, font=FONT_SMALL).pack()


# ============================================================
# Tab 2: Analytics — sidebar with sections
# ============================================================
class AnalyticsTab(tk.Frame):
    SECTIONS = [
        ("pca",   "PCA Space",    "Live position of your gesture in PCA eigenspace"),
        ("eigen", "Eigenvectors", "Top-5 eigenvectors as bar charts"),
        ("math",  "Math",         "Covariance matrix and spectral decomposition"),
        ("stats", "Stats",        "Confusion matrix and accuracy metrics"),
    ]

    def __init__(self, parent, app):
        super().__init__(parent, bg=COLOR_BG)
        self.app = app
        self.current = 'pca'
        self._build_ui()

    def _build_ui(self):
        sidebar = tk.Frame(self, bg=COLOR_BG_SECONDARY, width=220)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="ANALYTICS",
                 fg=COLOR_TEXT_MUTED, bg=COLOR_BG_SECONDARY,
                 font=FONT_LABEL,
                 anchor=tk.W, padx=22).pack(fill=tk.X, pady=(22, 14))

        self.section_rows = {}
        for key, name, _ in self.SECTIONS:
            is_active = (key == self.current)
            row = tk.Frame(sidebar, bg=COLOR_BG if is_active else COLOR_BG_SECONDARY)
            row.pack(fill=tk.X)
            btn = tk.Label(
                row, text=name,
                fg=COLOR_ACCENT if is_active else COLOR_TEXT_SECONDARY,
                bg=COLOR_BG if is_active else COLOR_BG_SECONDARY,
                font=FONT_BOLD if is_active else FONT_REGULAR,
                anchor=tk.W, padx=22, pady=11, cursor="hand2",
            )
            btn.pack(fill=tk.X)
            btn.bind("<Button-1>", lambda e, k=key: self._select(k))
            self.section_rows[key] = (row, btn)

        # Content area: stack frames using grid (use tkraise to switch)
        content = tk.Frame(self, bg=COLOR_BG)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        self.sections = {}
        for key, name, desc in self.SECTIONS:
            if key == 'pca':
                section = PCASpaceSection(content)
            else:
                section = PlaceholderSection(content, name, desc)
            section.grid(row=0, column=0, sticky='nsew')
            self.sections[key] = section

        self.sections[self.current].tkraise()

    def _select(self, key):
        self.current = key
        for k, (row, btn) in self.section_rows.items():
            if k == key:
                row.configure(bg=COLOR_BG)
                btn.configure(fg=COLOR_ACCENT, bg=COLOR_BG, font=FONT_BOLD)
            else:
                row.configure(bg=COLOR_BG_SECONDARY)
                btn.configure(fg=COLOR_TEXT_SECONDARY,
                              bg=COLOR_BG_SECONDARY, font=FONT_REGULAR)
        self.sections[key].tkraise()

    def update_projection(self, features_proj):
        """Forward live projection to PCA section (only one that uses it)."""
        pca = self.sections.get('pca')
        if pca and hasattr(pca, 'update_current'):
            pca.update_current(features_proj)


class PlaceholderTab(tk.Frame):
    def __init__(self, parent, title, subtitle, phase="Phase 3"):
        super().__init__(parent, bg=COLOR_BG)
        wrap = tk.Frame(self, bg=COLOR_BG)
        wrap.pack(expand=True)
        tk.Label(wrap, text=title, fg=COLOR_TEXT_PRIMARY,
                 bg=COLOR_BG, font=FONT_TITLE).pack(pady=(0, 10))
        tk.Label(wrap, text=subtitle, fg=COLOR_TEXT_SECONDARY,
                 bg=COLOR_BG, font=FONT_REGULAR).pack(pady=(0, 18))
        tk.Label(wrap, text=f"Coming in {phase}",
                 fg=COLOR_ACCENT, bg=COLOR_BG, font=FONT_SMALL).pack()


# ============================================================
# Main app
# ============================================================
class GesturePresenterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Presenter")
        self.root.geometry("1240x820")
        self.root.minsize(1100, 720)
        self.root.configure(bg=COLOR_BG)

        self.pipeline = None
        self.is_running = False
        self._update_job = None
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._result_queue = queue.Queue(maxsize=1)

        self._setup_styles()
        self._build_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        print("[App] Window opened. Ready.")

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        style.configure('TNotebook', background=COLOR_BG, borderwidth=0,
                        tabmargins=[16, 14, 0, 0])
        style.configure('TNotebook.Tab', padding=(22, 10), font=FONT_TAB,
                        background=COLOR_BG_SECONDARY,
                        foreground=COLOR_TEXT_SECONDARY,
                        borderwidth=0, focuscolor='')
        style.map('TNotebook.Tab',
                  background=[('selected', COLOR_BG),
                              ('active', COLOR_BG_TERTIARY)],
                  foreground=[('selected', COLOR_ACCENT),
                              ('active', COLOR_TEXT_PRIMARY)])
        style.configure('Accent.Horizontal.TProgressbar',
                        background=COLOR_ACCENT,
                        troughcolor=COLOR_BG_TERTIARY,
                        borderwidth=0, thickness=6,
                        lightcolor=COLOR_ACCENT, darkcolor=COLOR_ACCENT)

    def _build_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.live_tab = LiveTab(self.notebook, self)
        self.notebook.add(self.live_tab, text="Live")

        self.analytics_tab = AnalyticsTab(self.notebook, self)
        self.notebook.add(self.analytics_tab, text="Analytics")

        self.settings_tab = PlaceholderTab(
            self.notebook, "Settings",
            "Camera, classifier thresholds, and gesture mappings",
        )
        self.notebook.add(self.settings_tab, text="Settings")

        self.help_tab = PlaceholderTab(
            self.notebook, "Help",
            "Gesture guide, troubleshooting, and about",
        )
        self.notebook.add(self.help_tab, text="Help")

    def start(self):
        if self.is_running:
            return
        if self.pipeline is None:
            try:
                self.pipeline = Pipeline()
            except Exception as e:
                messagebox.showerror(
                    "Failed to start pipeline",
                    f"{type(e).__name__}: {e}",
                )
                return
        self._clear_result_queue()
        self._stop_event.clear()
        self.is_running = True
        self.live_tab.update_status("Running", COLOR_SUCCESS)
        self.live_tab.set_running(True)
        self._worker_thread = threading.Thread(
            target=self._pipeline_worker,
            name="gesture-pipeline-worker",
            daemon=True,
        )
        self._worker_thread.start()
        self._schedule_result_poll()

    def stop(self):
        self.is_running = False
        self._stop_event.set()
        if self._update_job:
            self.root.after_cancel(self._update_job)
            self._update_job = None
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._worker_thread = None
        if self.pipeline:
            self.pipeline.cleanup()
            self.pipeline = None
        self._clear_result_queue()

        self.live_tab.update_status("Stopped", COLOR_INACTIVE)
        self.live_tab.set_running(False)
        self.live_tab.clear_video()
        self.live_tab.update_gesture("idle", tracking_state="idle")
        self.live_tab.update_confidence(0)
        self.live_tab.update_metrics(0.0, 0.0)
        self.live_tab.update_last_action("—")
        self.analytics_tab.update_projection(None)

    def _schedule_result_poll(self):
        self._update_job = self.root.after(15, self._poll_results)

    def _pipeline_worker(self):
        try:
            while not self._stop_event.is_set() and self.pipeline is not None:
                result = self.pipeline.step(show_overlay=False)
                if result is None:
                    self._push_result({"error": "Failed to read a frame from the camera."})
                    break
                self._push_result(result)
        except Exception as exc:
            self._push_result({"error": f"{type(exc).__name__}: {exc}"})

    def _push_result(self, result):
        while True:
            try:
                self._result_queue.put_nowait(result)
                return
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    return

    def _clear_result_queue(self):
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                return

    def _poll_results(self):
        if not self.is_running:
            return
        latest = None
        while True:
            try:
                latest = self._result_queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            if "error" in latest:
                messagebox.showerror("Pipeline error", latest["error"])
                self.stop()
                return
            self._render_result(latest)

        self._schedule_result_poll()

    def _render_result(self, result):
        cw = self.live_tab.video_canvas.winfo_width()
        ch = self.live_tab.video_canvas.winfo_height()

        frame = result["frame"]
        if cw > 10 and ch > 10:
            h, w = frame.shape[:2]
            ratio = min(cw / w, ch / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        self.live_tab.update_video(photo)
        self.live_tab.update_gesture(
            result["gesture"],
            tracking_state=result.get("tracking_state", "ready"),
        )
        self.live_tab.update_confidence(result["confidence"] * 100)
        self.live_tab.update_metrics(result["fps"], result["cooldown_remaining"])

        if result["just_triggered"]:
            self.live_tab.update_last_action(result["trigger_message"])

        self.analytics_tab.update_projection(result.get("features_proj"))

    def _on_close(self):
        if self.pipeline:
            self.stop()
        print("[App] Closing...")
        self.root.destroy()


def main():
    root = tk.Tk()
    GesturePresenterApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
