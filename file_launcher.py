"""
File Launcher
=============
A Tkinter app that lets you browse for any file (document, image,
video, audio, etc.) and open/launch it with whatever application your
operating system has associated with that file type -- the same as
double-clicking it in File Explorer / Finder / your file manager.

Keeps a "Recent Files" list (saved to a small JSON file next to this
script) so you can quickly relaunch something you opened before.

Dependencies: none beyond the standard library.

Run:
    python file_launcher.py
"""

import json
import os
import platform
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

RECENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recent_files.json")
MAX_RECENTS = 15


def open_file_with_default_app(path):
    """Launch `path` with whatever the OS considers its default application.

    Raises OSError / subprocess errors on failure so the caller can show
    a message box.
    """
    system = platform.system()
    if system == "Windows":
        os.startfile(path)  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.run(["open", path], check=True)
    else:
        # Most Linux desktop environments
        subprocess.run(["xdg-open", path], check=True)


def load_recents():
    if os.path.exists(RECENTS_PATH):
        try:
            with open(RECENTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [p for p in data if isinstance(p, str)]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_recents(recents):
    try:
        with open(RECENTS_PATH, "w", encoding="utf-8") as f:
            json.dump(recents, f, indent=2)
    except OSError:
        pass


class FileLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("File Launcher")
        self.root.geometry("620x480")
        self.root.minsize(480, 380)

        self.recents = load_recents()
        self.selected_path_var = tk.StringVar(value="No file selected.")

        self._build_ui()
        self._refresh_recents_list()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=12)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Browse...", command=self.browse_file).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.selected_path_var, foreground="gray").pack(
            side=tk.LEFT, padx=10, fill=tk.X, expand=True
        )

        action_row = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        action_row.pack(side=tk.TOP, fill=tk.X)
        self.open_btn = ttk.Button(action_row, text="Open File", command=self.open_selected, state="disabled")
        self.open_btn.pack(side=tk.LEFT)
        ttk.Button(action_row, text="Clear Recents", command=self.clear_recents).pack(side=tk.LEFT, padx=8)

        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, padx=12)

        list_frame = ttk.Frame(self.root, padding=12)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(list_frame, text="Recent Files (double-click to open):").pack(anchor="w")

        self.recents_list = tk.Listbox(list_frame)
        self.recents_list.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(6, 0))
        self.recents_list.bind("<Double-Button-1>", self._on_recent_double_click)

        self.status_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status_var, foreground="gray", padding=(12, 0, 12, 10)).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    # ----------------------------------------------------------------- #
    def browse_file(self):
        path = filedialog.askopenfilename(title="Select a file to launch")
        if not path:
            return
        self.selected_path_var.set(path)
        self.open_btn.config(state="normal")

    def open_selected(self):
        path = self.selected_path_var.get()
        if not path or path == "No file selected.":
            return
        self._launch(path)

    def _on_recent_double_click(self, event):
        idx = self.recents_list.curselection()
        if not idx:
            return
        path = self.recents[idx[0]]
        self._launch(path)

    def _launch(self, path):
        if not os.path.exists(path):
            messagebox.showerror("Not Found", f"This file no longer exists:\n{path}")
            self._remove_from_recents(path)
            return
        try:
            open_file_with_default_app(path)
            self.status_var.set(f"Opened: {path}")
            self._add_to_recents(path)
        except (OSError, subprocess.CalledProcessError) as e:
            messagebox.showerror("Could Not Open File", f"{path}\n\n{e}")

    # ----------------------------------------------------------------- #
    def _add_to_recents(self, path):
        if path in self.recents:
            self.recents.remove(path)
        self.recents.insert(0, path)
        self.recents = self.recents[:MAX_RECENTS]
        save_recents(self.recents)
        self._refresh_recents_list()

    def _remove_from_recents(self, path):
        if path in self.recents:
            self.recents.remove(path)
            save_recents(self.recents)
            self._refresh_recents_list()

    def clear_recents(self):
        self.recents = []
        save_recents(self.recents)
        self._refresh_recents_list()

    def _refresh_recents_list(self):
        self.recents_list.delete(0, tk.END)
        for path in self.recents:
            self.recents_list.insert(tk.END, path)


def main():
    root = tk.Tk()
    FileLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
