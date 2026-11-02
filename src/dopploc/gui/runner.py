"""Run dopploc commands as child processes with a streamed log (GLib main loop, no threads).

Heavy commands go through ``scripts/run-capped.sh`` when it exists, so an overrun kills only
the job (own cgroup scope with a memory cap) and never the desktop session.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from gi.repository import Gio, GLib

PKG_ROOT = Path(__file__).resolve().parents[3]  # source checkout: <root>/src/dopploc/gui
CAPPED = PKG_ROOT / "scripts" / "run-capped.sh"


def dopploc_argv(*args: str) -> list[str]:
    return [sys.executable, "-P", "-m", "dopploc.cli", *map(str, args)]


def capped(argv: list[str], cap: str | None) -> list[str]:
    if cap and CAPPED.exists() and shutil.which("systemd-run"):
        return [str(CAPPED), cap, *argv]
    return argv


class Job:
    """One child process. ``on_line(str)`` per output line, ``on_exit(int)`` once."""

    def __init__(self, argv: list[str], cwd: Path, on_line, on_exit):
        self.argv = argv
        self.on_line = on_line
        self.on_exit = on_exit
        launcher = Gio.SubprocessLauncher.new(Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_MERGE)
        launcher.set_cwd(str(cwd))
        launcher.setenv("PYTHONUNBUFFERED", "1", True)
        self.proc = launcher.spawnv(argv)
        self.stream = Gio.DataInputStream.new(self.proc.get_stdout_pipe())
        self.cancel = Gio.Cancellable()
        self._read()
        self.proc.wait_async(None, self._done)

    def _read(self):
        self.stream.read_line_async(GLib.PRIORITY_DEFAULT, self.cancel, self._line)

    def _line(self, stream, res):
        try:
            line, _ = stream.read_line_finish_utf8(res)
        except GLib.Error:
            return
        if line is None:
            return
        self.on_line(line)
        self._read()

    def _done(self, proc, res):
        try:
            proc.wait_finish(res)
        except GLib.Error:
            pass
        # drain what is still buffered, then report
        GLib.timeout_add(200, self._finish)

    def _finish(self):
        self.cancel.cancel()
        rc = self.proc.get_exit_status() if self.proc.get_if_exited() else -1
        self.on_exit(rc)
        return False

    def stop(self):
        """SIGTERM to the child (run-capped.sh forwards it to the scope)."""
        if self.proc.get_identifier():
            try:
                os.kill(int(self.proc.get_identifier()), 15)
            except (ProcessLookupError, ValueError):
                pass
