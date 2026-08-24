"""Python-side bridges to the C and Ruby layers.

Both bridges are deliberately kept "pay once, call many times":

- ``CVelocityEngine`` loads ``fraud_core.dll``/``.so`` once via ``ctypes``
  and calls into it directly (no subprocess) -- this is what actually makes
  the C layer sub-millisecond: an in-process native call, not IPC.
- ``RubyRulesEngine`` spawns ``ruby rules_engine.rb --server`` ONCE and
  keeps it alive for the life of the bridge, talking to it over
  newline-delimited JSON on its stdin/stdout. A fresh `ruby rules_engine.rb`
  process per transaction would cost tens of milliseconds just for Ruby's
  own interpreter startup -- see README's architecture/latency section for
  the measured difference between the two approaches.

``TransactionContext``/``VelocityMetrics`` mirror the structs in
``src/c/fraud_core.h`` field-for-field; changing one without the other will
silently corrupt data across the FFI boundary.
"""
from __future__ import annotations

import ctypes
import json
import pathlib
import platform
import shutil
import subprocess
import threading

ROOT = pathlib.Path(__file__).resolve().parents[2]

# 30 days in seconds -- mirrors FRAUD_NO_HISTORY_SECONDS in fraud_core.h.
NO_HISTORY_SECONDS = 30 * 24 * 3600.0


class TransactionContext(ctypes.Structure):
    _fields_ = [
        ("current_lat", ctypes.c_double),
        ("current_lon", ctypes.c_double),
        ("prev_lat", ctypes.c_double),
        ("prev_lon", ctypes.c_double),
        ("has_prev", ctypes.c_int),
        ("seconds_since_prev", ctypes.c_double),
        ("amount_clp", ctypes.c_double),
        ("hist_mean_amount", ctypes.c_double),
        ("hist_std_amount", ctypes.c_double),
        ("txn_count_last_1h", ctypes.c_long),
        ("txn_count_last_24h", ctypes.c_long),
    ]


class VelocityMetrics(ctypes.Structure):
    _fields_ = [
        ("distance_from_prev_km", ctypes.c_double),
        ("implied_speed_kmh", ctypes.c_double),
        ("is_impossible_travel", ctypes.c_int),
        ("amount_zscore", ctypes.c_double),
        ("velocity_score", ctypes.c_double),
    ]


def _default_library_path() -> pathlib.Path:
    name = "fraud_core.dll" if platform.system() == "Windows" else "libfraud_core.so"
    return ROOT / "outputs" / "models" / name


class CVelocityEngine:
    """Thin ctypes wrapper around ``compute_velocity_metrics`` in fraud_core."""

    def __init__(self, library_path: pathlib.Path | None = None):
        self.library_path = library_path or _default_library_path()
        if not self.library_path.exists():
            raise FileNotFoundError(
                f"{self.library_path} not found. Build it first: make -C src/c build"
            )
        self._lib = ctypes.CDLL(str(self.library_path))
        self._lib.compute_velocity_metrics.argtypes = [
            ctypes.POINTER(TransactionContext),
            ctypes.POINTER(VelocityMetrics),
        ]
        self._lib.compute_velocity_metrics.restype = None

    def compute(
        self,
        current_lat: float,
        current_lon: float,
        amount_clp: float,
        prev_lat: float | None = None,
        prev_lon: float | None = None,
        seconds_since_prev: float | None = None,
        hist_mean_amount: float = 0.0,
        hist_std_amount: float = 0.0,
        txn_count_last_1h: int = 0,
        txn_count_last_24h: int = 0,
    ) -> dict:
        has_prev = prev_lat is not None and prev_lon is not None
        ctx = TransactionContext(
            current_lat=current_lat,
            current_lon=current_lon,
            prev_lat=prev_lat or 0.0,
            prev_lon=prev_lon or 0.0,
            has_prev=1 if has_prev else 0,
            seconds_since_prev=seconds_since_prev if seconds_since_prev is not None else NO_HISTORY_SECONDS,
            amount_clp=amount_clp,
            hist_mean_amount=hist_mean_amount,
            hist_std_amount=hist_std_amount,
            txn_count_last_1h=txn_count_last_1h,
            txn_count_last_24h=txn_count_last_24h,
        )
        out = VelocityMetrics()
        self._lib.compute_velocity_metrics(ctypes.byref(ctx), ctypes.byref(out))

        return {
            "distance_from_prev_km": out.distance_from_prev_km,
            "implied_speed_kmh": out.implied_speed_kmh,
            "is_impossible_travel": bool(out.is_impossible_travel),
            "amount_zscore": out.amount_zscore,
            "velocity_score": out.velocity_score,
        }


def _default_ruby_executable() -> str:
    found = shutil.which("ruby")
    if found:
        return found
    fallback = pathlib.Path("C:/Ruby32-x64/bin/ruby.exe")
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("ruby executable not found on PATH or at the default install location")


class RubyRulesEngine:
    """Keeps a single `ruby rules_engine.rb --server` process alive and
    exchanges one JSON line per transaction over its stdin/stdout."""

    def __init__(self, ruby_executable: str | None = None, script_path: pathlib.Path | None = None):
        self.ruby_executable = ruby_executable or _default_ruby_executable()
        self.script_path = script_path or (ROOT / "src" / "ruby" / "rules_engine.rb")
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            [self.ruby_executable, str(self.script_path), "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

    def evaluate(self, transaction: dict) -> dict:
        with self._lock:
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(f"Ruby rules engine process died: {stderr}")

            self._proc.stdin.write(json.dumps(transaction) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                stderr = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(f"Ruby rules engine produced no output: {stderr}")
            return json.loads(line)

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
