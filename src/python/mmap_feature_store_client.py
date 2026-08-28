"""Python client for the C ``feature_store_server`` (see src/c/feature_store_server.c),
speaking a raw shared-memory IPC channel -- a genuinely different transport
from ``CVelocityEngine`` (in-process ctypes DLL call, no IPC at all) and
``RubyRulesEngine`` (a subprocess pipe over stdin/stdout, see bridge.py).

``ctypes.Structure.from_buffer`` overlays a ctypes struct directly onto the
mmap'd bytes -- reads/writes to the Python object are reads/writes into the
same physical memory the C server reads/writes, with no serialization step
(no JSON, no protobuf) on either side. That's the appeal of shared-memory
IPC over a socket/pipe transport: zero marshalling cost per call, at the
price of a hand-rolled synchronization protocol (see the module docstring
in feature_store_server.c for the scope this channel is -- and isn't --
built for: one client at a time, busy-wait handshake, no auth/framing).
"""
from __future__ import annotations

import ctypes
import mmap
import platform
import time

from .bridge import TransactionContext, VelocityMetrics

CHANNEL_NAME = "FraudFeatureStoreChannel"
NO_HISTORY_SECONDS = 30 * 24 * 3600.0

# Estado 0=libre esperando request, 1=request escrito por el cliente,
# 2=response escrito por el servidor, 3=señal de apagado.
STATE_IDLE = 0
STATE_REQUEST_READY = 1
STATE_RESPONSE_READY = 2
STATE_SHUTDOWN = 3


class _FeatureStoreChannel(ctypes.Structure):
    # Reutiliza TransactionContext/VelocityMetrics de bridge.py TAL CUAL
    # (sin _pack_=1) -- alineacion natural, calzando con feature_store_server.c
    # (que tampoco usa #pragma pack). La primera version de este archivo
    # definia copias propias con _pack_=1, forzando byte-packing en el lado
    # Python mientras el struct de C mantenia alineacion natural -- el
    # resultado eran campos leidos desde offsets incorrectos (valores
    # denormales sin sentido), detectado comparando contra CVelocityEngine
    # para el mismo input, no por inspeccion del codigo.
    _fields_ = [
        ("state", ctypes.c_long),
        ("request", TransactionContext),
        ("response", VelocityMetrics),
    ]


class MmapFeatureStoreClient:
    """Conecta al canal de memoria compartida que abre `feature_store_server.exe`.

    El servidor debe estar corriendo antes de instanciar este cliente --
    a diferencia de `CVelocityEngine` (que carga la DLL en el mismo
    proceso), este es un proceso C completamente separado.
    """

    def __init__(self, channel_name: str = CHANNEL_NAME, timeout_seconds: float = 5.0):
        if platform.system() != "Windows":
            raise NotImplementedError(
                "MmapFeatureStoreClient usa CreateFileMapping (Windows). "
                "El equivalente POSIX (shm_open/mmap) no esta implementado en este proyecto."
            )
        self.timeout_seconds = timeout_seconds
        size = ctypes.sizeof(_FeatureStoreChannel)
        try:
            self._mmap = mmap.mmap(-1, size, tagname=channel_name)
        except OSError as exc:
            raise ConnectionError(
                f"No se pudo abrir el canal de memoria compartida '{channel_name}'. "
                "¿Esta corriendo feature_store_server.exe? "
                "(make -C src/c build && outputs/models/feature_store_server.exe)"
            ) from exc
        self._channel = _FeatureStoreChannel.from_buffer(self._mmap)

    def _wait_for_state(self, expected_state: int) -> None:
        deadline = time.perf_counter() + self.timeout_seconds
        while self._channel.state != expected_state:
            if time.perf_counter() > deadline:
                raise TimeoutError(
                    f"Timeout esperando state={expected_state} "
                    f"(state actual={self._channel.state}) -- ¿el servidor sigue vivo?"
                )

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

        self._wait_for_state(STATE_IDLE)

        req = self._channel.request
        req.current_lat = current_lat
        req.current_lon = current_lon
        req.prev_lat = prev_lat or 0.0
        req.prev_lon = prev_lon or 0.0
        req.has_prev = 1 if has_prev else 0
        req.seconds_since_prev = (
            seconds_since_prev if seconds_since_prev is not None else NO_HISTORY_SECONDS
        )
        req.amount_clp = amount_clp
        req.hist_mean_amount = hist_mean_amount
        req.hist_std_amount = hist_std_amount
        req.txn_count_last_1h = txn_count_last_1h
        req.txn_count_last_24h = txn_count_last_24h

        self._channel.state = STATE_REQUEST_READY
        self._wait_for_state(STATE_RESPONSE_READY)

        resp = self._channel.response
        result = {
            "distance_from_prev_km": resp.distance_from_prev_km,
            "implied_speed_kmh": resp.implied_speed_kmh,
            "is_impossible_travel": bool(resp.is_impossible_travel),
            "amount_zscore": resp.amount_zscore,
            "velocity_score": resp.velocity_score,
        }
        self._channel.state = STATE_IDLE
        return result

    def shutdown_server(self) -> None:
        """Le pide al servidor que termine limpiamente (state=3)."""
        self._wait_for_state(STATE_IDLE)
        self._channel.state = STATE_SHUTDOWN

    def close(self) -> None:
        # `_channel` was built with `ctypes.Structure.from_buffer(self._mmap)`,
        # which holds a live buffer-protocol export into the mmap -- closing
        # the mmap while that reference still exists raises `BufferError:
        # cannot close exported pointers exist`. Dropping the reference
        # first releases the export so close() can actually run.
        del self._channel
        self._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
