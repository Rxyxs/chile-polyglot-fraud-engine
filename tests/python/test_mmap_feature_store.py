import pathlib
import platform
import subprocess
import time

import pytest

from src.python.bridge import CVelocityEngine
from src.python.mmap_feature_store_client import MmapFeatureStoreClient

ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVER_EXE = ROOT / "outputs" / "models" / "feature_store_server.exe"

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows", reason="MmapFeatureStoreClient uses CreateFileMapping (Windows only)."
)


@pytest.fixture(scope="module")
def running_server():
    if not SERVER_EXE.exists():
        pytest.skip(f"{SERVER_EXE} no existe. Compila primero: make -C src/c build")

    proc = subprocess.Popen([str(SERVER_EXE)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # El servidor imprime una linea "ready" al iniciar; darle un instante
    # para crear el mapping antes de que el cliente intente abrirlo.
    time.sleep(0.3)
    if proc.poll() is not None:
        pytest.fail(f"feature_store_server.exe termino inmediatamente: {proc.stdout.read()}")
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


@pytest.fixture()
def mmap_client(running_server):
    with MmapFeatureStoreClient() as client:
        yield client


@pytest.fixture(scope="module")
def c_engine():
    return CVelocityEngine()


def test_mmap_client_matches_ctypes_engine_for_impossible_travel(mmap_client, c_engine):
    kwargs = dict(
        current_lat=-18.47, current_lon=-70.30, amount_clp=500,
        prev_lat=-33.45, prev_lon=-70.66, seconds_since_prev=60,
        hist_mean_amount=10000, hist_std_amount=3000,
        txn_count_last_1h=4, txn_count_last_24h=6,
    )
    ctypes_result = c_engine.compute(**kwargs)
    mmap_result = mmap_client.compute(**kwargs)
    assert mmap_result == ctypes_result


def test_mmap_client_matches_ctypes_engine_for_first_transaction(mmap_client, c_engine):
    kwargs = dict(current_lat=-33.45, current_lon=-70.66, amount_clp=10000)
    assert mmap_client.compute(**kwargs) == c_engine.compute(**kwargs)


def test_mmap_client_matches_ctypes_engine_for_capped_zscore(mmap_client, c_engine):
    kwargs = dict(
        current_lat=-33.45, current_lon=-70.66, amount_clp=50_000_000,
        hist_mean_amount=10000, hist_std_amount=100,
    )
    assert mmap_client.compute(**kwargs) == c_engine.compute(**kwargs)


def test_mmap_client_survives_many_sequential_calls(mmap_client):
    # Guarda de regresion para el protocolo de 3 estados (idle/request/response):
    # un handshake que se desincroniza tras la primera llamada se notaria
    # aqui como un TimeoutError o resultados incorrectos en llamadas tardias.
    for i in range(200):
        out = mmap_client.compute(
            current_lat=-33.45, current_lon=-70.66, amount_clp=1000 + i,
            hist_mean_amount=5000, hist_std_amount=1000,
        )
        assert out["amount_zscore"] < 0  # monto bajo el promedio historico en todas las iteraciones
