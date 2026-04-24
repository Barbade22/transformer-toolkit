"""
RunLogger — Universal Training Monitor
=======================================
from runlogger import RunLogger

logger = RunLogger(
    base_url="http://localhost:8000",
    project_name="my-project",
    api_token="rl-...",
    run_name="run-1",
)
logger.log(step=100, loss=0.5, lr=0.001)
logger.finish()
"""

import asyncio
import hashlib
import hmac
import json
import threading
import time
from typing import Dict, List, Optional

import requests

try:
    import websockets
except ImportError:
    websockets = None

# ── optional GPU stats ────────────────────────────────────────
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
    pynvml.nvmlInit()
    _nvml = True
except Exception:
    _nvml = False

# ── optional CPU/RAM stats ────────────────────────────────────
try:
    import psutil
    _psutil = True
except Exception:
    _psutil = False


def _gpu_stats() -> dict:
    if not _nvml:
        return {}
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "gpu_util":      u.gpu,
            "gpu_mem_used":  m.used  // 1024 // 1024,
            "gpu_mem_total": m.total // 1024 // 1024,
        }
    except Exception:
        return {}


def _sys_stats() -> dict:
    if not _psutil:
        return {}
    try:
        vm = psutil.virtual_memory()
        return {
            "cpu_util":  psutil.cpu_percent(interval=None),
            "ram_used":  vm.used  // 1024 // 1024,
            "ram_total": vm.total // 1024 // 1024,
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
class RunLogger:
# ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        base_url:        str,
        api_token:       str,
        project_name:    str,
        run_name:        str,
        config:          Optional[Dict] = None,
        start_step:      int            = 0,
        metrics:         Optional[List] = None,
        tags:            Optional[List] = None,
        notes:           str            = "",
        log_system_stats: bool          = True,
    ):
        self.base          = base_url.rstrip("/")
        self.headers       = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
        self._log_system   = log_system_stats

        # internal state
        self._lock          = threading.Lock()
        self._pending       = None        # latest payload waiting to be sent
        self._log_count     = 0
        self._stop          = False
        self._ws            = None        # live websocket (set by sender loop)
        self._pause_flag    = False
        self._banned        = False
        self.api_token = api_token

        # ── fetch plan ────────────────────────────────────────
        try:
            r = requests.get(f"{self.base}/api/me/plan-config", headers=self.headers, timeout=5)
            plan = r.json() if r.ok else {}
        except Exception:
            plan = {}
        self._log_interval = float(plan.get("log_interval", 10))
        self._secret       = plan.get("client_secret", "")

        # ── resolve project ───────────────────────────────────
        r = requests.get(
            f"{self.base}/api/projects/by-name/{requests.utils.quote(project_name)}",
            headers=self.headers,
        )
        if r.status_code in (401, 403):
            raise RuntimeError(f"Invalid API token. Server: {r.text}")
        r.raise_for_status()
        project_id = r.json()["id"]
        print(f"[RunLogger] project={project_name} ({project_id})")

        # ── create run ────────────────────────────────────────
        r = requests.post(
            f"{self.base}/api/projects/{project_id}/runs",
            headers=self.headers,
            json={
                "name":       run_name,
                "config":     config     or {},
                "start_step": start_step,
                "metrics":    metrics    or [],
                "tags":       tags       or [],
                "notes":      notes,
            },
        )
        if r.status_code in (401, 403):
            raise RuntimeError(f"Could not create run. Server: {r.text}")
        r.raise_for_status()
        self.run_id = r.json()["id"]
        print(f"[RunLogger] run={run_name} ({self.run_id})")

        # ── start WebSocket sender (after run_id is set) ──────
        threading.Thread(target=self._ws_loop, daemon=True).start()

    # ── public API ────────────────────────────────────────────

    def log(self, step: int, **kwargs) -> bool:
        """Log a training step. Call every step — buffering handles the rest."""
        if self._banned:
            return False
        self._buffer(step, is_eval=False, **kwargs)
        # refresh plan every 5000 packets (~once per long run)
        if self._log_count > 0 and self._log_count % 5000 == 0:
            threading.Thread(target=self._refresh_plan, daemon=True).start()
        return True

    def log_eval(self, step: int, **kwargs) -> bool:
        """Log an eval/validation step — always sent, never rate-limited."""
        if self._banned:
            return False
        self._buffer(step, is_eval=True, **kwargs)
        return True

    def should_pause(self) -> bool:
        """Call in your training loop — returns True if dashboard requested pause."""
        if self._pause_flag:
            self._clear_pause()
            print("Deleted Pause")
            return True
        return False

    def finish(self, status: str = "completed") -> None:
        """Mark run as finished. Call at end of training."""
        if self._banned:
            return
        self._stop = True
        try:
            requests.patch(
                f"{self.base}/api/runs/{self.run_id}",
                headers=self.headers,
                json={"status": status},
                timeout=5,
            )
            print(f"[RunLogger] run marked as {status}")
        except Exception as e:
            print(f"[RunLogger] warning: could not mark run as {status}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.finish("crashed" if exc_type else "completed")
        return False

    # ── internal: buffer ──────────────────────────────────────

    def _buffer(self, step: int, is_eval: bool, **kwargs):
        """Build payload and put it in the single-slot buffer."""
        ts = time.time()
        payload = {
            "step":  int(step),
            "_ts":   ts,
            "_sig":  self._sign(ts),
            "_eval": is_eval,
            **_gpu_stats(),
            **(_sys_stats() if self._log_system else {}),
            **kwargs,
        }
        with self._lock:
            self._pending = payload

    def _sign(self, ts: float) -> str:
        if not self._secret:
            return ""
        msg = f"{self.run_id}:{ts:.3f}".encode()
        return hmac.new(self._secret.encode(), msg, hashlib.sha256).hexdigest()

    # ── internal: WebSocket sender loop ──────────────────────
    def _ws_loop(self):
        """Runs in a background thread. Sends buffered payloads over WebSocket."""
        
        async def _run():
            proto  = "wss" if self.base.startswith("https") else "ws"
            ws_url = f"{proto}://{self.base.split('://')[1]}/ws/ingest/{self.run_id}?token={self.api_token}"
            print(ws_url)

            # ── dynamic control from server ──────────────────────
            self._min_interval = self._log_interval
            self._ws_delay     = 0.0

            _last_send_t = 0.0
            _banned_permanently = False

            while not self._stop and not _banned_permanently:
                try:
                    async with websockets.connect(ws_url, additional_headers=self.headers) as ws:
                        self._ws = ws
                        print("[RunLogger] connected")

                        while not self._stop:

                            # ── RECEIVE control messages FIRST ──
                            try:
                                msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                                data = json.loads(msg)
                                # print(data)

                                ctrl = data.get("_control")
                                print(ctrl)
                                

                                if ctrl == "config":
                                    self._min_interval = float(data.get("min_interval", self._min_interval))
                                    self._ws_delay     = float(data.get("ws_delay", 0))
                                    print(f"[RunLogger] ⚙ config: interval={self._min_interval:.3f}s delay={self._ws_delay:.3f}s")

                                elif ctrl == "throttle_update":
                                    self._min_interval = float(data.get("min_interval", self._min_interval))
                                    self._ws_delay     = float(data.get("ws_delay", self._ws_delay))
                                    print(f"[RunLogger] ⚡ throttle: interval={self._min_interval:.3f}s")

                                elif ctrl == "pause":
                                    self._pause_flag = True
                                    print("\n[RunLogger] ⏸ paused")

                                elif ctrl == "resume":
                                    self._pause_flag = False
                                    print("\n[RunLogger] ▶ resumed")
                                elif ctrl == "banned":
                                    print("\n[RunLogger] ❌ ACCOUNT BANNED:", data.get("reason"))

                                    self._banned = True
                                    _banned_permanently = True
                                    self._stop = True   # 🔥 STOP WS LOOP
                                    raise RuntimeError("RunLogger: account banned due to misuse")
                                
                            except asyncio.TimeoutError:
                                pass
                            except Exception:
                                pass

                            # ── SEND logic with server control ──
                            payload = None
                            with self._lock:
                                if self._pending is not None:
                                    payload = self._pending
                                    self._pending = None

                            if payload:
                                now = time.monotonic()

                                # enforce server interval
                                if now - _last_send_t < self._min_interval:
                                    # too early → skip (client obeys server)
                                    continue

                                _last_send_t = now

                                if self._ws_delay > 0:
                                    await asyncio.sleep(self._ws_delay)

                                await ws.send(json.dumps(payload))
                                self._log_count += 1

                                print(f"[WS] pkt #{self._log_count} | interval={self._min_interval:.3f}s")

                            await asyncio.sleep(0.001)

                except Exception as e:
                    print(e)
                    self._ws = None
                    print(f"[RL] disconnected: {e} — retrying in 2s")
                    await asyncio.sleep(2)

        asyncio.run(_run())
    # ── internal: helpers ─────────────────────────────────────

    def _refresh_plan(self):
        """Periodically refresh plan config to pick up upgrades mid-run."""
        try:
            r = requests.get(f"{self.base}/api/me/plan-config", headers=self.headers, timeout=5)
            if r.ok:
                new_interval = float(r.json().get("log_interval", self._log_interval))
                if new_interval != self._log_interval:
                    print(f"[RunLogger] plan interval updated: {self._log_interval}s → {new_interval}s")
                    self._log_interval = new_interval
        except Exception:
            pass

    def _clear_pause(self):
        """Tell server the client has acknowledged the pause."""
        try:
            requests.delete(f"{self.base}/api/runs/pause/{self.run_id}", headers=self.headers, timeout=3)
            self._pause_flag = False
        except Exception:
            pass