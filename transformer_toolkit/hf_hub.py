# hf_hub.py

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import json
import copy
import queue
import shutil
import threading
import torch
from pathlib import Path
from dataclasses import asdict
from .colors import C, _section, _info, _ok, _err


# ─── login ────────────────────────────────────────────────────────────────────

def login(token: str = None, username: str = None, password: str = None) -> bool:
    try:
        from huggingface_hub import login as hf_login, HfApi
    except ImportError:
        _err("pip install huggingface_hub")
        return False

    _section("🤗 HuggingFace Login")

    try:
        import hf_transfer
        _ok("hf_transfer active — fast uploads enabled")
    except ImportError:
        print(f"  {C.YELLOW}⚠  pip install hf-transfer for faster uploads{C.RESET}")

    if token:
        _info("method", "token")
        try:
            hf_login(token=token, add_to_git_credential=False)
            _ok(f"logged in as {C.CYAN}{C.BOLD}{HfApi().whoami()['name']}{C.RESET}")
            return True
        except Exception as e:
            _err(f"failed — {e}")
            return False

    if username and password:
        _info("method", "username + password")
        try:
            import requests
            resp = requests.post(
                "https://huggingface.co/api/login",
                json={"username": username, "password": password}
            )
            if resp.status_code != 200:
                _err(f"failed — {resp.json().get('error', resp.status_code)}")
                return False
            hf_login(token=resp.json().get("token"), add_to_git_credential=False)
            _ok(f"logged in as {C.CYAN}{C.BOLD}{username}{C.RESET}")
            return True
        except Exception as e:
            _err(f"failed — {e}")
            return False

    _err("provide token= or username= + password=")
    return False


# ─── push ─────────────────────────────────────────────────────────────────────

def push_to_hub(
    repo_id:     str,
    model_state: dict            = None,
    model:       torch.nn.Module = None,
    cfg                          = None,
    tokenizer                    = None,
    metrics:     dict            = None,
    step:        int             = None,
    private:     bool            = True,
    tmp_dir:     str             = "tmp_hf_push",
    optimizer_state: dict = None,
    scaler_state:    dict = None,
    val_loss:        float = None,
    silent: bool = False,
):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        _err("pip install huggingface_hub")
        return
    def _print(*a, **kw):
        if not silent: print(*a, **kw)

    state = model_state or (model.state_dict() if model else None)
    if state is None:
        _err("provide model= or model_state=")
        return

    _section(f"🚀 Pushing → {repo_id}")

    api = HfApi()
    tmp = Path(tmp_dir)
    tmp.mkdir(exist_ok=True)

    try:
        create_repo(repo_id, private=private, exist_ok=True)
        _ok(f"repo ready")
    except Exception as e:
        _err(f"could not create repo — {e}")
        return

    # ── write files to tmp ────────────────────────────────
    _write_weights(state, tmp, optimizer_state=optimizer_state, scaler_state=scaler_state, step=step, val_loss=val_loss)
    _write_config(cfg,        tmp)
    _write_tokenizer(tokenizer, tmp)
    _write_metrics(metrics, step, tmp)
    _write_card(repo_id, cfg, metrics, step, tmp)

    # ── upload ────────────────────────────────────────────
    commit_msg = f"step {step}" if step else "push"
    print(f"\n  {C.YELLOW}⏳ uploading...{C.RESET}", flush=True)
    try:
        api.upload_folder(folder_path=str(tmp), repo_id=repo_id, commit_message=commit_msg)
        _ok(f"pushed → {C.BLUE}https://huggingface.co/{repo_id}{C.RESET}")
    except Exception as e:
        _err(f"upload_folder failed — {e}, trying file-by-file")
        _upload_files(api, tmp, repo_id, commit_msg)

    shutil.rmtree(tmp, ignore_errors=True)



# ─── pull ─────────────────────────────────────────────────────────────────────

def pull_from_hub(repo_id: str, save_dir: str = "checkpoints"):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _err("pip install huggingface_hub")
        return

    _section(f"⬇️  Pulling ← {repo_id}")
    Path(save_dir).mkdir(exist_ok=True)

    for fname in ["model.pt", "checkpoint.pt", "tokenizer.json", "config.json", "metrics.json"]:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=save_dir)
            _ok(f"{fname:<20} → {path}")
        except Exception:
            _info(fname, "not found — skipping")

    print(f"\n  {C.DIM}resume: trainer.train(resume_from='{save_dir}/checkpoint.pt'){C.RESET}\n")


# ─── background worker ────────────────────────────────────────────────────────

class HFSyncWorker:
    def __init__(self):
        self._queue  = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            job = self._queue.get()
            if job is None: break
            try:
                push_to_hub(**job)
            except Exception as e:
                _err(f"background push failed — {e}")
            finally:
                self._queue.task_done()

    def push(self, model: torch.nn.Module, optimizer=None, scaler=None, val_loss=None, **kwargs):
        kwargs["model_state"] = copy.deepcopy(model.state_dict())
        if optimizer is not None:
            kwargs["optimizer_state"] = copy.deepcopy(optimizer.state_dict())
        if scaler is not None:
            kwargs["scaler_state"] = copy.deepcopy(scaler.state_dict())
        if val_loss is not None:
            kwargs["val_loss"] = val_loss
        self._queue.put(kwargs)

    def wait(self):
        if not self._queue.empty():
            print(f"  {C.YELLOW}⏳ waiting for uploads...{C.RESET}")
            self._queue.join()
            _ok("all uploads done")

    def shutdown(self):
        self.wait()
        self._queue.put(None)
        self._thread.join()


# ─── helpers ──────────────────────────────────────────────────────────────────

def _write_weights(state, tmp, optimizer_state=None, scaler_state=None, step=None, val_loss=None):
    # lightweight weights-only file (for inference)
    path = tmp / "model.pt"
    torch.save(state, str(path))
    _info("weights", f"model.pt  {path.stat().st_size/1e6:.1f} MB")

    # full resume checkpoint
    ckpt = {"model_state_dict": state}
    if optimizer_state is not None:
        ckpt["optimizer_state_dict"] = optimizer_state
    if scaler_state is not None:
        ckpt["scaler_state_dict"] = scaler_state
    if step is not None:
        ckpt["step"] = step
    if val_loss is not None:
        ckpt["val_loss"] = val_loss

    ckpt_path = tmp / "checkpoint.pt"
    torch.save(ckpt, str(ckpt_path))
    _info("checkpoint", f"checkpoint.pt  {ckpt_path.stat().st_size/1e6:.1f} MB")
    
def _write_config(cfg, tmp):
    if cfg is None: return
    try:
        d = asdict(cfg)
    except TypeError:
        d = cfg.__dict__
    d = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v for k, v in d.items()}
    (tmp / "config.json").write_text(json.dumps(d, indent=2))
    _info("config", "config.json")


def _write_tokenizer(tokenizer, tmp):
    if tokenizer is None: return
    try:
        tokenizer.save(str(tmp / "tokenizer.json"))
        _info("tokenizer", "tokenizer.json")
    except Exception as e:
        _err(f"tokenizer save failed — {e}")


def _write_metrics(metrics, step, tmp):
    if metrics is None: return
    m = dict(metrics)
    if step: m["step"] = step
    (tmp / "metrics.json").write_text(json.dumps(m, indent=2))
    _info("metrics", str(m))


def _write_card(repo_id, cfg, metrics, step, tmp):
    try:
        d = asdict(cfg)
    except Exception:
        d = cfg.__dict__ if cfg else {}
    d = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v for k, v in d.items()}

    lines = [
        f"# {repo_id.split('/')[-1]}",
        f"",
        f"Trained with **transformer-toolkit**.",
        f"",
        f"## Architecture",
        f"| param | value |", f"|---|---|",
        *[f"| `{k}` | `{v}` |" for k, v in d.items()],
    ]
    if metrics:
        lines += [
            f"", f"## Metrics",
            f"| metric | value |", f"|---|---|",
            *[f"| `{k}` | `{v}` |" for k, v in metrics.items()],
        ]
    (tmp / "README.md").write_text("\n".join(lines))


def _upload_files(api, tmp, repo_id, commit_msg):
    import time
    for path in tmp.iterdir():
        for attempt in range(3):
            try:
                api.upload_file(
                    path_or_fileobj = str(path),
                    path_in_repo    = path.name,
                    repo_id         = repo_id,
                    commit_message  = commit_msg,
                )
                _ok(f"{path.name}")
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    _err(f"gave up on {path.name} — {e}")