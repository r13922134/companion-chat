from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .storage import (
    claim_next_depression_job,
    finish_depression_job,
    heartbeat_depression_job,
    heartbeat_depression_worker,
    initialize_database,
    resolve_database_path,
    update_depression_worker,
    update_realtime_session_run_depression,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATHS = (PROJECT_ROOT / ".env", Path(__file__).resolve().parent / ".env")


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_dotenv_files() -> None:
    for path in DOTENV_PATHS:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            if name and name not in os.environ:
                os.environ[name] = _strip_env_value(value)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name) or default).strip())
    except (TypeError, ValueError):
        return default


def safe_gpu_id(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned or "default"


@contextmanager
def acquire_gpu_process_lock(gpu_id: str):
    lock_dir = PROJECT_ROOT / "data"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"depression-worker-gpu-{safe_gpu_id(gpu_id)}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"A depression worker already owns GPU lock {lock_path}."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "gpu_id": gpu_id,
                    "started_at": now_iso(),
                },
                ensure_ascii=True,
            )
        )
        handle.flush()
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class JobHeartbeat:
    def __init__(
        self,
        *,
        database_path: Path,
        job_id: int,
        worker_id: str,
        lease_seconds: float,
    ) -> None:
        self.database_path = database_path
        self.job_id = int(job_id)
        self.worker_id = worker_id
        self.lease_seconds = max(30.0, float(lease_seconds))
        self.interval = max(5.0, min(30.0, self.lease_seconds / 3.0))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"depression-job-heartbeat-{job_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval + 2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if not heartbeat_depression_job(
                    self.database_path,
                    self.job_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                ):
                    return
                heartbeat_depression_worker(
                    self.database_path,
                    worker_id=self.worker_id,
                    status="running",
                )
            except Exception as exc:
                print(
                    f"[DEPRESSION-WORKER] Heartbeat failed | job={self.job_id} "
                    f"| error={exc}",
                    flush=True,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent single-GPU depression inference worker."
    )
    parser.add_argument(
        "--gpu-id",
        default=str(os.environ.get("DEPRESSION_GPU_ID") or "0"),
        help="Physical GPU selector used for CUDA_VISIBLE_DEVICES and process lock.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=env_float("DEPRESSION_WORKER_POLL_SECONDS", 2.0),
    )
    parser.add_argument(
        "--lease-seconds",
        type=float,
        default=env_float("DEPRESSION_WORKER_LEASE_SECONDS", 300.0),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one queued job and exit.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Load models lazily. Intended only for diagnostics.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU inference. Production workers require CUDA by default.",
    )
    return parser


def main() -> int:
    load_dotenv_files()
    args = build_parser().parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    visible_gpu_id = str(os.environ.get("CUDA_VISIBLE_DEVICES") or args.gpu_id)

    import torch

    if not args.allow_cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable for the depression worker. Refusing silent CPU "
            f"fallback. torch={torch.__version__} torch_cuda={torch.version.cuda}. "
            "Install the project-pinned CUDA wheel or pass --allow-cpu only for "
            "diagnostics."
        )

    from .depression_detector import get_detector, run_depression_detection_job

    database_path = resolve_database_path(PROJECT_ROOT)
    initialize_database(database_path)
    hostname = socket.gethostname()
    started_at = now_iso()
    worker_id = (
        f"{hostname}:{os.getpid()}:gpu-{safe_gpu_id(visible_gpu_id)}:"
        f"{uuid4().hex[:8]}"
    )
    stop_event = threading.Event()

    def request_stop(signum, _frame) -> None:
        print(
            f"[DEPRESSION-WORKER] Stop requested | signal={signum}",
            flush=True,
        )
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    with acquire_gpu_process_lock(visible_gpu_id) as lock_path:
        update_depression_worker(
            database_path,
            worker_id=worker_id,
            gpu_id=visible_gpu_id,
            pid=os.getpid(),
            hostname=hostname,
            status="starting",
            started_at=started_at,
            details={"lock_path": str(lock_path)},
        )
        try:
            warmup = {"status": "skipped"}
            if not args.no_warmup:
                print("[DEPRESSION-WORKER] Loading and warming models...", flush=True)
                warmup_started = time.monotonic()
                warmup = get_detector().warm_up()
                warmup["elapsed_seconds"] = round(
                    time.monotonic() - warmup_started,
                    3,
                )
                print(
                    "[DEPRESSION-WORKER] Models ready | "
                    + json.dumps(warmup, ensure_ascii=True),
                    flush=True,
                )
            update_depression_worker(
                database_path,
                worker_id=worker_id,
                gpu_id=visible_gpu_id,
                pid=os.getpid(),
                hostname=hostname,
                status="ready",
                started_at=started_at,
                details={"lock_path": str(lock_path), "warmup": warmup},
            )

            while not stop_event.is_set():
                update_depression_worker(
                    database_path,
                    worker_id=worker_id,
                    gpu_id=visible_gpu_id,
                    pid=os.getpid(),
                    hostname=hostname,
                    status="ready",
                    started_at=started_at,
                )
                job = claim_next_depression_job(
                    database_path,
                    worker_id=worker_id,
                    lease_seconds=args.lease_seconds,
                )
                if job is None:
                    if args.once:
                        break
                    stop_event.wait(max(0.1, float(args.poll_seconds)))
                    continue

                job_id = int(job["id"])
                session_hash = str(job["session_hash"])
                run_id = str(job["run_id"])
                session_dir = Path(str(job["session_dir"]))
                print(
                    f"[DEPRESSION-WORKER] Claimed | job={job_id} "
                    f"| session={session_hash} | run={run_id} "
                    f"| attempt={job['attempts']}/{job['max_attempts']}",
                    flush=True,
                )
                update_depression_worker(
                    database_path,
                    worker_id=worker_id,
                    gpu_id=visible_gpu_id,
                    pid=os.getpid(),
                    hostname=hostname,
                    status="running",
                    started_at=started_at,
                    details={
                        "job_id": job_id,
                        "session_hash": session_hash,
                        "run_id": run_id,
                        "warmup": warmup,
                    },
                )
                heartbeat = JobHeartbeat(
                    database_path=database_path,
                    job_id=job_id,
                    worker_id=worker_id,
                    lease_seconds=args.lease_seconds,
                )
                heartbeat.start()
                job_started = time.monotonic()
                try:
                    result = run_depression_detection_job(
                        database_path,
                        session_hash,
                        run_id,
                        session_dir,
                    )
                    succeeded = result.get("status") == "ok"
                    finish_depression_job(
                        database_path,
                        job_id,
                        worker_id=worker_id,
                        status="completed" if succeeded else "error",
                        error=None if succeeded else str(result.get("error") or ""),
                    )
                    print(
                        f"[DEPRESSION-WORKER] Finished | job={job_id} "
                        f"| status={'completed' if succeeded else 'error'} "
                        f"| elapsed={time.monotonic() - job_started:.3f}s",
                        flush=True,
                    )
                except Exception as exc:
                    completed_at = now_iso()
                    update_realtime_session_run_depression(
                        database_path,
                        session_hash,
                        run_id,
                        {
                            "status": "error",
                            "error": str(exc),
                            "completed_at": completed_at,
                            "result": {
                                "status": "error",
                                "error": str(exc),
                                "completed_at": completed_at,
                            },
                        },
                    )
                    finish_depression_job(
                        database_path,
                        job_id,
                        worker_id=worker_id,
                        status="error",
                        error=str(exc),
                    )
                    print(
                        f"[DEPRESSION-WORKER] Job crashed | job={job_id} "
                        f"| error={exc}",
                        flush=True,
                    )
                finally:
                    heartbeat.stop()

                if args.once:
                    break
        finally:
            update_depression_worker(
                database_path,
                worker_id=worker_id,
                gpu_id=visible_gpu_id,
                pid=os.getpid(),
                hostname=hostname,
                status="stopped",
                started_at=started_at,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
