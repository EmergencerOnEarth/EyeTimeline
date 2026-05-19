#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modality: str
    raw_dir: str
    labels_csv: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backbone: str
    normalization: str
    mae_variant: str
    checkpoint_by_modality: dict[str, str | None]


DATASETS = {
    "aptos2019": DatasetSpec("aptos2019", "cfp", "APTOS2019", "cfp/aptos2019_labels.csv"),
    "idrid": DatasetSpec("idrid", "cfp", "IDRiD_data", "cfp/idrid_labels.csv"),
    "papila": DatasetSpec("papila", "cfp", "PAPILA", "cfp/papila_labels.csv"),
    "glaucoma_fundus": DatasetSpec(
        "glaucoma_fundus", "cfp", "Glaucoma_fundus", "cfp/glaucoma_fundus_labels.csv"
    ),
    "octid": DatasetSpec("octid", "oct", "OCTID", "oct/octid_labels.csv"),
}


def build_model_specs(asset_root: Path) -> dict[str, ModelSpec]:
    bw = asset_root / "baseline_weights"
    ours = asset_root / "our_weights"
    return {
        "ours": ModelSpec(
            name="ours",
            backbone="eyetimeline_mae",
            normalization="imagenet",
            mae_variant="large",
            checkpoint_by_modality={
                "cfp": str(ours / "cfp" / "checkpoint_best.pth"),
                "oct": str(ours / "oct" / "checkpoint_best.pth"),
            },
        ),
        "retfound": ModelSpec(
            name="retfound",
            backbone="retfound_mae",
            normalization="imagenet",
            mae_variant="large",
            checkpoint_by_modality={
                "cfp": str(bw / "retfound" / "RETFound_mae_natureCFP" / "RETFound_mae_natureCFP.pth"),
                "oct": str(bw / "retfound" / "RETFound_mae_natureOCT" / "RETFound_mae_natureOCT.pth"),
            },
        ),
        "eyeclip": ModelSpec(
            name="eyeclip",
            backbone="eyeclip",
            normalization="clip",
            mae_variant="large",
            checkpoint_by_modality={
                "cfp": str(bw / "eyeclip" / "eyeclip_visual.pt"),
                "oct": str(bw / "eyeclip" / "eyeclip_visual.pt"),
            },
        ),
        "imagenet_mae": ModelSpec(
            name="imagenet_mae",
            backbone="imagenet_mae",
            normalization="imagenet",
            mae_variant="large",
            checkpoint_by_modality={
                "cfp": str(bw / "imagenet_mae" / "mae_vit_large_imagenet.bin"),
                "oct": str(bw / "imagenet_mae" / "mae_vit_large_imagenet.bin"),
            },
        ),
        "random_mae": ModelSpec(
            name="random_mae",
            backbone="random_mae",
            normalization="imagenet",
            mae_variant="large",
            checkpoint_by_modality={"cfp": None, "oct": None},
        ),
    }


@dataclass
class Job:
    mode: str
    dataset: DatasetSpec
    model: ModelSpec
    data_root: Path
    labels_csv: Path
    checkpoint: str | None
    output_dir: Path
    log_path: Path

    @property
    def job_id(self) -> str:
        return f"{self.mode}/{self.dataset.modality}/{self.dataset.name}/{self.model.name}"


class Progress:
    def __init__(self, run_dir: Path, total: int) -> None:
        self.run_dir = run_dir
        self.total = total
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.running: dict[str, str | None] = {}
        self.lock = threading.Lock()
        self.progress_jsonl = run_dir / "progress.jsonl"
        self.status_json = run_dir / "status.json"

    def event(self, job: Job, status: str, **extra: Any) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        rec = {"time": now, "job_id": job.job_id, "status": status, **extra}
        with self.lock:
            if status == "running":
                self.running[job.job_id] = extra.get("gpu")
            elif status in {"done", "failed", "skipped"}:
                self.running.pop(job.job_id, None)
                if status == "done":
                    self.completed += 1
                elif status == "failed":
                    self.failed += 1
                elif status == "skipped":
                    self.skipped += 1
            with self.progress_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            status_doc = {
                "updated_at": now,
                "total": self.total,
                "completed": self.completed,
                "failed": self.failed,
                "skipped": self.skipped,
                "remaining": self.total - self.completed - self.failed - self.skipped,
                "running": self.running,
            }
            self.status_json.write_text(json.dumps(status_doc, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run EyeTimeline downstream benchmark matrix")
    p.add_argument("--asset-root", default=os.environ.get("ASSET_ROOT", "/data1/kechuang/EyeTimelineAssets"))
    p.add_argument("--output-root", default=None)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--models", nargs="+", default=["ours", "retfound", "eyeclip"])
    p.add_argument("--modes", nargs="+", choices=["linear", "full"], default=["linear"])
    p.add_argument("--epochs-linear", type=int, default=50)
    p.add_argument("--epochs-full", type=int, default=50)
    p.add_argument("--batch-size-linear", type=int, default=64)
    p.add_argument("--batch-size-full", type=int, default=16)
    p.add_argument("--lr-linear", type=float, default=1e-3)
    p.add_argument("--lr-full", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--pool", choices=["cls", "avg"], default="cls")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpus", default=os.environ.get("BENCHMARK_GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
    p.add_argument("--max-parallel", type=int, default=None)
    p.add_argument("--force", action="store_true", help="Rerun jobs even when eval_test/metrics.json already exists.")
    p.add_argument("--skip-missing", action="store_true", help="Skip jobs with missing data or checkpoints.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def validate_args(args: argparse.Namespace, model_specs: dict[str, ModelSpec]) -> None:
    unknown_datasets = sorted(set(args.datasets) - set(DATASETS))
    unknown_models = sorted(set(args.models) - set(model_specs))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}; available={sorted(DATASETS)}")
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}; available={sorted(model_specs)}")


def make_jobs(args: argparse.Namespace, asset_root: Path, output_root: Path) -> list[Job]:
    model_specs = build_model_specs(asset_root)
    validate_args(args, model_specs)
    jobs: list[Job] = []
    raw_root = asset_root / "benchmark_data" / "raw"
    processed_root = asset_root / "benchmark_data" / "processed"
    logs_root = output_root / "job_logs"

    for mode in args.modes:
        for dataset_name in args.datasets:
            ds = DATASETS[dataset_name]
            for model_name in args.models:
                model = model_specs[model_name]
                checkpoint = model.checkpoint_by_modality[ds.modality]
                out = output_root / mode / ds.modality / ds.name / model.name
                jobs.append(Job(
                    mode=mode,
                    dataset=ds,
                    model=model,
                    data_root=raw_root / ds.raw_dir,
                    labels_csv=processed_root / ds.labels_csv,
                    checkpoint=checkpoint,
                    output_dir=out,
                    log_path=logs_root / mode / ds.modality / ds.name / f"{model.name}.log",
                ))
    return jobs


def _check_job_inputs(job: Job) -> list[str]:
    missing: list[str] = []
    if not job.data_root.exists():
        missing.append(str(job.data_root))
    if not job.labels_csv.exists():
        missing.append(str(job.labels_csv))
    if job.checkpoint and not Path(job.checkpoint).exists():
        missing.append(job.checkpoint)
    return missing


def _train_cmd(job: Job, args: argparse.Namespace) -> list[str]:
    epochs = args.epochs_linear if job.mode == "linear" else args.epochs_full
    batch_size = args.batch_size_linear if job.mode == "linear" else args.batch_size_full
    lr = args.lr_linear if job.mode == "linear" else args.lr_full
    cmd = [
        sys.executable,
        "benchmarks/train_classifier.py",
        "--labels-csv", str(job.labels_csv),
        "--data-root", str(job.data_root),
        "--backbone", job.model.backbone,
        "--mae-variant", job.model.mae_variant,
        "--normalization", job.model.normalization,
        "--pool", args.pool,
        "--img-size", str(args.img_size),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--weight-decay", str(args.weight_decay),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output-dir", str(job.output_dir),
    ]
    if job.checkpoint:
        cmd += ["--checkpoint", job.checkpoint]
    if job.mode == "linear":
        cmd += ["--freeze-encoder", "--save-head-only"]
    return cmd


def _eval_cmd(job: Job, args: argparse.Namespace) -> list[str]:
    batch_size = args.batch_size_linear if job.mode == "linear" else args.batch_size_full
    return [
        sys.executable,
        "benchmarks/evaluate_classifier.py",
        "--labels-csv", str(job.labels_csv),
        "--data-root", str(job.data_root),
        "--split", "test",
        "--checkpoint", str(job.output_dir / "checkpoint_best.pth"),
        "--batch-size", str(batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--output-dir", str(job.output_dir / "eval_test"),
    ]


def _run_command(cmd: list[str], log_file, env: dict[str, str]) -> int:
    log_file.write(f"\n$ {shlex.join(cmd)}\n")
    log_file.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.wait()


def run_job(job: Job, args: argparse.Namespace, progress: Progress, gpu: str | None) -> None:
    metrics_path = job.output_dir / "eval_test" / "metrics.json"
    if metrics_path.exists() and not args.force:
        progress.event(job, "skipped", reason="metrics_exists")
        return

    missing = _check_job_inputs(job)
    if missing:
        msg = f"Missing inputs: {missing}"
        if args.skip_missing:
            progress.event(job, "skipped", reason=msg)
            return
        progress.event(job, "failed", reason=msg)
        raise FileNotFoundError(msg)

    progress.event(job, "running", gpu=gpu, log=str(job.log_path))
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    if args.dry_run:
        with job.log_path.open("w", encoding="utf-8") as f:
            f.write(f"[DryRun] {job.job_id}\n")
            f.write(shlex.join(_train_cmd(job, args)) + "\n")
            f.write(shlex.join(_eval_cmd(job, args)) + "\n")
        progress.event(job, "done", dry_run=True)
        return

    started = time.time()
    with job.log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"[Job] {job.job_id}\n")
        log_file.write(f"[GPU] {gpu}\n")
        log_file.write(f"[Started] {datetime.now().isoformat(timespec='seconds')}\n")
        code = _run_command(_train_cmd(job, args), log_file, env)
        if code != 0:
            progress.event(job, "failed", stage="train", exit_code=code, log=str(job.log_path))
            raise RuntimeError(f"{job.job_id} train failed: {code}")
        code = _run_command(_eval_cmd(job, args), log_file, env)
        if code != 0:
            progress.event(job, "failed", stage="eval", exit_code=code, log=str(job.log_path))
            raise RuntimeError(f"{job.job_id} eval failed: {code}")
        log_file.write(f"[Finished] {datetime.now().isoformat(timespec='seconds')}\n")
    progress.event(job, "done", seconds=round(time.time() - started, 2), log=str(job.log_path))


def worker_loop(
    q: queue.Queue[Job],
    args: argparse.Namespace,
    progress: Progress,
    gpu: str | None,
    failures: list[str],
    failure_lock: threading.Lock,
) -> None:
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(job, args, progress, gpu)
        except Exception as exc:
            with failure_lock:
                failures.append(f"{job.job_id}: {exc}")
        finally:
            q.task_done()


def write_manifest(run_dir: Path, args: argparse.Namespace, jobs: list[Job], output_root: Path) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "args": vars(args),
        "jobs": [
            {
                "job_id": job.job_id,
                "data_root": str(job.data_root),
                "labels_csv": str(job.labels_csv),
                "checkpoint": job.checkpoint,
                "output_dir": str(job.output_dir),
                "log_path": str(job.log_path),
            }
            for job in jobs
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    asset_root = Path(args.asset_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else asset_root / "benchmark_outputs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / run_id
    latest_path = output_root / "latest_run.txt"
    output_root.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir), encoding="utf-8")

    jobs = make_jobs(args, asset_root, output_root)
    write_manifest(run_dir, args, jobs, output_root)
    progress = Progress(run_dir, total=len(jobs))
    progress.status_json.write_text(json.dumps({
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(jobs),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "remaining": len(jobs),
        "running": {},
    }, indent=2), encoding="utf-8")

    gpus = _split_csv(args.gpus) if args.gpus else [None]
    if args.max_parallel is not None:
        worker_count = max(1, args.max_parallel)
    else:
        worker_count = max(1, len(gpus))
    print(f"[Run] output_root={output_root}")
    print(f"[Run] run_dir={run_dir}")
    print(f"[Run] jobs={len(jobs)} workers={worker_count} gpus={gpus}")

    q: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        q.put(job)

    failures: list[str] = []
    failure_lock = threading.Lock()
    threads = []
    for i in range(worker_count):
        gpu = gpus[i % len(gpus)] if gpus else None
        t = threading.Thread(target=worker_loop, args=(q, args, progress, gpu, failures, failure_lock), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    summary_cmd = [
        sys.executable,
        "benchmarks/summarize_results.py",
        "--output-root", str(output_root),
        "--summary-csv", str(output_root / "summary.csv"),
        "--summary-md", str(output_root / "summary.md"),
    ]
    subprocess.run(summary_cmd, cwd=REPO_ROOT, check=False)
    if failures:
        print("[Run] Failures:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("[Run] Done")


if __name__ == "__main__":
    main()
