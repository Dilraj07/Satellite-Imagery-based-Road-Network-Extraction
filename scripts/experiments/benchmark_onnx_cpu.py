"""Benchmark exported ONNX models on CPU with ONNX Runtime (5 warmup + 20 timed runs, 4 threads).

Matches the Intel Core i5-1035G4 benchmarking protocol in Section IV-C and Table I
of the manuscript, requiring only numpy and onnxruntime (no PyTorch).

Usage:
    python scripts/experiments/benchmark_onnx_cpu.py --dir Outputs/onnx
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import time

import numpy as np
import onnxruntime as ort

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def cpu_name():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
    return platform.processor() or "unknown CPU"


def onnx_size_mb(path):
    total = os.path.getsize(path)
    if os.path.exists(path + ".data"):
        total += os.path.getsize(path + ".data")
    return total / (2 ** 20)


def time_model(path, tile=1024, warmup=5, runs=20, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal((1, 3, tile, tile), dtype=np.float32)

    def _run_once(arr):
        t0 = time.perf_counter()
        sess.run(None, {inp: arr})
        return time.perf_counter() - t0

    mode = f"{tile}x{tile}"
    try:
        for _ in range(warmup):
            _run_once(x)
        times = [_run_once(x) for _ in range(runs)]
    except Exception:
        # Memory fallback for heavy models (e.g. U-Net) timed as four 512x512 crops.
        half = tile // 2
        crops = [np.ascontiguousarray(x[..., i:i + half, j:j + half]) for i in (0, half) for j in (0, half)]
        for _ in range(warmup):
            for c in crops:
                _run_once(c)
        times = [sum(_run_once(c) for c in crops) for _ in range(runs)]
        mode = f"4x{half}x{half}"

    return {
        "file": os.path.basename(path),
        "size_mb": round(onnx_size_mb(path), 2),
        "mode": mode,
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "min_s": round(min(times), 4),
        "max_s": round(max(times), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "Outputs", "onnx"), help="Directory of exported .onnx files")
    ap.add_argument("--out", default=os.path.join(REPO, "Outputs", "onnx_benchmark_cpu.json"))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    paths = []
    if os.path.isdir(args.dir):
        paths.extend(os.path.join(args.dir, f) for f in sorted(os.listdir(args.dir)) if f.endswith(".onnx"))
    v4_onnx = os.path.join(REPO, "models", "mobilevit_v2.onnx")
    if os.path.isfile(v4_onnx) and not any(os.path.basename(p) in ("v4.onnx", "mobilevit_v2.onnx") for p in paths):
        paths.insert(0, v4_onnx)

    if not paths:
        raise FileNotFoundError(f"No .onnx files found in {args.dir} or {v4_onnx}")

    results = {
        "cpu": cpu_name(),
        "onnxruntime": ort.__version__,
        "threads": args.threads,
        "warmup": args.warmup,
        "runs": args.runs,
        "models": {},
    }
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        print(f"Benchmarking {name} ({p})...", flush=True)
        res = time_model(p, warmup=args.warmup, runs=args.runs, threads=args.threads)
        results["models"][name] = res
        print(" ", res)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
