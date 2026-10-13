"""Public CLI; environment flags precede model-library imports."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run pinned Pythia-14m residual patches and write an offline report.")
    parser.add_argument("--output", type=Path, default=Path("activation-report"))
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/activation-intervention-lab")
    parser.add_argument("--offline", action="store_true", help="Require an already cached pinned checkpoint")
    parser.add_argument("--patch-batch-size", type=int, default=4)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                        help="FP64 is the validated demo default; FP32 numerical controls can fail")
    args = parser.parse_args()
    if args.patch_batch_size < 1 or not 1 <= args.benchmark_repeats <= 10 or args.threads < 1:
        parser.error("patch batch size and threads must be positive; benchmark repeats must be 1..10")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("output already exists; choose a new directory to preserve earlier results")
    os.environ["HF_HOME"] = str(args.cache_dir.resolve())
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_TOKEN", None)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from .experiment import run_experiment
    from .report import render_report
    json_path, html_path = args.output / "report.json", args.output / "report.html"
    try:
        report = run_experiment(patch_batch_size=args.patch_batch_size, repeats=args.benchmark_repeats,
                                threads=args.threads, dtype=args.dtype)
    except Exception as exc:
        failure = getattr(exc, "partial_report", {"failure": {"type": type(exc).__name__, "message": str(exc)}})
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False), encoding="utf-8")
        raise
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    html_path.write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path.resolve()), "html": str(html_path.resolve()),
                      "conditions": len(report["conditions"]), "remaining_hooks": report["remaining_hooks"],
                      "cache_released": report["cache_released"]}, indent=2))


if __name__ == "__main__":
    main()
