from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bot import ROOT, load_config
from recognition import LocalOCR, clear_ocr_runtime
from replay_recognition import build_matcher, replay_sample


DEFAULT_IMAGE = ROOT / "references" / "tasks.png"
DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "recognition" / "manifest.json"


def read_image(path: Path) -> Any:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取基准图像：{path}")
    return image


def serialize_tokens(tokens):
    return [
        {
            "text": token.text,
            "confidence": round(token.confidence, 6),
            "rect": list(token.rect),
        }
        for token in tokens
    ]


def replay_signature(config, ocr, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matcher = build_matcher(config)
    from recognition import RecognitionEngine

    engine = RecognitionEngine(config, cv2, np, ocr=ocr)
    return [
        {
            "file": sample["file"],
            "state": result["state"],
            "controls": result["controls"],
            "control_counts": result["control_counts"],
        }
        for sample in manifest["samples"]
        for result in [
            replay_sample(sample, manifest_path.parent, config, matcher, engine)
        ]
    ]


def benchmark_backend(base_config, backend, image, roi, iterations, include_replay):
    clear_ocr_runtime(engines=True)
    config = copy.deepcopy(base_config)
    settings = config["recognition_v2"]
    settings["ocr_backend"] = backend
    settings["ocr_cache"] = False
    ocr = LocalOCR(config, np_module=np)

    init_wall = time.perf_counter()
    init_cpu = time.process_time()
    initial_runtime = ocr.runtime_info()
    initialization = {
        "wall_seconds": time.perf_counter() - init_wall,
        "cpu_seconds": time.process_time() - init_cpu,
    }

    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]
    warm_wall = time.perf_counter()
    warm_cpu = time.process_time()
    warm_tokens = ocr.read(crop)
    warmup = {
        "wall_seconds": time.perf_counter() - warm_wall,
        "cpu_seconds": time.process_time() - warm_cpu,
        "tokens": serialize_tokens(warm_tokens),
    }

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    tokens = []
    for _ in range(iterations):
        tokens = ocr.read(crop)
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    result = {
        "backend": backend,
        "runtime": ocr.runtime_info(),
        "initial_runtime": initial_runtime,
        "initialization": initialization,
        "warmup": warmup,
        "measurement": {
            "iterations": iterations,
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "wall_seconds_per_iteration": wall_seconds / iterations,
            "cpu_seconds_per_iteration": cpu_seconds / iterations,
            "tokens": serialize_tokens(tokens),
        },
    }
    if include_replay:
        result["replay_signature"] = replay_signature(
            config, ocr, DEFAULT_MANIFEST
        )
    return result


def comparisons(results):
    baseline = next((item for item in results if item["backend"] == "cpu"), None)
    output = []
    if baseline is None:
        return output
    cpu_measurement = baseline["measurement"]
    for item in results:
        if item is baseline:
            continue
        active = item["runtime"]["active_backend"]
        comparison = {
            "backend": item["backend"],
            "active_backend": active,
            "same_tokens": (
                item["measurement"]["tokens"] == cpu_measurement["tokens"]
            ),
        }
        if "replay_signature" in baseline and "replay_signature" in item:
            comparison["same_replay_signature"] = (
                item["replay_signature"] == baseline["replay_signature"]
            )
        if active in {"cuda", "dml"}:
            comparison.update(
                {
                    "wall_ratio_to_cpu": (
                        item["measurement"]["wall_seconds"]
                        / cpu_measurement["wall_seconds"]
                    ),
                    "cpu_ratio_to_cpu": (
                        item["measurement"]["cpu_seconds"]
                        / cpu_measurement["cpu_seconds"]
                    ),
                }
            )
            comparison["passes_performance_gate"] = (
                comparison["wall_ratio_to_cpu"] <= 1.1
                and comparison["cpu_ratio_to_cpu"] < 1.0
            )
        else:
            comparison["passes_performance_gate"] = None
            comparison["reason"] = "GPU Provider unavailable; backend used CPU fallback"
        output.append(comparison)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="OCR 后端性能与一致性基准")
    parser.add_argument(
        "--backend",
        action="append",
        choices=("auto", "cpu", "cuda", "dml"),
        dest="backends",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--roi", type=int, nargs=4, default=(90, 170, 1091, 565))
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations 必须至少为 1")
    backends = args.backends or ["cpu", "auto"]
    image = read_image(args.image)
    config = load_config(ROOT / "config.json")
    results = [
        benchmark_backend(
            config,
            backend,
            image,
            tuple(args.roi),
            args.iterations,
            args.replay,
        )
        for backend in backends
    ]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "image": str(args.image),
        "roi": list(args.roi),
        "results": results,
        "comparisons": comparisons(results),
    }
    output = args.output or (
        ROOT / "runs" / f"ocr-benchmark-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
