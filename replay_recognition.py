from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bot import DesktopGame, ROOT, load_config
from recognition import RecognitionEngine, ViewportCalibrator, normalize_frame


DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "recognition" / "manifest.json"


def read_image(path: Path) -> Any:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取回放图像：{path}")
    return image


def build_matcher(config: dict[str, Any]) -> DesktopGame:
    game = DesktopGame.__new__(DesktopGame)
    game.config = config
    game.cv2 = cv2
    game.np = np
    game.templates = game._load_templates()
    game.multi_anchor_templates = game._load_multi_anchor_templates()
    return game


def replay_sample(
    sample: dict[str, Any],
    fixture_dir: Path,
    config: dict[str, Any],
    matcher: DesktopGame,
    engine: RecognitionEngine | None = None,
) -> dict[str, Any]:
    raw = read_image(fixture_dir / sample["file"])
    size = (raw.shape[1], raw.shape[0])

    online_calibrator = ViewportCalibrator(config, cv2, np)
    captures = iter([raw.copy() for _ in range(online_calibrator.frame_count)])
    online_raw, calibration, recalibrated = online_calibrator.ensure(
        size, lambda: next(captures)
    )
    online = normalize_frame(
        online_raw,
        calibration,
        config["reference_size"],
        config.get("reference_content_top", 0),
        cv2,
        np,
    )
    offline = normalize_frame(
        raw,
        calibration,
        config["reference_size"],
        config.get("reference_content_top", 0),
        cv2,
        np,
    )
    if not np.array_equal(online, offline):
        raise AssertionError(f"{sample['file']} 在线/离线标准化像素不一致")

    online_page, online_scores = matcher.detect_page(online)
    offline_page, offline_scores = matcher.detect_page(offline)
    if online_scores != offline_scores:
        raise AssertionError(f"{sample['file']} 在线/离线模板分数不一致")

    engine = engine or RecognitionEngine(config, cv2, np)
    observation = engine.observe(offline, template_scores=offline_scores)
    return {
        "file": sample["file"],
        "source_size": list(size),
        "content_top": calibration.content_top,
        "calibration_confidence": calibration.confidence,
        "recalibrated": recalibrated,
        "normalized_sha256": hashlib.sha256(offline.tobytes()).hexdigest(),
        "legacy_page": offline_page,
        "legacy_score": round(float(offline_scores.get(offline_page, 0.0)), 6),
        "state": observation.state,
        "state_confidence": observation.state_confidence,
        "numeric_values": dict(sorted(observation.numeric_values.items())),
        "controls": sorted({control.name for control in observation.controls}),
        "control_counts": dict(
            sorted(Counter(control.name for control in observation.controls).items())
        ),
        "control_rects": {
            control.name: list(control.rect) for control in observation.controls
        },
        "control_sources": {
            name: sorted(
                {
                    control.source
                    for control in observation.controls
                    if control.name == name
                }
            )
            for name in sorted({control.name for control in observation.controls})
        },
    }


def run_replay(manifest_path: Path, config_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    matcher = build_matcher(config)
    engine = RecognitionEngine(config, cv2, np)
    fixture_dir = manifest_path.parent
    return [
        replay_sample(sample, fixture_dir, config, matcher, engine)
        for sample in manifest["samples"]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回放真实截图并核对 V2 识别结果")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_replay(args.manifest, args.config)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
