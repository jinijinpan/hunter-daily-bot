from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
REFERENCE_SIZE = (1091, 700)
REFERENCE_CONTENT_TOP = 53


def normalize(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if image.size == REFERENCE_SIZE:
        return image
    source_width, source_height = image.size
    content_top = min(REFERENCE_CONTENT_TOP, source_height - 1)
    chrome = image.crop((0, 0, source_width, content_top)).resize(
        (REFERENCE_SIZE[0], REFERENCE_CONTENT_TOP), Image.Resampling.LANCZOS
    )
    content = image.crop((0, content_top, source_width, source_height)).resize(
        (REFERENCE_SIZE[0], REFERENCE_SIZE[1] - REFERENCE_CONTENT_TOP),
        Image.Resampling.LANCZOS,
    )
    normalized = Image.new("RGB", REFERENCE_SIZE)
    normalized.paste(chrome, (0, 0))
    normalized.paste(content, (0, REFERENCE_CONTENT_TOP))
    return normalized


def compress(path: Path, colors: int) -> tuple[int, int, bool]:
    original_size = path.stat().st_size
    with Image.open(path) as source:
        requires_normalization = source.size != REFERENCE_SIZE
        image = normalize(source)
        compressed = image.quantize(
            colors=colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        temporary = path.with_name(f".{path.name}.compressing")
        try:
            compressed.save(temporary, format="PNG", optimize=True, compress_level=9)
            with Image.open(temporary) as check:
                check.verify()
            compressed_size = temporary.stat().st_size
            replaced = requires_normalization or compressed_size < original_size
            if replaced:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return original_size, path.stat().st_size, replaced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="压缩 recordings 中的 PNG 截图")
    parser.add_argument(
        "--directory", type=Path, default=ROOT / "recordings", help="录制目录"
    )
    parser.add_argument(
        "--colors", type=int, default=128, choices=range(16, 257), metavar="16-256"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.directory.rglob("*.png"))
    before = 0
    after = 0
    replaced = 0
    for index, path in enumerate(paths, 1):
        original_size, compressed_size, was_replaced = compress(path, args.colors)
        before += original_size
        after += compressed_size
        replaced += int(was_replaced)
        if index % 50 == 0 or index == len(paths):
            print(f"已处理 {index}/{len(paths)}")
    saved = before - after
    print(
        f"完成：{before / 1024**2:.1f} MB -> {after / 1024**2:.1f} MB，"
        f"节省 {saved / 1024**2:.1f} MB；更新 {replaced}/{len(paths)} 张"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
