from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def collect_chars(text: str) -> list[str]:
    return sorted(set(text), key=ord)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect unique characters from text files into a JSON map."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Input text files (keys are derived from file stems).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: stdout).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input files (default: utf-8).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data: dict[str, list[str]] = {}

    for file_arg in args.files:
        path = Path(file_arg)
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 1

        key = path.stem
        if key in data:
            print(f"error: duplicate key '{key}' from {path}", file=sys.stderr)
            return 1

        try:
            text = path.read_text(encoding=args.encoding)
        except Exception as exc:
            print(f"error: failed to read {path}: {exc}", file=sys.stderr)
            return 1

        data[key] = collect_chars(text)

    output = json.dumps(data, ensure_ascii=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
