"""Command-line interface:  python -m langscout "Ceci est une belle journée"."""
from __future__ import annotations

import argparse
import json
import sys

from .detector import LanguageDetector


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="langscout", description="Detect the language of a text (17 languages).")
    ap.add_argument("text", nargs="*", help="text to analyse (reads stdin if omitted)")
    ap.add_argument("--segments", action="store_true", help="detect the language of every sentence separately")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--top", type=int, default=3, help="number of alternatives to show")
    args = ap.parse_args(argv)

    text = " ".join(args.text) if args.text else sys.stdin.read()
    detector = LanguageDetector()
    preds = detector.detect_segments(text) if args.segments else [detector.detect(text, top_k=args.top)]

    if args.json:
        print(json.dumps([p.as_dict() for p in preds], ensure_ascii=False, indent=2))
        return 0
    for p in preds:
        shown = p.text if len(p.text) <= 60 else p.text[:57] + "..."
        if p.language is None:
            print(f"{shown!r}\n  -> {p.status}: {p.note}")
            continue
        flag = "" if p.is_confident else "  (uncertain)"
        alts = ", ".join(f"{l} {pr:.1%}" for l, pr in p.top_k)
        print(f"{shown!r}\n  -> {p.language} ({p.confidence:.1%}){flag}   [{alts}]")
        if p.note:
            print(f"     note: {p.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
