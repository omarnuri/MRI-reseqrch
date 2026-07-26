"""Command line entry point.

    python -m spinelab run  --dicom /content/drive/MyDrive/mri/study.zip \
                            --cache /content/drive/MyDrive/spinelab-cache
    python -m spinelab run  --only marrow,posterior,report --force marrow
    python -m spinelab audit --dicom /path/to/dicom_dir
    python -m spinelab deid  --dicom /path/to/dicom_dir --out /path/to/anon

One command, resumable, no cell ordering to remember.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_STAGES, Config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spinelab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the pipeline")
    run_p.add_argument("--dicom", required=True, help="DICOM directory or .zip")
    run_p.add_argument("--work", default="/content/spine_work", help="workspace directory")
    run_p.add_argument("--cache", default=None,
                       help="persistent weights cache (put this on Google Drive)")
    run_p.add_argument("--subject", default="anon", help="subject id used in outputs")
    run_p.add_argument("--only", default=None,
                       help=f"comma-separated subset of: {','.join(DEFAULT_STAGES)}")
    run_p.add_argument("--skip", default=None, help="comma-separated stages to leave out")
    run_p.add_argument("--force", default=None,
                       help="comma-separated stages to re-run even if cached")
    run_p.add_argument("--no-gpu", action="store_true", help="allow running without a GPU")
    run_p.add_argument("--quality", action="store_true",
                       help="spend GPU time on reliability: independent cross-check model "
                            "and mirror TTA (for an A100-class GPU)")
    run_p.add_argument("--tta-mirror", action="store_true",
                       help="with --quality: re-run segmentation mirrored to test whether the "
                            "model's left/right assignment is stable")

    audit_p = sub.add_parser("audit", help="report identifying tags in a DICOM study")
    audit_p.add_argument("--dicom", required=True)
    audit_p.add_argument("--sample", type=int, default=40)

    deid_p = sub.add_parser("deid", help="write a de-identified copy of a DICOM study")
    deid_p.add_argument("--dicom", required=True)
    deid_p.add_argument("--out", required=True)

    insp_p = sub.add_parser("inspect", help="list the series in a study (no GPU, no dcm2niix)")
    insp_p.add_argument("--dicom", required=True, help="DICOM directory or .zip")
    insp_p.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    sub.add_parser("stages", help="list pipeline stages in order")
    return parser


def _stage_list(only: str | None, skip: str | None) -> tuple[str, ...]:
    stages = tuple(s.strip() for s in only.split(",") if s.strip()) if only else DEFAULT_STAGES
    if skip:
        drop = {s.strip() for s in skip.split(",") if s.strip()}
        stages = tuple(s for s in stages if s not in drop)
    return stages


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "stages":
        for i, name in enumerate(DEFAULT_STAGES, start=1):
            print(f"{i:2d}. {name}")
        return 0

    if args.command == "audit":
        from .dicom_audit import audit

        print(json.dumps(audit(args.dicom, sample=args.sample), indent=2, ensure_ascii=False))
        return 0

    if args.command == "inspect":
        from .inspect_study import format_table, inspect

        report = inspect(args.dicom)
        print(json.dumps(report, indent=2, ensure_ascii=False) if args.json
              else format_table(report))
        return 0

    if args.command == "deid":
        from .dicom_audit import deidentify

        print(json.dumps(deidentify(args.dicom, args.out), indent=2, ensure_ascii=False))
        return 0

    config = Config(
        dicom_source=args.dicom,
        subject_id=args.subject,
        work_dir=Path(args.work),
        cache_dir=Path(args.cache) if args.cache else None,
        stages=_stage_list(args.only, args.skip),
        force=tuple(s.strip() for s in (args.force or "").split(",") if s.strip()),
        gpu_required=not args.no_gpu,
        quality=args.quality,
        tta_mirror=args.tta_mirror,
    )
    results = run_pipeline(config)
    failed = [n for n, r in results.items() if r.status.value == "failed"]
    print(f"\nresults: {config.results_dir}")
    print(f"report:  {config.results_dir / 'report.html'}")
    if failed:
        print(f"failed stages: {', '.join(failed)}")

    summary = json.loads((config.results_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("aborted"):
        # A run that never read a study must not exit 0: that is how "it completed
        # and found nothing" gets mistaken for a result.
        print(f"ABORTED: {summary['aborted']}")
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
