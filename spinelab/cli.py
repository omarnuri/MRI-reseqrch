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
    run_p.add_argument("--dicom", default="",
                       help="DICOM directory, .zip or URL. Omit to let the pipeline "
                            "find the study (Drive, /content, or the repository)")
    run_p.add_argument("--work", default="/content/spine_work", help="workspace directory")
    run_p.add_argument("--cache", default=None,
                       help="persistent weights cache (put this on Google Drive)")
    run_p.add_argument("--subject", default="anon", help="subject id used in outputs")
    run_p.add_argument("--only", default=None,
                       help=f"comma-separated subset of: {','.join(DEFAULT_STAGES)}")
    run_p.add_argument("--skip", default=None, help="comma-separated stages to leave out")
    run_p.add_argument("--force", default=None,
                       help="comma-separated stages to re-run even if cached")
    run_p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                       help="where the segmentation models run. cpu is ~10x slower but "
                            "makes the whole pipeline runnable off a GPU host")
    run_p.add_argument("--no-gpu", action="store_true",
                       help="alias for --device cpu (this flag previously did nothing)")
    run_p.add_argument("--timeout", type=int, default=None,
                       help="per-segmentation-tool timeout in seconds (default 2400; "
                            "raise it for --device cpu)")
    run_p.add_argument("--quality", action="store_true",
                       help="spend GPU time on reliability: independent cross-check model "
                            "and mirror TTA (for an A100-class GPU)")
    run_p.add_argument("--tta-mirror", action="store_true",
                       help="with --quality: re-run segmentation mirrored to test whether the "
                            "model's left/right assignment is stable")

    audit_p = sub.add_parser("audit", help="report identifying tags in a DICOM study")
    audit_p.add_argument("--dicom", required=True)
    audit_p.add_argument("--sample", type=int, default=40)

    sub.add_parser("find", help="show which study would be used, and how it was found")

    setup_p = sub.add_parser("setup", help="install and verify the tools the pipeline needs")
    setup_p.add_argument("--check-only", action="store_true", help="report, change nothing")
    setup_p.add_argument("--force", action="store_true", help="reinstall even if importable")
    setup_p.add_argument("--no-segmentation", action="store_true",
                        help="core I/O only (no GPU stack)")
    setup_p.add_argument("--no-apt", action="store_true", help="skip apt-get")

    deid_p = sub.add_parser("deid", help="write a de-identified copy of a DICOM study")
    deid_p.add_argument("--dicom", required=True)
    deid_p.add_argument("--out", required=True)

    insp_p = sub.add_parser("inspect", help="list the series in a study (no GPU, no dcm2niix)")
    insp_p.add_argument("--dicom", required=True, help="DICOM directory or .zip")
    insp_p.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    diag_p = sub.add_parser("diagnose", help="paste-sized digest of the last run")
    diag_p.add_argument("--work", default="/content/spine_work", help="workspace directory")
    diag_p.add_argument("--log-tail", type=int, default=40,
                       help="how many trailing log lines to include (0 for none)")
    diag_p.add_argument("--numbers", action="store_true",
                       help="also print the key measurements from findings.json")

    sub.add_parser("stages", help="list pipeline stages in order")
    return parser


def _key_numbers(work: Path) -> str:
    """The measurements worth comparing between runs, flattened for pasting."""
    findings = json.loads((Path(work) / "results" / "findings.json").read_text(encoding="utf-8"))
    data = {k: (v.get("data") or {}) for k, v in findings.get("stages", {}).items()}
    lines = ["", "=== key numbers ==="]

    def add(label: str, value):
        lines.append(f"{label:34s} {value}")

    ingest = data.get("ingest", {})
    for series in ingest.get("series", []):
        if not series.get("localizer"):
            add(f"series {series.get('description','?')[:22]}",
                f"{series.get('sequence_label')} {series.get('plane')} "
                f"{series.get('n_slices')} slices")
    qc = data.get("fatsat_qc", {})
    add("fat suppression effective", qc.get("suppression_effective"))
    for name, target in (data.get("register", {}).get("targets") or {}).items():
        reg = target.get("registration", {})
        add(f"register {name}", f"applied={target.get('applied')} "
                                f"{reg.get('translation_magnitude_mm')} mm / "
                                f"{reg.get('rotation_deg')} deg")
    mirror = (data.get("spineps", {}).get("mirror_consistency") or {})
    add("mirror TTA min dice", (mirror.get("side_label_agreement") or {}).get("min_dice"))
    add("crosscheck mean dice", data.get("crosscheck", {}).get("mean_dice"))
    add("crosscheck weak levels", data.get("crosscheck", {}).get("levels_needing_visual_check"))
    geom = data.get("geometry", {})
    add("levels measured", geom.get("levels_measured"))
    add("max wedge angle", geom.get("max_wedge_angle_deg"))
    add("scheuermann pattern", geom.get("scheuermann_pattern"))
    for v in geom.get("per_vertebra", []):
        add(f"  wedge {v.get('name')}", f"{v.get('wedge_angle_deg')} deg "
                                        f"(ant {v.get('anterior_height_mm')} / "
                                        f"post {v.get('posterior_height_mm')} mm)")
    for joint in data.get("facets_axial", {}).get("joints", []):
        sides = joint.get("sides", {})
        add(f"  facet {joint.get('joint')}",
            f"L {(sides.get('left') or {}).get('bright_fraction')} / "
            f"R {(sides.get('right') or {}).get('bright_fraction')} "
            f"comparable={joint.get('comparable')}")
    for group, entry in (data.get("posterior", {}).get("groups") or {}).items():
        cmp_ = entry.get("bright_fraction_comparison") or {}
        add(f"posterior {group}", f"{entry.get('status')} ratio={cmp_.get('ratio')} "
                                  f"higher={cmp_.get('higher_side')}")
    add("marrow candidates", data.get("marrow", {}).get("top_candidates"))
    for muscle, m in (data.get("muscles", {}).get("muscles") or {}).items():
        add(f"muscle {muscle}", (m.get("comparison") or {}).get("diff_pct"))
    canal = data.get("canal", {})
    add("canal median / p10 / min mm2", f"{canal.get('median_area_mm2')} / "
                                        f"{canal.get('p10_area_mm2')} / "
                                        f"{canal.get('min_area_mm2')}")
    add("canal max narrowing %", canal.get("max_narrowing_pct"))
    add("cord dice (spineps vs tss)", data.get("agreement", {}).get("dice"))
    return "\n".join(lines)


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

    if args.command == "setup":
        from .envsetup import check, install

        segmentation = not args.no_segmentation
        if args.check_only:
            state = check(segmentation)
            for module, info in state["modules"].items():
                print(f" {' ' if info['importable'] else '!'} {module:18s} "
                      f"{info['detail']}")
            for binary, path in state["binaries"].items():
                print(f" {' ' if path else '!'} {binary:18s} {path or 'NOT ON PATH'}")
            print(f"   GPU: {state['gpu']}")
            print("ready" if state["ready"] else f"missing: {', '.join(state['missing_modules'])}")
        else:
            state = install(segmentation=segmentation, force=args.force, apt=not args.no_apt)
        return 0 if state["ready"] else 1

    if args.command == "find":
        from .discover import discover

        found = discover(None)
        print(f"source : {found.source or '(not found)'}")
        print(f"how    : {found.how}")
        for candidate in found.candidates:
            print(f"  candidate: {candidate}")
        return 0 if found.source else 1

    if args.command == "diagnose":
        from .runlog import digest

        print(digest(args.work, log_tail=args.log_tail))
        if args.numbers:
            try:
                print(_key_numbers(Path(args.work)))
            except Exception as exc:  # noqa: BLE001
                print(f"\n(no findings.json to read: {exc})")
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

    timeouts = ({"timeout_spineps_s": args.timeout, "timeout_tss_s": args.timeout,
                 "timeout_ts_s": args.timeout} if args.timeout else {})
    config = Config(
        dicom_source=args.dicom,
        subject_id=args.subject,
        work_dir=Path(args.work),
        cache_dir=Path(args.cache) if args.cache else None,
        stages=_stage_list(args.only, args.skip),
        force=tuple(s.strip() for s in (args.force or "").split(",") if s.strip()),
        device="cpu" if args.no_gpu else args.device,
        quality=args.quality,
        tta_mirror=args.tta_mirror,
        **timeouts,
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
