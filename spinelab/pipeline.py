"""Stage runner: resumable, order-independent, no shared notebook globals.

The old notebook was one long chain of cells that passed state through module
globals (`seg_files`, `T2_SAG`, `STATUS`, …). Any restart — and Cell 1c
deliberately SIGKILLed the kernel — left half those globals undefined, so later
cells either crashed with NameError or silently analysed the wrong volume.

Here each stage:
  * receives an explicit `Context`,
  * returns a `StageResult`,
  * is persisted to results/stages/<name>.json,
  * and reads its inputs back from those files, not from memory.

That makes a run resumable after a Colab disconnect (the normal case on a slow
connection) and makes each stage unit-testable in isolation.
"""

from __future__ import annotations

import platform
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import __version__, runlog
from .config import Config
from .evidence import Evidence, Status
from .utils import human_duration, read_json, write_json


@dataclass
class StageResult:
    name: str
    status: Status
    evidence: Evidence = Evidence.MEASUREMENT
    reason: str | None = None
    data: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    spinelab_version: str = __version__

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value if isinstance(self.status, Status) else str(self.status),
            "evidence": self.evidence.value if isinstance(self.evidence, Evidence) else str(self.evidence),
            "reason": self.reason,
            "data": self.data,
            "artifacts": [str(a) for a in self.artifacts],
            "duration_s": round(self.duration_s, 2),
            "spinelab_version": self.spinelab_version,
        }

    @property
    def ok(self) -> bool:
        return self.status in (Status.OK, Status.PARTIAL, Status.CACHED)


@dataclass
class Context:
    """Everything a stage may read. No globals, no implicit ordering."""

    config: Config
    results: dict[str, StageResult] = field(default_factory=dict)

    # ---- convenience accessors used by stages ---------------------------
    def stage_data(self, name: str) -> dict:
        """Data of a previous stage, from memory or from its JSON marker."""
        if name in self.results:
            return self.results[name].data
        payload = read_json(self.config.stage_dir / f"{name}.json", {}) or {}
        return payload.get("data", {}) or {}

    def stage_ok(self, name: str) -> bool:
        if name in self.results:
            return self.results[name].ok
        payload = read_json(self.config.stage_dir / f"{name}.json", {}) or {}
        return payload.get("status") in ("ok", "partial", "cached")

    def sequence(self, key: str) -> str | None:
        """Path of a picked sequence ('T2_SAG', 'FATSAT_BEST', ...) or None."""
        picks = self.stage_data("ingest").get("picks", {})
        value = picks.get(key)
        return value if value else None

    def require_sequence(self, key: str) -> str:
        value = self.sequence(key)
        if not value:
            raise SkipStage(f"sequence {key} not available in this study")
        return value

    def path_list(self, stage: str, key: str) -> list[Path]:
        return [Path(p) for p in (self.stage_data(stage).get(key) or [])]

    def masks_in_space(self, target: str) -> dict | None:
        """Masks already resampled onto `target`'s grid by `register`.

        `target` is "fatsat" or "axial". Returns None when that stage did not run
        or produced nothing for this target, in which case callers fall back to
        header-based resampling. The distinction matters: at facet scale a couple
        of millimetres of inter-series motion moves the region of interest off the
        joint, so whether alignment was refined has to stay visible.
        """
        entry = (self.stage_data("register").get("targets") or {}).get(target)
        if entry and entry.get("instance_mask"):
            return entry
        return None


class SkipStage(Exception):
    """Raised by a stage when its inputs are legitimately absent.

    A missing sequence is not a bug and must not be reported as a failure —
    conflating the two is why the old status table was unreadable.
    """


StageFn = Callable[[Context], StageResult]


def _registry() -> dict[str, StageFn]:
    # Imported lazily so `import spinelab.pipeline` stays cheap and so a missing
    # optional dependency only breaks the stage that needs it.
    from .stages import (
        agreement, canal, crosscheck, discs, facets_axial, fatsat_qc, geometry,
        ingest, marrow, muscles, posterior, radiomics, register, report,
        seg_spineps, seg_totalsegmentator, seg_totalspineseg,
    )

    return {
        "ingest": ingest.run,
        "spineps": seg_spineps.run,
        "totalspineseg": seg_totalspineseg.run,
        "totalsegmentator": seg_totalsegmentator.run,
        "register": register.run,
        "fatsat_qc": fatsat_qc.run,
        "crosscheck": crosscheck.run,
        "facets_axial": facets_axial.run,
        "geometry": geometry.run,
        "muscles": muscles.run,
        "marrow": marrow.run,
        "posterior": posterior.run,
        "canal": canal.run,
        "discs": discs.run,
        "radiomics": radiomics.run,
        "agreement": agreement.run,
        "report": report.run,
    }


def run_pipeline(config: Config, log=print) -> dict[str, StageResult]:
    config.ensure_dirs()
    config.export_env()
    write_json(config.results_dir / "run_config.json", {
        "config": config.to_dict(),
        "spinelab_version": __version__,
        "python": platform.python_version(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    log_path = runlog.setup(config.results_dir)
    logger = runlog.get_logger()
    logger.info("run started: stages=%s force=%s", list(config.stages), list(config.force))
    logger.info("dicom_source=%r work_dir=%s cache_dir=%s", config.dicom_source,
                config.work_dir, config.cache_dir)
    runlog.event("run_start", stages=list(config.stages), force=list(config.force),
                 dicom_source=config.dicom_source, work_dir=str(config.work_dir),
                 cache_dir=str(config.cache_dir) if config.cache_dir else None,
                 quality=config.quality, tta_mirror=config.tta_mirror)
    runlog.log_environment({"spinelab_version": __version__})
    # The probe above imports the segmentation stack to read its versions, and
    # `import totalspineseg` overwrites nnUNet_results with a relative path. Reassert
    # our cache locations so in-process readers see them too; child processes are
    # covered separately by utils.child_env.
    config.export_env()
    log(f"log: {log_path}")

    registry = _registry()
    ctx = Context(config=config)
    unknown = [s for s in config.stages if s not in registry]
    if unknown:
        raise ValueError(f"unknown stage(s): {', '.join(unknown)}")

    # A run with no study data at all must not look like a run that found nothing.
    # Without this, a wrong `dicom_source` produces 16 tidy "skipped" lines, zero
    # failures and a full report — which reads as success. It happened.
    aborted: str | None = None

    # Once a stage actually re-runs, every later stage's cache is stale: they all
    # depend, directly or transitively, on what came before. Forcing `ingest` alone
    # would otherwise leave the downstream markers from the previous run in place and
    # mix numbers from two different inventories.
    invalidated = False

    for name in config.stages:
        if aborted and name != "report":
            ctx.results[name] = StageResult(
                name=name, status=Status.SKIPPED,
                reason=f"run aborted before this stage: {aborted}")
            continue

        marker = config.stage_dir / f"{name}.json"
        if marker.exists() and name not in config.force and name != "report" \
                and not invalidated:
            cached = read_json(marker, {}) or {}
            if cached.get("status") in ("ok", "partial", "cached"):
                ctx.results[name] = StageResult(
                    name=name,
                    status=Status.CACHED,
                    evidence=Evidence(cached.get("evidence", "measurement")),
                    reason=f"resumed from {marker.name} (use force to re-run)",
                    data=cached.get("data", {}) or {},
                    artifacts=cached.get("artifacts", []) or [],
                )
                log(f"[{name:17s}] cached — skipping (delete {marker.name} to re-run)")
                continue

        log(f"[{name:17s}] running…")
        logger.info("=== stage %s: start ===", name)
        runlog.event("stage_start", stage=name)
        if not invalidated:
            invalidated = True
            logger.info("cache invalidated from %s onwards (this stage re-ran)", name)
        t0 = time.time()
        try:
            result = registry[name](ctx)
        except SkipStage as exc:
            result = StageResult(name=name, status=Status.SKIPPED, reason=str(exc))
            logger.info("stage %s skipped: %s", name, exc)
        except Exception as exc:  # noqa: BLE001 — a broken stage must not kill the run
            tb = traceback.format_exc()
            result = StageResult(
                name=name,
                status=Status.FAILED,
                reason=f"{type(exc).__name__}: {exc}"[:400],
                data={"traceback": tb[-2000:]},
            )
            logger.error("stage %s FAILED: %s\n%s", name, exc, tb)
            runlog.event("stage_failed", stage=name, error=f"{type(exc).__name__}: {exc}",
                         traceback=tb[-2000:])
        result.duration_s = time.time() - t0
        ctx.results[name] = result
        write_json(marker, result.to_dict())
        logger.info("=== stage %s: %s in %s%s ===", name, result.status.value,
                    human_duration(result.duration_s),
                    f" — {result.reason}" if result.reason else "")
        runlog.event("stage_end", stage=name, status=result.status.value,
                     evidence=result.evidence.value, seconds=round(result.duration_s, 2),
                     reason=result.reason, artifacts=len(result.artifacts),
                     data_keys=sorted(result.data)[:25])

        icon = {"ok": "OK", "partial": "PARTIAL", "skipped": "SKIP",
                "failed": "FAIL", "cached": "CACHED"}[result.status.value]
        log(f"[{name:17s}] {icon} ({human_duration(result.duration_s)})"
            + (f" — {result.reason}" if result.reason else ""))

        if name == "ingest" and not (result.ok and result.data.get("picks")):
            aborted = result.reason or "the study could not be read"
            logger.error("RUN ABORTED — no study data: %s (dicom_source=%r)",
                         aborted, config.dicom_source)
            runlog.event("run_aborted", stage=name, reason=aborted,
                         dicom_source=config.dicom_source)
            log("")
            log("=" * 70)
            log("RUN ABORTED — no study data was read, so nothing below could run.")
            log(f"  reason: {aborted}")
            log(f"  dicom_source was: {config.dicom_source!r}")
            log("  Fix that path (or URL) and run again. Every later stage would")
            log("  otherwise report 'skipped' and the report would look complete.")
            log("=" * 70)

    summary = summarise(ctx)
    summary["aborted"] = aborted
    summary["log"] = str(log_path)
    write_json(config.results_dir / "summary.json", summary)
    logger.info("run finished: %d failed, %d skipped%s", summary["n_failed"],
                len(summary["skipped"]), " (ABORTED)" if aborted else "")
    runlog.event("run_end", failed=summary["failed"], skipped=summary["skipped"],
                 aborted=aborted)
    return ctx.results


def summarise(ctx: Context) -> dict:
    """Machine-readable run summary — what ran, what didn't, and why."""
    stages = {name: res.to_dict() for name, res in ctx.results.items()}
    failed = [n for n, r in ctx.results.items() if r.status is Status.FAILED]
    skipped = [n for n, r in ctx.results.items() if r.status is Status.SKIPPED]
    return {
        "spinelab_version": __version__,
        "subject_id": ctx.config.subject_id,
        "stages": stages,
        "n_failed": len(failed),
        "failed": failed,
        "skipped": skipped,
        "disclaimer": "Research screening output. Not a diagnosis.",
    }
