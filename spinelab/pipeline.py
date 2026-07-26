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

from . import __version__
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

    def masks_in_fatsat_space(self) -> dict | None:
        """Masks already resampled onto the fat-suppressed grid by `register`.

        Returns None when that stage did not run, in which case callers fall back
        to header-based resampling. The distinction matters: at facet scale a
        couple of millimetres of inter-series motion moves the region of interest
        off the joint, so whether alignment was refined has to stay visible.
        """
        data = self.stage_data("register")
        if data.get("space") == "fatsat" and data.get("instance_mask"):
            return data
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
        agreement, canal, crosscheck, discs, geometry, ingest, marrow, muscles,
        posterior, radiomics, register, report, seg_spineps,
        seg_totalsegmentator, seg_totalspineseg,
    )

    return {
        "ingest": ingest.run,
        "spineps": seg_spineps.run,
        "totalspineseg": seg_totalspineseg.run,
        "totalsegmentator": seg_totalsegmentator.run,
        "register": register.run,
        "crosscheck": crosscheck.run,
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

    registry = _registry()
    ctx = Context(config=config)
    unknown = [s for s in config.stages if s not in registry]
    if unknown:
        raise ValueError(f"unknown stage(s): {', '.join(unknown)}")

    for name in config.stages:
        marker = config.stage_dir / f"{name}.json"
        if marker.exists() and name not in config.force and name != "report":
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
        t0 = time.time()
        try:
            result = registry[name](ctx)
        except SkipStage as exc:
            result = StageResult(name=name, status=Status.SKIPPED, reason=str(exc))
        except Exception as exc:  # noqa: BLE001 — a broken stage must not kill the run
            result = StageResult(
                name=name,
                status=Status.FAILED,
                reason=f"{type(exc).__name__}: {exc}"[:400],
                data={"traceback": traceback.format_exc()[-2000:]},
            )
        result.duration_s = time.time() - t0
        ctx.results[name] = result
        write_json(marker, result.to_dict())

        icon = {"ok": "OK", "partial": "PARTIAL", "skipped": "SKIP",
                "failed": "FAIL", "cached": "CACHED"}[result.status.value]
        log(f"[{name:17s}] {icon} ({human_duration(result.duration_s)})"
            + (f" — {result.reason}" if result.reason else ""))

    write_json(config.results_dir / "summary.json", summarise(ctx))
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
