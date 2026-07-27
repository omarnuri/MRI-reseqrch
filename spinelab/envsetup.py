"""Environment setup as code, not as a notebook cell.

A notebook cell is the worst place for this. The cell a user has open in the browser
is whatever they opened; pulling a fix updates the package on disk and leaves that
cell untouched, so a bug in the setup logic survives every update until the notebook
itself is reopened. Exactly that happened here: a cell that wrote its "installed"
marker without checking pip's exit status kept reporting success for a whole session
while every segmentation tool was missing.

So the logic lives here, and the notebook cell is a one-liner that calls it. Pulling
the branch now updates the setup too.

    python -m spinelab setup                 # install what is missing, verify
    python -m spinelab setup --check-only    # report, change nothing
    python -m spinelab setup --force         # reinstall everything
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys

#: Installed for every run. Only `nibabel` and `pydicom` are hard requirements;
#: SimpleITK is what the `register` stage needs, and that stage skips cleanly
#: without it (masks then sit where the DICOM geometry puts them).
CORE_PACKAGES = ("nibabel", "pydicom", "SimpleITK")
CORE_MODULES = ("nibabel", "pydicom")
RECOMMENDED_MODULES = ("SimpleITK",)

#: The segmentation stack. Installed as one command so pip resolves a single
#: consistent set of versions; the old notebook installed nnunetv2 and then
#: force-pinned an older nnunetv2 with --no-deps on top of it.
SEGMENTATION_PACKAGES = ("nnunetv2>=2.8.1", "SPINEPS>=2.0.0",
                         "totalspineseg>=20260623", "TotalSegmentator>=2.16.0")
SEGMENTATION_MODULES = ("nnunetv2", "spineps", "totalspineseg", "totalsegmentator")

#: nnunetv2 requires acvl-utils, which ships **no wheel** — every release on PyPI is
#: an sdist, so pip has to build it. That makes it the most fragile link in the
#: chain, and it is installed on its own first so a build failure is isolated and
#: its output is visible instead of being buried in a four-package resolution.
BUILD_PREREQUISITES = ("setuptools", "wheel")
FRAGILE_PACKAGES = ("acvl-utils>=0.2.6,<0.3",)

APT_PACKAGES = ("dcm2niix", "unzip")
#: CLI entry points the pipeline shells out to.
BINARIES = ("dcm2niix", "spineps", "totalspineseg")


def required_modules(segmentation: bool = True) -> tuple[str, ...]:
    return CORE_MODULES + (SEGMENTATION_MODULES if segmentation else ())


def _probe(module: str) -> tuple[bool, str]:
    """Is the module available, and which version?

    An already-imported module is reported from sys.modules rather than
    re-imported: numpy and other C extensions refuse to load twice in one process
    ("cannot load module more than once per process"), and in Colab numpy is always
    imported before this runs — so purging and re-importing reported the tools as
    broken when they were fine.

    `invalidate_caches` matters for the opposite case: a package pip installed
    moments ago in this same process.
    """
    cached = sys.modules.get(module)
    if cached is not None:
        return True, str(getattr(cached, "__version__", "present"))
    try:
        importlib.invalidate_caches()
        loaded = importlib.import_module(module)
        return True, str(getattr(loaded, "__version__", "present"))
    except Exception as exc:  # noqa: BLE001 — any import failure is a missing tool
        return False, f"{type(exc).__name__}: {exc}"[:140]


def check(segmentation: bool = True) -> dict:
    """What is present, what is missing. Never installs anything."""
    probed = ("torch", "numpy") + required_modules(segmentation) + RECOMMENDED_MODULES
    modules = {m: _probe(m) for m in dict.fromkeys(probed)}
    binaries = {b: shutil.which(b) for b in BINARIES}
    missing = [m for m in required_modules(segmentation) if not modules[m][0]]
    return {
        "modules": {m: {"importable": ok, "detail": detail} for m, (ok, detail) in modules.items()},
        "binaries": binaries,
        "missing_modules": missing,
        "missing_binaries": [b for b, path in binaries.items()
                             if path is None and (b == "dcm2niix" or segmentation)],
        "missing_recommended": [m for m in RECOMMENDED_MODULES if not modules[m][0]],
        "ready": not missing,
        "gpu": _gpu_line(),
    }


def _gpu_line() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "no CUDA device visible"
        props = torch.cuda.get_device_properties(0)
        return f"{props.name} ({props.total_memory / 1e9:.0f} GB)"
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({str(exc)[:60]})"


def _run(cmd: list[str], log) -> bool:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"   ! failed: {' '.join(cmd)[:120]}")
        for line in (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]:
            log(f"     {line[:160]}")
    return proc.returncode == 0


def _pip(packages, log, *, force: bool = False) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if force:
        cmd += ["--upgrade", "--force-reinstall"]
    ok = _run(cmd + list(packages), log)
    log(f"   {'ok' if ok else 'FAILED'}: {' '.join(packages)}")
    return ok


def install(*, segmentation: bool = True, force: bool = False, apt: bool = True,
            log=print) -> dict:
    """Install what is missing and verify by importing. Returns the final check()."""
    state = check(segmentation)
    if state["ready"] and not force:
        log("all required packages already importable — nothing to install")
    else:
        if apt and shutil.which("apt-get"):
            log(f"apt: {', '.join(APT_PACKAGES)}")
            _run(["apt-get", "-qq", "update"], log)
            _run(["apt-get", "-qq", "install", "-y", *APT_PACKAGES], log)

        log("python: core I/O")
        _pip(CORE_PACKAGES, log, force=force)

        if segmentation:
            log("python: build prerequisites")
            _pip(BUILD_PREREQUISITES, log)
            log("python: acvl-utils (source-only, built here so its errors are visible)")
            if not _pip(FRAGILE_PACKAGES, log, force=force):
                log("   ! acvl-utils could not be built — nnunetv2 cannot install without it,"
                    " and the output above is the reason")

            log("python: segmentation stack (~5 min)")
            if not _pip(SEGMENTATION_PACKAGES, log, force=force):
                # One unresolvable dependency must not leave the whole stack
                # uninstalled, and the failing package has to be named.
                log("   combined install failed — retrying one by one")
                for package in SEGMENTATION_PACKAGES:
                    _pip([package], log, force=force)

        state = check(segmentation)

    log("")
    for module, info in state["modules"].items():
        mark = " " if info["importable"] else "!"
        log(f" {mark} {module:18s} "
            f"{info['detail'] if info['importable'] else 'NOT IMPORTABLE — ' + info['detail']}")
    for binary, path in state["binaries"].items():
        log(f" {' ' if path else '!'} {binary:18s} {path or 'NOT ON PATH'}")
    log(f"   GPU: {state['gpu']}")

    if state["missing_modules"]:
        log("")
        log("=" * 70)
        log(f"NOT READY: {', '.join(state['missing_modules'])} cannot be imported.")
        log("  1) Runtime -> Restart session (keeps the VM, the packages and /content),")
        log("     then run this again — a fresh install often only needs a new kernel.")
        log("  2) If it still fails, the pip output above names the reason.")
        log("  Do not start the pipeline: every stage would skip and the report be empty.")
        log("=" * 70)
    else:
        log("\nready — the pipeline can run")
    return state
