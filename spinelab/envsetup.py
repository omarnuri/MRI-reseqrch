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

#: The part of the segmentation stack that resolves cleanly. SPINEPS is *not* here;
#: see the acvl-utils note below for why it has to be installed separately.
SEGMENTATION_PACKAGES = ("nnunetv2>=2.8.1", "totalspineseg>=20260623",
                         "TotalSegmentator>=2.16.0")
SEGMENTATION_MODULES = ("nnunetv2", "spineps", "totalspineseg", "totalsegmentator")

#: acvl-utils ships **no wheel** — every release on PyPI is an sdist — so pip has to
#: build it, and it must be installed before anything that depends on it.
BUILD_PREREQUISITES = ("setuptools", "wheel")

#: The dependency conflict that broke every install attempt, in full:
#:
#:     Because spineps>=2.0.0 depends on acvl-utils==0.2 and nnunetv2>=2.8.1 depends
#:     on acvl-utils>=0.2.6,<0.3, we can conclude that nnunetv2>=2.8.1 and
#:     spineps>=2.0.0 are incompatible.
#:
#: SPINEPS 2.0.0 pins `acvl-utils==0.2` exactly, while nnU-Net has required >=0.2.3
#: since 2.6.0 and >=0.2.6 since 2.7.0. A correct resolver therefore cannot put
#: SPINEPS 2.0.0 next to any nnU-Net newer than 2.5.2 — and SPINEPS itself asks for
#: `nnunetv2>=2.4.2,<3.0.0`, so the pin contradicts its own stack.
#:
#: The pin is stale, not real. SPINEPS imports exactly three names from acvl_utils
#: (ACVL_SYMBOLS below), all present in 0.2.6 with unchanged signatures. So we
#: install 0.2.6, install SPINEPS with --no-deps, supply its other dependencies
#: ourselves, and then **verify those three imports**. If upstream ever makes the pin
#: real, that check fails by name instead of the pipeline crashing mid-segmentation.
ACVL_UTILS = "acvl-utils==0.2.6"
SPINEPS_PACKAGE = "SPINEPS==2.0.0"

#: SPINEPS 2.0.0's requires_dist, verbatim, minus acvl-utils. Kept explicit because
#: --no-deps means pip will not read it for us; a new SPINEPS release needs this list
#: refreshed, which `spinelab setup` reports as a missing module rather than hiding.
SPINEPS_DEPS = (
    "TPTBox", "TypeSaveArgParse>=1.0.1,<2.0.0", "antspyx==0.4.2",
    "einops>=0.6.1,<0.7.0", "monai>=1.3.0,<2.0.0", "nnunetv2>=2.4.2,<3.0.0",
    "pytorch-lightning>=2.0.8,<3.0.0", "rich>=13.6.0,<14.0.0",
    "torchmetrics>=1.1.2,<2.0.0", "tqdm>=4.66.1,<5.0.0",
)

#: What SPINEPS actually uses from acvl_utils — the evidence that the pin is stale.
ACVL_SYMBOLS = (
    ("acvl_utils.cropping_and_padding.bounding_boxes",
     ("get_bbox_from_mask", "bounding_box_to_slice")),
    ("acvl_utils.cropping_and_padding.padding", ("pad_nd_image",)),
)

#: nnU-Net excludes torch 2.9.* (`torch!=2.9.*,>=2.1.2`). On a host that ships 2.9.x
#: — some Colab images did — pip silently replaces torch, which is a multi-GB
#: download on a metered line. Worth saying out loud before it happens.
TORCH_EXCLUDED_PREFIX = "2.9."

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


def acvl_symbols() -> list[str]:
    """Which of the three names SPINEPS needs from acvl_utils are missing.

    Empty list means the stale `acvl-utils==0.2` pin really is only stale. A
    non-empty one means overriding it is no longer safe, and says which name broke.
    """
    missing = []
    for module, names in ACVL_SYMBOLS:
        try:
            loaded = importlib.import_module(module)
        except Exception:  # noqa: BLE001 — absent module is absent symbols
            missing += [f"{module}.{n}" for n in names]
            continue
        missing += [f"{module}.{n}" for n in names if not hasattr(loaded, n)]
    return missing


#: How to ask each CLI to prove it can start. Cheap, and it catches the failure
#: mode that --no-deps introduces: an importable package whose entry point dies on
#: a dependency we forgot to list.
SMOKE_COMMANDS = {"spineps": ["--help"], "totalspineseg": ["--help"], "dcm2niix": ["-h"]}


def smoke_test(binaries: dict[str, str | None]) -> dict[str, str]:
    """Start each CLI and see whether it gets as far as printing its usage.

    Exit status alone is not the test — dcm2niix answers `-h` with a non-zero
    status — so a Python traceback in the output is what counts as broken.
    """
    out = {}
    for name, path in binaries.items():
        if path is None:
            continue
        try:
            proc = subprocess.run([path, *SMOKE_COMMANDS.get(name, ["--help"])],
                                  capture_output=True, text=True, timeout=180)
        except Exception as exc:  # noqa: BLE001
            out[name] = f"did not start: {type(exc).__name__}: {exc}"[:160]
            continue
        text = (proc.stdout or "") + (proc.stderr or "")
        if "Traceback (most recent call last)" in text:
            last = [line for line in text.strip().splitlines() if line.strip()][-1]
            out[name] = f"starts but crashes: {last}"[:200]
        else:
            out[name] = "ok"
    return out


def check(segmentation: bool = True, *, smoke: bool = False) -> dict:
    """What is present, what is missing. Never installs anything."""
    probed = ("torch", "numpy") + required_modules(segmentation) + RECOMMENDED_MODULES
    modules = {m: _probe(m) for m in dict.fromkeys(probed)}
    binaries = {b: shutil.which(b) for b in BINARIES}
    missing = [m for m in required_modules(segmentation) if not modules[m][0]]
    broken = acvl_symbols() if segmentation and modules["spineps"][0] else []
    smoke_results = smoke_test(binaries) if smoke else {}
    crashing = [name for name, verdict in smoke_results.items() if verdict != "ok"]
    return {
        "modules": {m: {"importable": ok, "detail": detail} for m, (ok, detail) in modules.items()},
        "binaries": binaries,
        "missing_modules": missing,
        "missing_binaries": [b for b, path in binaries.items()
                             if path is None and (b == "dcm2niix" or segmentation)],
        "missing_recommended": [m for m in RECOMMENDED_MODULES if not modules[m][0]],
        "missing_acvl_symbols": broken,
        "smoke": smoke_results,
        "crashing_binaries": crashing,
        "ready": not missing and not broken and not crashing,
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


def _pip(packages, log, *, force: bool = False, no_deps: bool = False) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if force:
        cmd += ["--upgrade", "--force-reinstall"]
    if no_deps:
        cmd.append("--no-deps")
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
            _warn_about_torch(state, log)

            log("python: build prerequisites")
            _pip(BUILD_PREREQUISITES, log)
            log(f"python: {ACVL_UTILS} (source-only, built here so its errors are visible)")
            if not _pip([ACVL_UTILS], log, force=force):
                log("   ! acvl-utils could not be built — nothing below can install"
                    " without it, and the output above is the reason")

            log("python: nnU-Net, TotalSpineSeg, TotalSegmentator (~5 min)")
            if not _pip(SEGMENTATION_PACKAGES, log, force=force):
                # One unresolvable dependency must not leave the whole stack
                # uninstalled, and the failing package has to be named.
                log("   combined install failed — retrying one by one")
                for package in SEGMENTATION_PACKAGES:
                    _pip([package], log, force=force)

            # SPINEPS separately, and deliberately: its `acvl-utils==0.2` pin
            # contradicts every nnU-Net it claims to support, so pip is not allowed
            # to read it. See the ACVL_UTILS comment for the full reasoning.
            log(f"python: {SPINEPS_PACKAGE} without its stale acvl-utils pin")
            _pip([SPINEPS_PACKAGE], log, force=force, no_deps=True)
            _pip(SPINEPS_DEPS, log, force=force)

        state = check(segmentation, smoke=segmentation)

    log("")
    for module, info in state["modules"].items():
        mark = " " if info["importable"] else "!"
        log(f" {mark} {module:18s} "
            f"{info['detail'] if info['importable'] else 'NOT IMPORTABLE — ' + info['detail']}")
    for binary, path in state["binaries"].items():
        verdict = state["smoke"].get(binary)
        detail = path or "NOT ON PATH"
        if verdict and verdict != "ok":
            detail = f"{path} — {verdict}"
        log(f" {' ' if path and verdict != 'starts but crashes' else '!'} {binary:18s} {detail}")
    log(f"   GPU: {state['gpu']}")

    if state["crashing_binaries"]:
        log("")
        log("=" * 70)
        log(f"These CLIs are installed but cannot start: "
            f"{', '.join(state['crashing_binaries'])}")
        for name in state["crashing_binaries"]:
            log(f"    {name}: {state['smoke'][name]}")
        log("  A missing import here usually means SPINEPS_DEPS is out of date —")
        log("  SPINEPS is installed with --no-deps, so that list is the only thing")
        log("  supplying its dependencies. Add the named package to it.")
        log("=" * 70)

    if state["missing_acvl_symbols"]:
        log("")
        log("=" * 70)
        log("SPINEPS is installed but acvl_utils no longer provides what it imports:")
        for name in state["missing_acvl_symbols"]:
            log(f"    {name}")
        log(f"  The `acvl-utils==0.2` pin we override is therefore no longer only stale.")
        log(f"  Pin nnunetv2==2.5.2 (the newest that accepts acvl-utils 0.2) or check")
        log("  whether a newer SPINEPS release dropped the pin.")
        log("=" * 70)

    if state["missing_modules"]:
        log("")
        log("=" * 70)
        log(f"NOT READY: {', '.join(state['missing_modules'])} cannot be imported.")
        log("  1) Runtime -> Restart session (keeps the VM, the packages and /content),")
        log("     then run this again — a fresh install often only needs a new kernel.")
        log("  2) If it still fails, the pip output above names the reason.")
        log("  Do not start the pipeline: every stage would skip and the report be empty.")
        log("=" * 70)
    elif state["ready"]:
        log("\nready — the pipeline can run")
    return state


def _warn_about_torch(state: dict, log) -> None:
    """Say it before pip does it: an excluded torch means a multi-GB replacement."""
    info = state["modules"].get("torch") or {}
    version = info.get("detail", "")
    if info.get("importable") and version.startswith(TORCH_EXCLUDED_PREFIX):
        log(f"   ! torch {version} is installed, and nnU-Net excludes torch"
            f" {TORCH_EXCLUDED_PREFIX}* — pip will replace it (a large download)")
