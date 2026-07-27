"""Colab-side bootstrap: mounting Drive and choosing a weights cache.

Drive is a convenience. On a slow line it keeps ~6–8 GB of model weights across
sessions, which is worth a lot — but it is not a requirement, and a failed mount
must not end the run. One did: the whole first cell aborted on

    ValueError: mount failed

and every later cell never executed, so a Google authentication hiccup read as
"the pipeline is broken".

Two things changed. The mount is advisory now, with a local cache as the fallback.
And the cache directory is created only once the mount is *verified*, because
`mkdir(parents=True)` under an unmounted `/content/drive` is actively harmful: it
leaves a real directory at the mount point, and `drive.mount` then refuses to mount
into a non-empty directory. One failed mount poisoned every retry in the same VM,
which is the kind of thing that looks like bad luck three sessions in a row.

This lives in the package rather than in the notebook cell for the same reason
`envsetup` does: pulling a fix updates the package on disk, never the cell the user
has open in the browser.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Where Colab mounts Drive, and the directory that must exist inside it.
MOUNTPOINT = "/content/drive"
DRIVE_ROOT = "/content/drive/MyDrive"
#: Cache location when Drive is unavailable. Lives as long as the VM does.
LOCAL_CACHE = "/content/spinelab-cache"

WEIGHT_CACHES = ("spineps", "totalsegmentator", "totalspineseg", "huggingface")


def in_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").is_dir()


def drive_is_mounted(mountpoint: str = MOUNTPOINT) -> bool:
    """Is Drive *actually* mounted, as opposed to a directory of the same name?

    The distinction is the whole point: a leftover stub directory looks identical
    to a mount from `Path.exists()`, and treating it as a mount is how weights end
    up written to the VM's own disk while the log claims they went to Drive.
    """
    path = Path(mountpoint)
    if not path.is_dir():
        return False
    try:
        if os.path.ismount(mountpoint):
            return True
    except OSError:
        pass
    # Colab's FUSE mount is not always reported by ismount; MyDrive existing
    # inside it is the practical test.
    return Path(mountpoint, "MyDrive").is_dir()


def stub_at_mountpoint(mountpoint: str = MOUNTPOINT) -> bool:
    """A non-empty real directory sitting where Drive should mount."""
    path = Path(mountpoint)
    if not path.is_dir() or drive_is_mounted(mountpoint):
        return False
    return any(path.iterdir())


def mount_drive(mountpoint: str = MOUNTPOINT, *, log=print) -> tuple[bool, str]:
    """Try to mount Drive. Returns (mounted, reason) and never raises."""
    if drive_is_mounted(mountpoint):
        return True, "already mounted"
    if not in_colab():
        return False, "not running on Colab"

    if stub_at_mountpoint(mountpoint):
        # Report, do not delete: it is the caller's data, even in a throwaway VM.
        log(f"   ! {mountpoint} exists as a plain directory, and Drive will not mount"
            f" into a non-empty one.")
        log(f"     This is usually left over from an earlier failed mount. Clear it with"
            f" `!rm -rf {mountpoint}` and run this cell again.")

    try:
        from google.colab import drive
    except Exception as exc:  # noqa: BLE001
        return False, f"google.colab.drive unavailable ({type(exc).__name__}: {exc})"

    for attempt, kwargs in enumerate(({}, {"force_remount": True}), start=1):
        try:
            drive.mount(mountpoint, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a failed mount is not fatal
            reason = f"{type(exc).__name__}: {exc}"
            log(f"   Drive mount attempt {attempt} failed — {reason}")
            continue
        if drive_is_mounted(mountpoint):
            return True, "mounted"
        log(f"   Drive mount attempt {attempt} returned without error but"
            f" {mountpoint}/MyDrive is not there")
    return False, "mount failed"


def resolve_cache_dir(preferred: str | os.PathLike[str], *,
                      mountpoint: str = MOUNTPOINT,
                      fallback: str = LOCAL_CACHE) -> tuple[Path, str]:
    """Where the weights cache can actually live, and why.

    A path inside an unmounted Drive is not usable, and must not be created — see
    the module docstring. The fallback works; it just does not survive the VM.
    """
    path = Path(preferred)
    if _is_under(path, mountpoint) and not drive_is_mounted(mountpoint):
        return Path(fallback), (
            f"Drive is not mounted, so the cache goes to {fallback} instead."
            " Everything works; model weights are simply re-downloaded if this VM"
            " is recycled.")
    return path, ""


def _is_under(path: Path, parent: str) -> bool:
    try:
        path.resolve().relative_to(Path(parent).resolve())
    except (ValueError, OSError):
        return False
    return True


def cache_sizes(cache_dir: str | os.PathLike[str]) -> dict[str, float]:
    """Bytes already cached per tool, in GB — what a resumed session saves."""
    out = {}
    for name in WEIGHT_CACHES:
        directory = Path(cache_dir) / "weights" / name
        total = 0
        if directory.is_dir():
            total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
        out[name] = total / 1e9
    return out


def prepare(preferred_cache: str | os.PathLike[str] = f"{DRIVE_ROOT}/spinelab-cache",
            *, mount: bool = True, mountpoint: str = MOUNTPOINT,
            fallback: str = LOCAL_CACHE, log=print) -> dict:
    """Mount Drive if possible, pick a usable cache directory, report both."""
    mounted, reason = (mount_drive(mountpoint, log=log) if mount
                       else (False, "mounting disabled"))
    log(f"Drive: {'mounted' if mounted else 'not mounted'} ({reason})")

    cache_dir, note = resolve_cache_dir(preferred_cache, mountpoint=mountpoint,
                                       fallback=fallback)
    if note:
        log(f"   {note}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    log(f"weights cache: {cache_dir}")
    sizes = cache_sizes(cache_dir)
    for name, size_gb in sizes.items():
        log(f"  cached {name:18s} {size_gb:5.2f} GB")
    return {"drive_mounted": mounted, "drive_reason": reason,
            "cache_dir": str(cache_dir), "cache_sizes_gb": sizes}
